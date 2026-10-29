use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::lsmr::fixtures::{jacobi, vandermonde_ls, IdentityOp};
use crate::{
    lsmr, lsmr_cancellable, mlsmr, mlsmr_cancellable, Cancellation, MlsmrOptions, Operator,
    SolveError,
};

#[test]
fn every_iteration_checkpoint_and_terminal_return_can_cancel() {
    let (a, b) = vandermonde_ls();
    let m = jacobi(&a);
    let warm = mlsmr(&a, &b, &m, 1e-12, 3, MlsmrOptions::default()).unwrap();
    for preconditioned in [false, true] {
        for x0 in [None, Some(&warm.x[..])] {
            let polls = AtomicUsize::new(0);
            let count = || {
                polls.fetch_add(1, Ordering::Relaxed);
                false
            };
            let run = |cancellation: Cancellation<'_>| {
                if preconditioned {
                    mlsmr_cancellable(
                        &a,
                        &b,
                        &m,
                        1e-12,
                        20,
                        MlsmrOptions {
                            warm_start: x0,
                            ..Default::default()
                        },
                        cancellation,
                    )
                } else {
                    lsmr_cancellable(&a, &b, 1e-12, 20, None, cancellation)
                }
            };
            let ordinary = run(Cancellation::default()).unwrap();
            let observed = run(Cancellation::new(&count)).unwrap();
            assert_eq!(ordinary.x, observed.x);
            assert_eq!(ordinary.iterations, observed.iterations);
            assert_eq!(ordinary.stop_reason, observed.stop_reason);
            assert_eq!(
                ordinary.residual_norm.to_bits(),
                observed.residual_norm.to_bits()
            );
            assert_eq!(
                ordinary.normal_eq_residual.to_bits(),
                observed.normal_eq_residual.to_bits()
            );
            assert_eq!(ordinary.true_residual, observed.true_residual);
            let count = polls.load(Ordering::Relaxed);
            assert_eq!(count, observed.iterations + 2);
            for stop in 1..=count {
                let polls = AtomicUsize::new(0);
                let query = || polls.fetch_add(1, Ordering::Relaxed) + 1 >= stop;
                assert!(
                    matches!(run(Cancellation::new(&query)), Err(SolveError::Cancelled)),
                    "stop={stop}"
                );
            }
        }
    }
}

#[test]
fn cancelled_zero_rhs_and_zero_budget_return_no_estimate() {
    let query = || true;
    let cancellation = Cancellation::new(&query);
    let identity = IdentityOp { n: 2 };
    for b in [&[0.0, 0.0][..], &[1.0, 2.0][..]] {
        assert!(matches!(
            lsmr_cancellable(&identity, b, 1e-8, 0, None, cancellation),
            Err(SolveError::Cancelled)
        ));
        assert!(matches!(
            mlsmr_cancellable(
                &identity,
                b,
                &identity,
                1e-8,
                0,
                MlsmrOptions::default(),
                cancellation,
            ),
            Err(SolveError::Cancelled)
        ));
    }
    assert!(
        lsmr(&identity, &[0.0, 0.0], 1e-8, 0, None)
            .unwrap()
            .converged
    );
}

#[test]
fn cancellation_during_a_failing_operator_is_not_a_numerical_failure() {
    struct Failing<'a>(&'a AtomicBool);
    impl Operator for Failing<'_> {
        fn nrows(&self) -> usize {
            2
        }
        fn ncols(&self) -> usize {
            2
        }
        fn apply(&self, _: &[f64], _: &mut [f64]) -> Result<(), SolveError> {
            self.0.store(true, Ordering::Relaxed);
            Err(SolveError::InvalidInput {
                context: "test operator",
                message: "failed".into(),
            })
        }
        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            self.apply(x, y)
        }
    }
    let cancelled = AtomicBool::new(false);
    let query = || cancelled.load(Ordering::Relaxed);
    let a = Failing(&cancelled);
    assert!(matches!(
        lsmr_cancellable(&a, &[1.0, 2.0], 1e-8, 20, None, Cancellation::new(&query)),
        Err(SolveError::Cancelled)
    ));
    cancelled.store(false, Ordering::Relaxed);
    assert!(matches!(
        mlsmr_cancellable(
            &a,
            &[1.0, 2.0],
            &IdentityOp { n: 2 },
            1e-8,
            20,
            MlsmrOptions::default(),
            Cancellation::new(&query),
        ),
        Err(SolveError::Cancelled)
    ));
}
