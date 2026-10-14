//! Warm starts and escalation: the preconditioner ladder.

use super::super::*;
use crate::lsmr::fixtures::*;
use crate::{Operator, SolveError};

#[test]
fn test_mlsmr_warm_tolerance_is_relative_to_original_rhs() {
    let op = DenseOp::vandermonde(30, 12);
    let x_true: Vec<f64> = (0..op.cols).map(|j| 1.0 / (1.0 + j as f64)).collect();
    let mut b = vec![0.0; op.rows];
    op.apply(&x_true, &mut b).expect("apply");
    let x0: Vec<f64> = x_true.iter().map(|v| v * 0.9999).collect();
    let tol = 1e-6;
    let window = Some(12);
    let m = jacobi(&op);
    let base = MlsmrOptions {
        local_size: window,
        ..Default::default()
    };
    let warm = mlsmr(
        &op,
        &b,
        &m,
        tol,
        200,
        MlsmrOptions {
            warm_start: Some(&x0),
            ..base
        },
    )
    .expect("warm solve");
    assert_eq!(warm.stop_reason, LsmrStopReason::ResidualTolerance);

    let mut r = vec![0.0; op.rows];
    op.apply(&x0, &mut r).expect("apply");
    for (ri, bi) in r.iter_mut().zip(&b) {
        *ri = bi - *ri;
    }
    let naive = mlsmr(&op, &r, &m, tol, 200, base).expect("naive restart");
    assert!(
        warm.iterations < naive.iterations,
        "warm took {} iters, naive restart {} — the tolerance basis is not `‖b‖`",
        warm.iterations,
        naive.iterations
    );
}

/// A zero RHS must not shadow warm-start validation: the same bad `x0` has to be
/// rejected whether or not `b` short-circuits the solve.
#[test]
fn test_mlsmr_rejects_bad_warm_start_for_any_rhs() {
    let op = IdentityOp { n: 3 };
    for b in [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]] {
        for x0 in [&[0.0, 0.0][..], &[0.0, f64::NAN, 0.0]] {
            let options = MlsmrOptions {
                warm_start: Some(x0),
                ..Default::default()
            };
            assert!(matches!(
                mlsmr(&op, &b, &op, 1e-8, 10, options),
                Err(SolveError::InvalidInput { .. })
            ));
        }
    }
}

/// Regression: `A·x₀` overflowing made `b - A·x₀` norm to NaN, which `init` read
/// as β₁ = 0 and reported as a converged solve with `x = x₀` and a NaN residual.
#[test]
fn test_mlsmr_warm_rejects_overflowing_warm_start_residual() {
    let op = DiagOp(vec![1e200; 3]);
    let m = IdentityOp { n: 3 };
    let x0 = [1e200; 3];
    let options = MlsmrOptions {
        warm_start: Some(&x0),
        ..Default::default()
    };
    assert!(matches!(
        mlsmr(&op, &[1.0, 2.0, 3.0], &m, 1e-8, 10, options),
        Err(SolveError::InvalidInput { .. })
    ));
}

#[test]
fn test_mlsmr_zero_rhs_corrects_non_exact_warm_start() {
    let op = IdentityOp { n: 3 };
    let x0 = [1.0, 2.0, 3.0];
    let options = MlsmrOptions {
        warm_start: Some(&x0),
        ..Default::default()
    };
    let result = mlsmr(&op, &[0.0, 0.0, 0.0], &op, 1e-10, 10, options).expect("warm correction");

    assert!(result.converged);
    assert!(vec_norm(&result.x) < 1e-12);
    assert!(result.iterations > 0);
}

#[test]
fn test_mlsmr_exact_warm_start_is_returned_untouched() {
    let cases: [(&dyn Operator, &[f64], &[f64]); 2] = [
        (&IdentityOp { n: 3 }, &[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]),
        (&ZeroSecondRow, &[0.0, 0.0], &[0.0, 7.0]),
    ];
    for (op, b, x0) in cases {
        let options = MlsmrOptions {
            warm_start: Some(x0),
            ..Default::default()
        };
        let m = IdentityOp { n: op.ncols() };
        let result = mlsmr(op, b, &m, 1e-10, 50, options).expect("exact warm start");
        assert_eq!(result.stop_reason, LsmrStopReason::WarmStartExact);
        assert!(result.converged);
        assert_eq!(result.x, x0);
        assert_eq!(result.iterations, 0);
    }
}

#[test]
fn test_mlsmr_options_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<MlsmrOptions<'static>>();
}

#[test]
fn test_mlsmr_converged_solve_is_never_escalated() {
    let opts = MlsmrOptions {
        escalation: Some(&FixedIterations(1)),
        ..Default::default()
    };
    let m = IdentityOp { n: 3 };
    let result = mlsmr(&IdentityOp { n: 3 }, &[1.0, 2.0, 3.0], &m, 1e-10, 50, opts).expect("solve");
    assert!(result.converged);
    assert_ne!(result.stop_reason, LsmrStopReason::Escalated);
}

#[test]
fn test_mlsmr_ladder_warm_starts_and_escalates() {
    let op = DenseOp::vandermonde(30, 12);
    let b: Vec<f64> = (0..op.rows)
        .map(|i| (1.0 + i as f64 / (op.rows - 1) as f64).ln())
        .collect();
    let (tol, window) = (1e-9, Some(12));
    let weak = IdentityOp { n: op.cols };
    let strong = jacobi(&op);
    let rung = MlsmrOptions {
        escalation: Some(&FixedIterations(3)),
        local_size: window,
        ..Default::default()
    };
    let final_rung = MlsmrOptions {
        local_size: window,
        ..Default::default()
    };

    let first = mlsmr(&op, &b, &weak, tol, 200, rung).expect("first rung");
    assert_eq!(first.stop_reason, LsmrStopReason::Escalated);

    let middle = mlsmr(
        &op,
        &b,
        &weak,
        tol,
        200,
        MlsmrOptions {
            warm_start: Some(&first.x),
            ..rung
        },
    )
    .expect("middle rung");
    assert_eq!(middle.stop_reason, LsmrStopReason::Escalated);

    let last = mlsmr(
        &op,
        &b,
        &strong,
        tol,
        200,
        MlsmrOptions {
            warm_start: Some(&middle.x),
            ..final_rung
        },
    )
    .expect("last rung");
    assert!(last.converged);

    let cold = mlsmr(&op, &b, &strong, tol, 200, final_rung).expect("cold solve");
    let ladder = normal_equation_residual(&op, &last.x, &b);
    assert!(
        ladder <= 10.0 * normal_equation_residual(&op, &cold.x, &b).max(1e-14),
        "ladder landed short of a cold solve: {ladder}"
    );
}

#[test]
fn test_staleness_escalates_only_the_stalling_preconditioner() {
    let rule = Staleness::try_new(4, 0.7).expect("valid staleness policy");
    let progress = |iteration, normal_eq_residual| Progress {
        iteration,
        normal_eq_residual,
    };
    let mut stalled = rule.handler();
    for (iteration, residual) in [1.0, 0.8, 0.64, 0.512].into_iter().enumerate() {
        assert!(!stalled.should_escalate(progress(iteration + 1, residual)));
    }
    assert!(stalled.should_escalate(progress(5, 0.4096)));

    let mut contracting = rule.handler();
    for (iteration, residual) in [1.0, 0.5, 0.25, 0.125, 0.0625].into_iter().enumerate() {
        assert!(!contracting.should_escalate(progress(iteration + 1, residual)));
    }
}

#[test]
fn test_staleness_rejects_invalid_configuration() {
    assert!(matches!(
        Staleness::try_new(0, 0.7),
        Err(StalenessError::ZeroWindow)
    ));
    for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.0, 1.5] {
        assert!(matches!(
            Staleness::try_new(4, threshold),
            Err(StalenessError::InvalidThreshold { .. })
        ));
    }
}

#[test]
fn original_scale_restart_uses_original_normal_equation_reference() {
    let op = DiagOp(vec![1.0, 2.0, 4.0]);
    let m = IdentityOp { n: 3 };
    let b = [2.0, 3.0, 5.0];
    let x0 = [1.99, 1.49, 1.24];
    let opts = MlsmrOptions {
        warm_start: Some(&x0),
        ..Default::default()
    };
    let old = mlsmr(&op, &b, &m, 0.0, 1, opts).unwrap();
    let restored = MlsmrContinuation::new(&op, &b, &m, 0.0, opts.warm_start, opts.local_size)
        .unwrap()
        .advance(&op, &m, 1, None)
        .unwrap();
    // Only the normalization reference changes; the fresh correction recurrence
    // and the recovered original-coordinate iterate are bit-identical.
    assert_eq!(old.x, restored.x);
    let residual0 = [b[0] - x0[0], b[1] - 2.0 * x0[1], b[2] - 4.0 * x0[2]];
    let original_ref = vec_norm(&[b[0], 2.0 * b[1], 4.0 * b[2]]);
    let correction_ref = vec_norm(&[residual0[0], 2.0 * residual0[1], 4.0 * residual0[2]]);
    assert!(
        (restored.normal_eq_residual - old.normal_eq_residual * correction_ref / original_ref)
            .abs()
            < 1e-16
    );
    let expected = vec_norm(&[
        b[0] - restored.x[0],
        2.0 * (b[1] - 2.0 * restored.x[1]),
        4.0 * (b[2] - 4.0 * restored.x[2]),
    ]) / original_ref;
    assert!((expected - restored.normal_eq_residual).abs() < 1e-15);
}

#[test]
fn original_scale_cold_path_is_exactly_legacy_path() {
    let op = DiagOp((0..30).map(|i| 1.0 + i as f64 / 20.0).collect());
    let b: Vec<f64> = (0..30).map(|i| ((i * 7) % 11) as f64).collect();
    let m = IdentityOp { n: 30 };
    for budget in [1, 20, 200] {
        let opts = MlsmrOptions {
            local_size: Some(12),
            ..Default::default()
        };
        let old = mlsmr(&op, &b, &m, 1e-11, budget, opts).unwrap();
        let new = MlsmrContinuation::new(&op, &b, &m, 1e-11, None, opts.local_size)
            .unwrap()
            .advance(&op, &m, budget, None)
            .unwrap();
        assert_eq!(old.x, new.x);
        assert_eq!(old.iterations, new.iterations);
        assert_eq!(old.residual_norm.to_bits(), new.residual_norm.to_bits());
        assert_eq!(
            old.normal_eq_residual.to_bits(),
            new.normal_eq_residual.to_bits()
        );
        assert_eq!(old.stop_reason, new.stop_reason);
    }
}

#[test]
fn paused_continuation_matches_uninterrupted_recurrence_exactly() {
    let op = DiagOp((0..40).map(|i| 0.5 + i as f64 / 7.0).collect());
    let b: Vec<f64> = (0..40).map(|i| ((i * 7) % 13) as f64 + 0.3).collect();
    let m = IdentityOp { n: 40 };
    let opts = MlsmrOptions {
        local_size: Some(9),
        ..Default::default()
    };
    let expected = mlsmr(&op, &b, &m, 0.0, 12, opts).unwrap();
    let mut continuation = MlsmrContinuation::new(&op, &b, &m, 0.0, None, opts.local_size).unwrap();
    let checkpoint = continuation
        .advance(&op, &m, 12, Some(&FixedIterations(5)))
        .unwrap();
    assert_eq!(checkpoint.stop_reason, LsmrStopReason::Escalated);
    assert_eq!(checkpoint.iterations, 5);
    let resumed = continuation.advance(&op, &m, 7, None).unwrap();
    assert_eq!(resumed.iterations, 7);
    assert_eq!(expected.x, resumed.x);
    assert_eq!(
        expected.normal_eq_residual.to_bits(),
        resumed.normal_eq_residual.to_bits()
    );
    assert_eq!(
        expected.residual_norm.to_bits(),
        resumed.residual_norm.to_bits()
    );
}

#[test]
fn residual_restart_preserves_nonunique_coefficient_gauge() {
    let op = ZeroSecondRow;
    let m = IdentityOp { n: 2 };
    let x0 = [0.5, 7.0];
    let mut continuation =
        MlsmrContinuation::new(&op, &[2.0, 0.0], &m, 1e-12, Some(&x0), None).unwrap();
    let result = continuation.advance(&op, &m, 5, None).unwrap();
    assert!(result.converged);
    assert_eq!(result.x[1].to_bits(), 7.0f64.to_bits());
}

#[test]
fn continuation_applies_one_inverse_per_step_without_reinitializing_on_resume() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Counted {
        n: usize,
        calls: AtomicUsize,
    }
    impl Operator for Counted {
        fn nrows(&self) -> usize {
            self.n
        }
        fn ncols(&self) -> usize {
            self.n
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            y.copy_from_slice(x);
            Ok(())
        }
        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            self.apply(x, y)
        }
    }
    let op = DiagOp((0..40).map(|i| 0.5 + i as f64 / 7.0).collect());
    let b: Vec<f64> = (0..40).map(|i| ((i * 7) % 13) as f64 + 0.3).collect();
    let m = Counted {
        n: 40,
        calls: AtomicUsize::new(0),
    };
    let mut continuation = MlsmrContinuation::new(&op, &b, &m, 0.0, None, Some(9)).unwrap();
    assert_eq!(m.calls.load(Ordering::Relaxed), 1);
    assert!(!continuation.advance(&op, &m, 5, None).unwrap().converged);
    assert_eq!(m.calls.load(Ordering::Relaxed), 6);
    assert!(!continuation.advance(&op, &m, 7, None).unwrap().converged);
    assert_eq!(m.calls.load(Ordering::Relaxed), 13);
}
