//! A failed application or exit audit must not refund iterations to an outer policy.
use schwarz_precond::{lsmr_counted, mlsmr_counted, MlsmrOptions, Operator, SolveError};
use std::sync::atomic::{AtomicUsize, Ordering};

struct FailingDiagonal {
    diagonal: Vec<f64>,
    applies: AtomicUsize,
    fail_at: usize,
}
impl FailingDiagonal {
    fn new(diagonal: Vec<f64>, fail_at: usize) -> Self {
        Self {
            diagonal,
            applies: AtomicUsize::new(0),
            fail_at,
        }
    }
    fn product(&self, x: &[f64], y: &mut [f64]) {
        for ((out, x), a) in y.iter_mut().zip(x).zip(&self.diagonal) {
            *out = a * x;
        }
    }
}
impl Operator for FailingDiagonal {
    fn nrows(&self) -> usize {
        self.diagonal.len()
    }
    fn ncols(&self) -> usize {
        self.diagonal.len()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        if self.applies.fetch_add(1, Ordering::SeqCst) + 1 == self.fail_at {
            return Err(SolveError::Synchronization {
                context: "injected application failure",
            });
        }
        self.product(x, y);
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.product(x, y);
        Ok(())
    }
}

#[test]
fn failing_iteration_is_charged_for_standard_and_modified_lsmr() {
    for modified in [false, true] {
        let op = FailingDiagonal::new((1..=8).map(f64::from).collect(), 3);
        let identity = FailingDiagonal::new(vec![1.0; 8], usize::MAX);
        let result = if modified {
            mlsmr_counted(&op, &[1.0; 8], &identity, 0.0, 10, MlsmrOptions::default())
        } else {
            lsmr_counted(&op, &[1.0; 8], 0.0, 10, None)
        };
        let error = result.err().expect("application must fail");
        assert_eq!(error.iterations, 3);
        assert_eq!(op.applies.load(Ordering::SeqCst), 3);
        assert!(matches!(error.source, SolveError::Synchronization { .. }));
    }
}

#[test]
fn failed_true_residual_audit_preserves_spent_iteration() {
    for modified in [false, true] {
        let op = FailingDiagonal::new(vec![1.0; 3], 2);
        let identity = FailingDiagonal::new(vec![1.0; 3], usize::MAX);
        let result = if modified {
            mlsmr_counted(
                &op,
                &[1.0; 3],
                &identity,
                1e-10,
                10,
                MlsmrOptions::default(),
            )
        } else {
            lsmr_counted(&op, &[1.0; 3], 1e-10, 10, None)
        };
        let error = result.err().expect("exit audit must fail");
        assert_eq!(error.iterations, 1);
        assert_eq!(op.applies.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn invalid_input_and_initial_preconditioner_failure_spend_zero() {
    let op = FailingDiagonal::new(vec![1.0; 3], usize::MAX);
    let bad = lsmr_counted(&op, &[f64::NAN; 3], 1e-10, 10, None)
        .err()
        .unwrap();
    assert_eq!(bad.iterations, 0);
    let preconditioner = FailingDiagonal::new(vec![1.0; 3], 1);
    let failed = mlsmr_counted(
        &op,
        &[1.0; 3],
        &preconditioner,
        1e-10,
        10,
        MlsmrOptions::default(),
    )
    .err()
    .unwrap();
    assert_eq!(failed.iterations, 0);
}
