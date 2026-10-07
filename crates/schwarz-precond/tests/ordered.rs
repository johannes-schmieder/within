use rayon::prelude::*;
use schwarz_precond::{
    LocalSolveError, LocalSolver, ReductionStrategy, SchwarzPreconditioner, SolveError,
    SubdomainCore, SubdomainEntry,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

struct DelayedDiagonal {
    scale: f64,
    delay: u64,
    fail: Arc<AtomicBool>,
}
impl LocalSolver for DelayedDiagonal {
    fn n_local(&self) -> usize {
        1
    }
    fn scratch_size(&self) -> usize {
        1
    }
    fn solve_local(
        &self,
        rhs: &mut [f64],
        out: &mut [f64],
        _: bool,
    ) -> Result<(), LocalSolveError> {
        std::thread::sleep(std::time::Duration::from_micros(self.delay));
        out[0] = rhs[0] * self.scale;
        if self.fail.load(Ordering::Relaxed) {
            return Err(LocalSolveError::BackendFailed {
                context: "ordered fixture",
                message: "injected failure".into(),
            });
        }
        Ok(())
    }
}

#[test]
fn ordered_windows_repeat_under_overlapping_delayed_and_concurrent_applies() {
    let entries = (0..7)
        .map(|i| {
            SubdomainEntry::try_new(
                SubdomainCore::uniform(vec![0]),
                DelayedDiagonal {
                    scale: if i == 0 { 1.0 } else { 2.0_f64.powi(-53) },
                    delay: (7 - i) * 17,
                    fail: Arc::new(AtomicBool::new(false)),
                },
            )
            .unwrap()
        })
        .collect();
    let op = SchwarzPreconditioner::with_n_dofs(entries, 2, ReductionStrategy::Ordered);
    for workers in [8, 1, 4, 2, 8] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(|| {
                assert_eq!(op.reduction_strategy(), ReductionStrategy::Ordered);
                for _ in 0..8 {
                    (0..9).into_par_iter().for_each(|_| {
                        let mut out = [f64::NAN; 2];
                        op.apply(&[1.0, 17.0], &mut out).unwrap();
                        assert_eq!(out, [1.0, 0.0]);
                    });
                }
            });
    }
}

#[test]
fn later_window_errors_clear_output_and_choose_first_domain_then_recover() {
    let failed = Arc::new(AtomicBool::new(true));
    let entries = (0..7)
        .map(|i| {
            SubdomainEntry::try_new(
                SubdomainCore::uniform(vec![0]),
                DelayedDiagonal {
                    scale: 1.0,
                    delay: (7 - i) * 10,
                    fail: if i >= 4 {
                        Arc::clone(&failed)
                    } else {
                        Arc::new(AtomicBool::new(false))
                    },
                },
            )
            .unwrap()
        })
        .collect();
    let op = SchwarzPreconditioner::new(entries, ReductionStrategy::Ordered);
    for workers in [1, 2, 4, 8] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(|| {
                let mut out = [97.0];
                assert!(matches!(
                    op.apply(&[1.0], &mut out),
                    Err(SolveError::LocalSolveFailed { subdomain: 4, .. })
                ));
                assert_eq!(out, [0.0]);
            });
    }
    failed.store(false, Ordering::Relaxed);
    let mut out = [97.0];
    op.apply(&[1.0], &mut out).unwrap();
    assert_eq!(out, [7.0]);
    let empty: SchwarzPreconditioner<DelayedDiagonal> =
        SchwarzPreconditioner::with_n_dofs(vec![], 2, ReductionStrategy::Ordered);
    let mut out = [97.0; 2];
    empty.apply(&[1.0; 2], &mut out).unwrap();
    assert_eq!(out, [0.0; 2]);
}
