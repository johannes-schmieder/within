//! Isolated first/repeated serial outer allocation boundary, including denied construction.
use schwarz_precond::{
    LocalSolveError, LocalSolver, ReductionStrategy, SchwarzPreconditioner, SolveError,
    SubdomainCore, SubdomainEntry,
};
use stats_alloc::{StatsAlloc, INSTRUMENTED_SYSTEM};
use std::{
    alloc::System,
    hint::black_box,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
struct Identity {
    n: usize,
    fail: Arc<AtomicBool>,
}
impl LocalSolver for Identity {
    fn n_local(&self) -> usize {
        self.n
    }
    fn scratch_size(&self) -> usize {
        self.n + 1
    }
    fn solve_local(
        &self,
        r: &mut [f64],
        z: &mut [f64],
        parallel: bool,
    ) -> Result<(), LocalSolveError> {
        assert!(!parallel);
        if self.fail.load(Ordering::Relaxed) {
            r.fill(f64::NAN);
            return Err(LocalSolveError::BackendFailed {
                context: "injected",
                message: String::new(),
            });
        }
        z[..self.n].copy_from_slice(&r[..self.n]);
        Ok(())
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let before = GLOBAL.stats();
    let control = black_box(vec![0.; 123]);
    assert_eq!((GLOBAL.stats() - before).allocations, 1);
    drop(control);
    let n = 12000;
    let fail = Arc::new(AtomicBool::new(false));
    let entries = (0..2)
        .map(|_| {
            SubdomainEntry::try_new(
                SubdomainCore::uniform((0..n as u32).collect()),
                Identity {
                    n,
                    fail: fail.clone(),
                },
            )
            .unwrap()
        })
        .collect();
    let owner = SchwarzPreconditioner::with_n_dofs(entries, n, ReductionStrategy::Auto);
    let rhs = vec![0.5; n];
    let mut out = vec![-1.; n];
    let expected = 8 * (n + 2 * (n + 1));
    let before = GLOBAL.stats();
    assert!(matches!(
        owner.try_serial_workspace(expected - 1),
        Err(SolveError::SerialWorkspaceBudget { .. })
    ));
    let d = GLOBAL.stats() - before;
    assert_eq!((d.allocations, d.reallocations, d.deallocations), (0, 0, 0));
    let before = GLOBAL.stats();
    let mut w = owner.try_serial_workspace(expected)?;
    let d = GLOBAL.stats() - before;
    assert_eq!((d.allocations, d.reallocations, d.deallocations), (1, 0, 0));
    assert_eq!(d.bytes_allocated, expected);
    assert_eq!(w.retained_payload_bytes(), expected);
    let before = GLOBAL.stats();
    for i in 0..65 {
        w.apply(black_box(&rhs), black_box(&mut out))?;
        assert!(out.iter().all(|&x| x == 1.));
        if i == 0 {
            assert_eq!((GLOBAL.stats() - before).allocations, 0);
        }
    }
    out.fill(9.);
    assert!(w.apply(&rhs[..n - 1], &mut out).is_err());
    assert!(out.iter().all(|&x| x == 9.));
    fail.store(true, Ordering::Relaxed);
    assert!(w.apply(&rhs, &mut out).is_err());
    assert!(out.iter().all(|&x| x == 9.));
    fail.store(false, Ordering::Relaxed);
    w.apply(&rhs, &mut out)?;
    assert!(out.iter().all(|&x| x == 1.));
    let d = GLOBAL.stats() - before;
    assert_eq!((d.allocations, d.reallocations, d.deallocations), (0, 0, 0));
    let before = GLOBAL.stats();
    drop(w);
    let d = GLOBAL.stats() - before;
    assert_eq!((d.allocations, d.reallocations, d.deallocations), (0, 0, 1));
    assert_eq!(d.bytes_deallocated, expected);
    println!("serial Schwarz: 1 allocation, {expected} bytes; first+64 repeats, typed denials, local failure/reuse: zero outer allocations; exact drop");
    Ok(())
}
