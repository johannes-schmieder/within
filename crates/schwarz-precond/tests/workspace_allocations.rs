//! Isolated allocator proof, including the very first serial solve above Rayon's legacy threshold.
use schwarz_precond::{
    mlsmr_with_workspace, MlsmrWorkspace, MlsmrWorkspaceOptions, OperatorMut, SolveError,
};
use stats_alloc::{StatsAlloc, INSTRUMENTED_SYSTEM};
use std::{alloc::System, hint::black_box};
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
struct Diagonal(usize);
impl OperatorMut for Diagonal {
    fn nrows(&self) -> usize {
        self.0
    }
    fn ncols(&self) -> usize {
        self.0
    }
    fn apply(&mut self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        for (i, (out, v)) in y.iter_mut().zip(x).enumerate() {
            *out = *v * (1.0 + (i % 4) as f64 * 0.25);
        }
        Ok(())
    }
    fn apply_adjoint(&mut self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Live positive control: the allocator must see a deliberate allocation.
    let before = GLOBAL.stats();
    let control = black_box(vec![0.0; 123]);
    assert_eq!((GLOBAL.stats() - before).allocations, 1);
    drop(control);
    let n = 12000;
    let mut op = Diagonal(n);
    let mut pre = Diagonal(n);
    let b: Vec<_> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
    let setup_before = GLOBAL.stats();
    let mut w = MlsmrWorkspace::try_new(n, n, Some(8))?;
    let setup = GLOBAL.stats() - setup_before;
    let expected = MlsmrWorkspace::required_payload_bytes(n, n, Some(8))?;
    assert_eq!(w.retained_payload_bytes()?, expected);
    assert_eq!(setup.bytes_allocated, expected);
    assert_eq!(
        (setup.allocations, setup.reallocations, setup.deallocations),
        (11, 0, 0)
    );
    let before = GLOBAL.stats();
    for _ in 0..16 {
        let result = mlsmr_with_workspace(
            &mut op,
            black_box(&b),
            &mut pre,
            1e-10,
            100,
            MlsmrWorkspaceOptions::default(),
            &mut w,
        )?;
        assert!(result.diagnostics.converged);
        black_box(result.x);
    }
    let delta = GLOBAL.stats() - before;
    assert_eq!(
        (delta.allocations, delta.reallocations, delta.deallocations),
        (0, 0, 0)
    );
    let before = GLOBAL.stats();
    drop(w);
    let delta = GLOBAL.stats() - before;
    assert_eq!(delta.bytes_deallocated, expected);
    assert_eq!((delta.allocations, delta.reallocations), (0, 0));
    println!(
        "serial LSMR: first/repeat16 allocations=0; n={n}; retained/released={expected} bytes"
    );
    Ok(())
}
