//! Isolated allocator proof, including the very first serial solve above Rayon's legacy threshold.
use schwarz_precond::{
    mlsmr_with_workspace, mlsmr_with_workspace_and_candidate_gate, MlsmrWorkspace,
    MlsmrWorkspaceOptions, OperatorMut, SolveError,
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
    let reference_norm = b
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let a = 1.0 + (i % 4) as f64 * 0.25;
            (a * value).powi(2)
        })
        .sum::<f64>()
        .sqrt();
    let gradient_ratio = |x: &[f64]| {
        x.iter()
            .zip(&b)
            .enumerate()
            .map(|(i, (value, target))| {
                let a = 1.0 + (i % 4) as f64 * 0.25;
                (a * (target - a * value)).powi(2)
            })
            .sum::<f64>()
            .sqrt()
            / reference_norm
    };
    let before = GLOBAL.stats();
    for _ in 0..16 {
        let mut calls = 0;
        let mut gate = |x: &[f64], offset: Option<&[f64]>| {
            assert!(offset.is_none());
            calls += 1;
            Ok(gradient_ratio(x) <= 1e-10)
        };
        let result = mlsmr_with_workspace_and_candidate_gate(
            &mut op,
            black_box(&b),
            &mut pre,
            1.0,
            100,
            MlsmrWorkspaceOptions::default(),
            &mut gate,
            &mut w,
        )?;
        assert!(calls > 1 && result.diagnostics.iterations > 1);
        assert!(gradient_ratio(result.x) <= 1e-10);
        black_box(result.x);
    }
    let mut gate_error = |_: &[f64], _: Option<&[f64]>| -> Result<bool, SolveError> {
        Err(SolveError::Synchronization {
            context: "injected candidate gate failure",
        })
    };
    assert!(mlsmr_with_workspace_and_candidate_gate(
        &mut op,
        &b,
        &mut pre,
        1.0,
        100,
        MlsmrWorkspaceOptions::default(),
        &mut gate_error,
        &mut w
    )
    .is_err());
    let delta = GLOBAL.stats() - before;
    assert_eq!(
        (delta.allocations, delta.reallocations, delta.deallocations),
        (0, 0, 0)
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
    // Finite input whose norm overflows, and warm residual overflow, exercise
    // rejection without entering bidiagonalization. Recovery uses the same arrays.
    let overflow = vec![f64::MAX; n];
    let warm = vec![-f64::MAX; n];
    let zero = vec![0.0; n];
    let before = GLOBAL.stats();
    for gated in [false, true] {
        for (rhs, warm_start) in [(&overflow, None), (&zero, Some(warm.as_slice()))] {
            let options = MlsmrWorkspaceOptions {
                warm_start,
                escalation: None,
            };
            let mut gate = |_: &[f64], _: Option<&[f64]>| -> Result<bool, SolveError> {
                panic!("invalid norm must not reach candidate gate")
            };
            let result = if gated {
                mlsmr_with_workspace_and_candidate_gate(
                    &mut op, rhs, &mut pre, 1e-10, 100, options, &mut gate, &mut w,
                )
            } else {
                mlsmr_with_workspace(&mut op, rhs, &mut pre, 1e-10, 100, options, &mut w)
            };
            assert!(
                matches!(result, Err(SolveError::NonFiniteResidualNorm { value_bits }) if !f64::from_bits(value_bits).is_finite())
            );
            let recovered = mlsmr_with_workspace(
                &mut op,
                &b,
                &mut pre,
                1e-10,
                100,
                MlsmrWorkspaceOptions::default(),
                &mut w,
            )?;
            assert!(recovered.diagnostics.converged);
            assert!(gradient_ratio(recovered.x) <= 1e-10);
        }
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
        "serial LSMR: first gated/repeat16/error and native/recovery/norm-overflow allocations=0; n={n}; retained/released={expected} bytes"
    );
    Ok(())
}
