//! Swapping a solver's preconditioner keeps its prepared design, weights and
//! adjoint index and is bitwise a fresh construction with the new config.

use within::{
    BuildError, Effect, LocalSolverConfig, LsmrOptions, PreconditionerConfig, ReductionStrategy,
    Solver, StepKernel,
};

fn additive() -> PreconditionerConfig {
    PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig::default(),
        reduction: ReductionStrategy::Ordered,
    }
}

fn bits(result: &within::SolveResult) -> Vec<u64> {
    result
        .x
        .iter()
        .chain(&result.demeaned)
        .map(|v| v.to_bits())
        .chain([result.iterations as u64, result.residual.to_bits()])
        .collect()
}

fn prepare(solver: Solver<'_>, kernel: StepKernel, reproducible: bool) -> Solver<'_> {
    let solver = solver.with_step_kernel(kernel).unwrap();
    if reproducible {
        solver.with_reproducible().unwrap()
    } else {
        solver
    }
}

#[test]
fn swapped_preconditioner_is_bitwise_a_fresh_solver() {
    let rows = 30_000usize;
    // The firm term dominates and is unsorted, so construction locality-sorts rows.
    let worker: Vec<u32> = (0..rows).map(|i| ((i * 7_919) % 6_007) as u32).collect();
    let firm: Vec<u32> = (0..rows).map(|i| ((i * 104_729) % 9_001) as u32).collect();
    let third: Vec<u32> = (0..rows).map(|i| (i % 32) as u32).collect();
    let slope: Vec<f64> = (0..rows).map(|i| (i % 13) as f64 * 0.1 - 0.6).collect();
    let y: Vec<f64> = (0..rows)
        .map(|i| ((i * 31) % 101) as f64 / 50.0 + f64::from(third[i]) * 0.01)
        .collect();
    let weights: Vec<f64> = (0..rows).map(|i| 0.5 + (i % 9) as f64 * 0.25).collect();
    let options = LsmrOptions {
        tol: 1e-10,
        maxiter: 300,
        local_size: None,
    };
    for slopes in [false, true] {
        let effects = || {
            let none: Vec<&[f64]> = Vec::new();
            vec![
                Effect::new(&worker, true, none.clone()).unwrap(),
                Effect::new(&firm, true, none).unwrap(),
                Effect::new(&third, true, slopes.then_some(slope.as_slice())).unwrap(),
            ]
        };
        for weighted in [false, true] {
            let w = weighted.then(|| weights.clone());
            for reproducible in [false, true] {
                for kernel in [StepKernel::Fused, StepKernel::Separate] {
                    let fresh = prepare(
                        Solver::new(effects(), w.clone(), additive()).unwrap(),
                        kernel,
                        reproducible,
                    );
                    let swapped = prepare(
                        Solver::new(effects(), w.clone(), PreconditionerConfig::Diagonal).unwrap(),
                        kernel,
                        reproducible,
                    )
                    .with_preconditioner(w.as_deref(), additive())
                    .unwrap();
                    let case = (slopes, weighted, reproducible, kernel);
                    assert_eq!(fresh.warnings().len(), swapped.warnings().len(), "{case:?}");
                    let expected = fresh.solve(&y, &options).unwrap();
                    let actual = swapped.solve(&y, &options).unwrap();
                    assert!(expected.converged, "{case:?}");
                    assert_eq!(bits(&expected), bits(&actual), "{case:?}");
                }
            }
        }
    }
}

#[test]
fn swapped_preconditioner_rejects_different_weights() {
    let levels: Vec<u32> = (0..100).map(|i| i % 7).collect();
    let other: Vec<u32> = (0..100).map(|i| i % 5).collect();
    let weights: Vec<f64> = (0..100).map(|i| 1.0 + (i % 3) as f64).collect();
    let solver = |w: Option<Vec<f64>>| {
        let none = std::iter::empty::<&[f64]>;
        Solver::new(
            vec![
                Effect::new(&levels, true, none()).unwrap(),
                Effect::new(&other, true, none()).unwrap(),
            ],
            w,
            PreconditionerConfig::Diagonal,
        )
        .unwrap()
    };
    let mut changed = weights.clone();
    changed[17] = 2.5;
    for (built, offered) in [
        (Some(weights.clone()), Some(changed.as_slice())),
        (Some(weights.clone()), None),
        (None, Some(weights.as_slice())),
    ] {
        assert!(matches!(
            solver(built).with_preconditioner(offered, additive()),
            Err(BuildError::WeightsDiffer)
        ));
    }
}
