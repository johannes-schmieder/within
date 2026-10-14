use super::*;
use crate::fixed::tests::ProjectedFactory;
use crate::{Design, Effect, LocalSolverConfig, PreconditionerConfig, ReductionStrategy};

fn config() -> PreconditionerConfig {
    PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig {
            dense_threshold: 2,
            ..Default::default()
        },
        reduction: ReductionStrategy::Ordered,
    }
}
fn options(maxiter: usize) -> LsmrOptions {
    LsmrOptions {
        tol: 1e-12,
        maxiter,
        local_size: Some(12),
    }
}
fn categorical() -> Design<'static> {
    let a: Vec<u32> = (0..60).map(|i| i % 6).collect();
    let b: Vec<u32> = (0..60).map(|i| (i / 6) % 5).collect();
    // Exact redundancy is legal: no joint full-rank coordinate check.
    Design::from_levels_for_test(vec![a, b.clone(), b])
}

#[test]
fn restart_preserves_original_budget_and_recovers_redundant_weighted_fit() {
    let weights: Vec<f64> = (0..60).map(|i| 0.2 + (i % 7) as f64).collect();
    let solver = Solver::new(categorical(), Some(weights.clone()), config())
        .unwrap()
        .with_reproducible()
        .unwrap();
    let y: Vec<f64> = (0..60)
        .map(|i| (i % 6) as f64 - 0.2 * ((i / 6) % 5) as f64 + ((i * 13) % 17) as f64 / 19.0)
        .collect();
    let replacement = solver
        .build_pair_replacement(&ProjectedFactory)
        .unwrap()
        .unwrap();
    let mut session = solver.start_session(&y, &options(80)).unwrap();
    let first = session.run_action(None, 1, None).unwrap();
    assert_eq!(first.iterations, 1);
    assert_eq!(session.remaining_iterations(), 79);
    let second = session
        .run_action(Some(&replacement), usize::MAX, None)
        .unwrap();
    assert_eq!(
        second.total_iterations,
        first.iterations + second.iterations
    );
    assert!(second.total_iterations <= 80);
    assert!(second.converged);
    let result = session.finish();
    let baseline = solver.solve(&y, &options(80)).unwrap();
    for (a, b) in result.demeaned.iter().zip(&baseline.demeaned) {
        assert!((a - b).abs() < 1e-9);
    }
    // Independent dummy normal equations in original observation order.
    for column in 0..3 {
        for level in 0..if column == 0 { 6 } else { 5 } {
            let moment: f64 = (0..60)
                .filter(|i| {
                    if column == 0 {
                        i % 6 == level
                    } else {
                        (i / 6) % 5 == level
                    }
                })
                .map(|i| weights[i] * result.demeaned[i])
                .sum();
            assert!(moment.abs() < 2e-9, "moment {moment}");
        }
    }
    let mut bounded = solver.start_session(&y, &options(2)).unwrap();
    bounded.run_action(None, 1, None).unwrap();
    let last = bounded.run_action(Some(&replacement), 100, None).unwrap();
    assert!(last.total_iterations <= 2);
    assert_eq!(bounded.remaining_iterations(), 0);
    assert_eq!(bounded.run_action(None, 100, None).unwrap().iterations, 0);
}

#[test]
fn replacement_requires_exact_originating_solver() {
    let a = Solver::new(categorical(), None, config()).unwrap();
    let b = Solver::new(categorical(), Some(vec![2.0; 60]), config()).unwrap();
    let replacement = a
        .build_pair_replacement(&ProjectedFactory)
        .unwrap()
        .unwrap();
    let mut session = b.start_session(&vec![1.0; 60], &options(10)).unwrap();
    assert!(session.run_action(Some(&replacement), 10, None).is_err());
    assert_eq!(session.remaining_iterations(), 10);
}

#[test]
fn restart_keeps_internal_slope_basis_and_recovers_once() {
    let a: Vec<u32> = (0..120).map(|i| (i % 6) as u32).collect();
    let b: Vec<u32> = (0..120).map(|i| ((i / 6) % 5) as u32).collect();
    let z: Vec<f64> = (0..120).map(|i| ((i * 7) % 19) as f64 / 11.0).collect();
    let y: Vec<f64> = (0..120)
        .map(|i| {
            1.5 + 0.2 * a[i] as f64 + z[i] * (1.0 + b[i] as f64) + ((i * 11) % 7) as f64 / 20.0
        })
        .collect();
    let design = Design::new(vec![
        Effect::new(&a, true, [&z[..]]).unwrap(),
        Effect::new(&b, true, []).unwrap(),
    ])
    .unwrap();
    let solver = Solver::new(design, None, config())
        .unwrap()
        .with_reproducible()
        .unwrap();
    let mut session = solver.start_session(&y, &options(200)).unwrap();
    session.run_action(None, 1, None).unwrap();
    session.restart_correction(1e-13).unwrap();
    assert!(session.run_action(None, 199, None).unwrap().converged);
    let result = session.finish();
    let baseline = solver.solve(&y, &options(200)).unwrap();
    for (got, want) in result.demeaned.iter().zip(&baseline.demeaned) {
        assert!((got - want).abs() < 1e-9);
    }
    // Verify the recovered coefficients predict directly in the original basis.
    for i in 0..120 {
        let fitted = result.x[a[i] as usize]
            + z[i] * result.x[6 + a[i] as usize]
            + result.x[12 + b[i] as usize];
        assert!((y[i] - fitted - result.demeaned[i]).abs() < 2e-13);
    }
}

#[test]
fn refused_build_and_snapshot_preserve_incumbent_recurrence_exactly() {
    struct Reject;
    impl crate::FixedReducedInverseFactory for Reject {
        fn build(
            &self,
            _: crate::PairComponent<'_>,
        ) -> Result<std::sync::Arc<dyn crate::FixedReducedInverse>, crate::BuildError> {
            Err(crate::BuildError::LocalSolverBuild("refused setup".into()))
        }
    }
    let solver = Solver::new(
        categorical(),
        Some((0..60).map(|i| 0.2 + (i % 7) as f64).collect()),
        config(),
    )
    .unwrap()
    .with_reproducible()
    .unwrap();
    let y: Vec<f64> = (0..60).map(|i| ((i * 13) % 17) as f64 / 19.0).collect();
    let options = options(80);
    let uninterrupted = solver.solve(&y, &options).unwrap();
    let mut session = solver.start_session(&y, &options).unwrap();
    let first = session.run_action(None, 1, None).unwrap();
    assert_eq!(first.iterations, 1);
    let checkpoint = session.snapshot();
    assert_eq!(checkpoint.iterations, 1);
    assert!(solver.build_pair_replacement(&Reject).is_err());
    session.run_action(None, 79, None).unwrap();
    let resumed = session.finish();
    assert_eq!(uninterrupted.x, resumed.x);
    assert_eq!(uninterrupted.demeaned, resumed.demeaned);
    assert_eq!(uninterrupted.iterations, resumed.iterations);
    assert_eq!(uninterrupted.residual.to_bits(), resumed.residual.to_bits());
}

#[test]
fn correction_keeps_budget_and_owned_replacement_survives_solver_move() {
    let solver = Solver::new(categorical(), None, config())
        .unwrap()
        .with_reproducible()
        .unwrap();
    let replacement = solver
        .build_pair_replacement(&ProjectedFactory)
        .unwrap()
        .unwrap();
    let moved = Box::new(solver);
    let y: Vec<f64> = (0..60).map(|i| ((i * 13) % 17) as f64 / 19.0).collect();
    let mut session = moved.start_session(&y, &options(80)).unwrap();
    session.run_action(Some(&replacement), 1, None).unwrap();
    let left = session.remaining_iterations();
    assert!(session.restart_correction(1e-8).is_err());
    session.restart_correction(1e-13).unwrap();
    assert_eq!(session.remaining_iterations(), left);
    session
        .run_action(Some(&replacement.clone()), left, None)
        .unwrap();
    assert!(session.finish().converged);
}
