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

#[test]
fn failed_partial_iteration_consumes_budget_and_keeps_last_accepted_iterate() {
    use crate::{
        BuildError, FixedReducedInverse, FixedReducedInverseFactory, LocalSolveError, PairComponent,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Failing {
        inner: crate::fixed::tests::ProjectedDiagonal,
        calls: AtomicUsize,
        fail_at: usize,
    }
    impl FixedReducedInverse for Failing {
        fn dimension(&self) -> usize {
            self.inner.dimension()
        }
        fn scratch_len(&self) -> usize {
            0
        }
        fn apply_in_place(
            &self,
            rhs: &mut [f64],
            scratch: &mut [f64],
        ) -> Result<(), LocalSolveError> {
            if self.calls.fetch_add(1, Ordering::Relaxed) + 1 == self.fail_at {
                Err(LocalSolveError::BackendFailed {
                    context: "iteration budget witness",
                    message: "injected failure".into(),
                })
            } else {
                self.inner.apply_in_place(rhs, scratch)
            }
        }
    }
    struct Factory(usize);
    impl FixedReducedInverseFactory for Factory {
        fn build(
            &self,
            component: PairComponent<'_>,
        ) -> Result<Arc<dyn FixedReducedInverse>, BuildError> {
            Ok(Arc::new(Failing {
                inner: crate::fixed::tests::ProjectedDiagonal(component.n_retained),
                calls: AtomicUsize::new(0),
                fail_at: self.0,
            }))
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
    let y: Vec<_> = (0..60).map(|i| ((i * 13) % 17) as f64 / 19.0).collect();
    for fail_at in [1, 2, 3] {
        let replacement = solver
            .build_pair_replacement(&Factory(fail_at))
            .unwrap()
            .unwrap();
        assert!(replacement.apply_work_bound().is_none());
        let mut session = solver
            .start_session(
                &y,
                &LsmrOptions {
                    tol: 0.0,
                    ..options(5)
                },
            )
            .unwrap();
        if fail_at == 3 {
            session.run_action(Some(&replacement), 1, None).unwrap();
        }
        let accepted = session.snapshot();
        assert!(session.run_action(Some(&replacement), 5, None).is_err());
        let consumed = fail_at - 1;
        assert_eq!(session.remaining_iterations(), 5 - consumed);
        assert!(session.run_action(None, 5, None).is_err());
        let failed = session.finish();
        assert_eq!(failed.iterations, consumed);
        assert_eq!(failed.x, accepted.x);
        assert_eq!(failed.demeaned, accepted.demeaned);
        assert!(!failed.converged);
    }
}

#[test]
fn categorical_seed_preserves_original_rhs_scale_gauge_and_budget() {
    let solver = Solver::new(
        categorical(),
        Some((0..60).map(|i| 0.2 + (i % 7) as f64).collect()),
        config(),
    )
    .unwrap()
    .with_reproducible()
    .unwrap();
    let y: Vec<f64> = (0..60)
        .map(|i| 1000.0 + (i % 6) as f64 + ((i * 13) % 17) as f64 / 19.0)
        .collect();
    let seed: Vec<f64> = (0..solver.n_dofs())
        .map(|i| 50.0 + i as f64 / 3.0)
        .collect();
    let mut session = solver
        .start_session_categorical_seeded(&y, &seed, &options(80))
        .unwrap();
    let cold = solver.start_session(&y, &options(80)).unwrap();
    assert_eq!(session.b, cold.b);
    assert_eq!(session.y, cold.y);
    assert_eq!(session.remaining_iterations(), 80);
    let initial = session.snapshot();
    assert_eq!(initial.x, seed);
    for i in 0..60 {
        let fit = seed[i % 6] + seed[6 + (i / 6) % 5] + seed[11 + (i / 6) % 5];
        assert!((initial.demeaned[i] - (y[i] - fit)).abs() < 2e-13);
    }
    // Duplicate factors admit a gauge shift. It must remain in the supplied
    // coordinates while the eventual fitted projection is independently correct.
    let gauge: Vec<f64> = seed
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if (6..11).contains(&i) {
                v + 17.0
            } else if i >= 11 {
                v - 17.0
            } else {
                v
            }
        })
        .collect();
    let gauge_initial = solver
        .start_session_categorical_seeded(&y, &gauge, &options(80))
        .unwrap()
        .snapshot();
    assert_eq!(gauge_initial.x, gauge);
    for (a, b) in initial.demeaned.iter().zip(gauge_initial.demeaned) {
        assert!((a - b).abs() < 2e-13);
    }
    session.run_action(None, 1, None).unwrap();
    let replacement = solver
        .build_pair_replacement(&ProjectedFactory)
        .unwrap()
        .unwrap();
    session.run_action(Some(&replacement), 79, None).unwrap();
    let actual = session.finish();
    let expected = solver.solve(&y, &options(80)).unwrap();
    assert!(actual.converged && actual.iterations <= 80);
    for (a, b) in actual.demeaned.iter().zip(expected.demeaned) {
        assert!((a - b).abs() < 1e-8);
    }
    let zeros = vec![0.0; solver.n_dofs()];
    let mut zero = solver
        .start_session_categorical_seeded(&y, &zeros, &options(80))
        .unwrap();
    let mut cold = solver.start_session(&y, &options(80)).unwrap();
    zero.run_action(None, 80, None).unwrap();
    cold.run_action(None, 80, None).unwrap();
    let (zero, cold) = (zero.finish(), cold.finish());
    assert_eq!(zero.x, cold.x);
    assert_eq!(zero.demeaned, cold.demeaned);
    assert_eq!(zero.iterations, cold.iterations);
    assert_eq!(zero.residual.to_bits(), cold.residual.to_bits());
    assert!(solver
        .start_session_categorical_seeded(&y, &[0.0], &options(80))
        .is_err());
    let mut bad = zeros;
    bad[0] = f64::NAN;
    assert!(solver
        .start_session_categorical_seeded(&y, &bad, &options(80))
        .is_err());
    let levels = [0, 0, 1, 1];
    let z = [1.0, 2.0, 3.0, 4.0];
    let slope = Solver::new(
        vec![Effect::new(&levels, true, [&z[..]]).unwrap()],
        None,
        PreconditionerConfig::Off,
    )
    .unwrap();
    assert!(slope
        .start_session_categorical_seeded(&z, &vec![0.0; slope.n_dofs()], &options(80))
        .is_err());
}

#[test]
fn unit_weight_sessions_hold_one_response_copy() {
    let y: Vec<f64> = (0..60).map(|i| ((i * 13) % 17) as f64 / 19.0).collect();
    let weights: Vec<f64> = (0..60).map(|i| 0.2 + (i % 7) as f64).collect();
    for weighted in [false, true] {
        let solver = Solver::new(
            categorical(),
            weighted.then(|| weights.clone()),
            PreconditionerConfig::Diagonal,
        )
        .unwrap();
        let mut session = solver.start_session(&y, &options(80)).unwrap();
        assert_eq!(session.b.is_some(), weighted);
        session.run_action(None, 2, None).unwrap();
        // An unseeded recurrence shares the solve RHS: `y`, or `W^{1/2} y` when weighted.
        let rhs = session.b.as_ref().unwrap_or(&session.y);
        assert_eq!(std::sync::Arc::strong_count(rhs), 2);
        let seed = vec![0.0; solver.n_dofs()];
        let mut seeded = solver
            .start_session_categorical_seeded(&y, &seed, &options(80))
            .unwrap();
        seeded.run_action(None, 2, None).unwrap();
        // A seeded recurrence owns its correction residual instead.
        assert_eq!(std::sync::Arc::strong_count(&seeded.y), 1);
    }
}
