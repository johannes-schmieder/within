//! Independent output witnesses; the output policy must not change the solve.
use within::{
    ControlledSolveResult, Design, Effect, LsmrOptions, PreconditionerConfig, ResidualOutput,
    SolveControl, Solver,
};

#[path = "fixtures/rectifier_residual.rs"]
mod rectifier;

fn options(maxiter: usize) -> LsmrOptions {
    LsmrOptions {
        tol: 1e-12,
        maxiter,
        local_size: None,
    }
}

fn same_solve(a: &ControlledSolveResult, b: &ControlledSolveResult) {
    assert_eq!(a.solution.x, b.solution.x);
    assert_eq!(a.solution.iterations, b.solution.iterations);
    assert_eq!(a.solution.converged, b.solution.converged);
    assert_eq!(a.stop_reason, b.stop_reason);
    assert_eq!(a.solution.residual.to_bits(), b.solution.residual.to_bits());
    assert_eq!(a.solution.unidentified, b.solution.unidentified);
    assert_eq!(
        format!("{:?}", a.solution.warnings),
        format!("{:?}", b.solution.warnings)
    );
}

struct StopAfterOne;

impl within::EscalationPolicy for StopAfterOne {
    fn handler(&self) -> Box<dyn within::EscalationHandler> {
        Box::new(Self)
    }
}

impl within::EscalationHandler for StopAfterOne {
    fn should_escalate(&mut self, progress: within::Progress) -> bool {
        progress.iteration >= 1
    }
}

#[test]
fn output_policy_preserves_early_stop_and_shared_budget_handoff_for_each_preconditioner() {
    let n = 900;
    let f: Vec<_> = (0..n).map(|i| (i * 7 % 31) as u32).collect();
    let g: Vec<_> = (0..n).map(|i| ((i / 3 + i % 7) % 23) as u32).collect();
    let h: Vec<_> = (0..n).map(|i| ((i / 5 + i % 11) % 19) as u32).collect();
    let y: Vec<_> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
    let w: Vec<_> = (0..n).map(|i| 0.1 + (i % 9) as f64).collect();
    for config in [
        PreconditionerConfig::Off,
        PreconditionerConfig::Diagonal,
        PreconditionerConfig::Additive {
            local_solver: Default::default(),
            reduction: within::ReductionStrategy::ParallelReduction,
        },
    ] {
        let solver = Solver::new(
            vec![
                Effect::new(&f, true, []).unwrap(),
                Effect::new(&g, true, []).unwrap(),
                Effect::new(&h, true, []).unwrap(),
            ],
            Some(&w),
            config,
        )
        .unwrap();
        let stop = |residual_output| {
            solver
                .solve_controlled(
                    &y,
                    &options(100),
                    SolveControl {
                        escalation: Some(&StopAfterOne),
                        residual_output,
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        let audit = stop(ResidualOutput::WeightedAudit);
        let model = stop(ResidualOutput::PreparedModel);
        same_solve(&audit, &model);
        assert_eq!(model.stop_reason, within::LsmrStopReason::Escalated);
        assert_eq!(model.solution.iterations, 1);
        let audit = output(
            &solver,
            &y,
            Some(&model.solution.x),
            99,
            ResidualOutput::WeightedAudit,
        );
        let model = output(
            &solver,
            &y,
            Some(&model.solution.x),
            99,
            ResidualOutput::PreparedModel,
        );
        same_solve(&audit, &model);
        assert!(model.solution.converged);
        assert!(model.solution.iterations < 100);
        let mut gradient = [0.0f64; 73];
        for i in 0..n {
            let residual = y[i]
                - (model.solution.x[f[i] as usize]
                    + model.solution.x[31 + g[i] as usize]
                    + model.solution.x[54 + h[i] as usize]);
            assert_eq!(model.solution.demeaned[i], residual);
            gradient[f[i] as usize] += w[i] * residual;
            gradient[31 + g[i] as usize] += w[i] * residual;
            gradient[54 + h[i] as usize] += w[i] * residual;
        }
        assert!(gradient.iter().all(|g| g.abs() < 1e-8));
    }
}

fn output(
    solver: &Solver<'_>,
    y: &[f64],
    seed: Option<&[f64]>,
    maxiter: usize,
    residual_output: ResidualOutput,
) -> ControlledSolveResult {
    solver
        .solve_controlled(
            y,
            &options(maxiter),
            SolveControl {
                initial_coefficients: seed,
                residual_output,
                ..Default::default()
            },
        )
        .unwrap()
}

#[test]
fn categorical_projection_matches_weighted_group_means_in_caller_order() {
    let f = [2, 0, 1, 2, 1, 0, 2, 0, 1, 1, 2, 0];
    let y = [3.0, -1.0, 8.0, 2.0, 4.0, 3.0, 7.0, 9.0, -2.0, 6.0, 4.0, 5.0];
    let ordinary = [0.3, 2.0, 4.0, 1.0, 3.0, 0.4, 2.0, 8.0, 3.0, 2.0, 1.0, 7.0];
    let extreme = [
        1e-18, 1.0, 1e-18, 1.0, 1.0, 1e-18, 1e-18, 1.0, 1.0, 1e-18, 1.0, 1.0,
    ];
    let zero = [0.0, 2.0, 0.0, 1.0, 3.0, 0.0, 2.0, 8.0, 3.0, 0.0, 1.0, 7.0];
    for weights in [
        None,
        Some(&ordinary[..]),
        Some(&extreme[..]),
        Some(&zero[..]),
    ] {
        for config in [PreconditionerConfig::Off, PreconditionerConfig::Diagonal] {
            let solver =
                Solver::new(vec![Effect::new(&f, true, []).unwrap()], weights, config).unwrap();
            let audit = output(&solver, &y, None, 50, ResidualOutput::WeightedAudit);
            let model = output(&solver, &y, None, 50, ResidualOutput::PreparedModel);
            same_solve(&audit, &model);
            assert!(model.solution.converged);
            let mut total = [0.0; 3];
            let mut mass = [0.0; 3];
            for i in 0..y.len() {
                let weight = weights.map_or(1.0, |w| w[i]);
                total[f[i] as usize] += weight * y[i];
                mass[f[i] as usize] += weight;
            }
            for i in 0..y.len() {
                let expected = y[i] - total[f[i] as usize] / mass[f[i] as usize];
                assert!((model.solution.demeaned[i] - expected).abs() < 2e-12);
                assert_eq!(
                    model.solution.demeaned[i],
                    y[i] - model.solution.x[f[i] as usize]
                );
            }
            if weights == Some(&zero[..]) {
                // The default already reconstructs when a zero weight erases a row.
                assert_eq!(audit.solution.demeaned, model.solution.demeaned);
            }
        }
    }
}

#[test]
fn sloped_rank_drop_and_warm_correction_keep_the_prepared_model() {
    let n = 180;
    let f: Vec<_> = (0..n).map(|i| (i * 7 % 3) as u32).collect();
    let z: Vec<_> = (0..n).map(|i| 8.0 + (i % 11) as f64 / 4.0).collect();
    let duplicate: Vec<_> = z.iter().map(|z| 2.0 * z).collect();
    let y: Vec<_> = (0..n).map(|i| (i as f64 * 0.3).sin() + z[i]).collect();
    for zero_weights in [false, true] {
        let w: Vec<_> = (0..n)
            .map(|i| {
                if zero_weights && i % 9 == 0 {
                    0.0
                } else {
                    1.0 + (i % 5) as f64
                }
            })
            .collect();
        for sorted in [false, true] {
            let terms = vec![Effect::new(&f, true, [&z[..], &duplicate[..]]).unwrap()];
            let design = if sorted {
                Design::new(terms)
            } else {
                Design::new_unsorted(terms)
            }
            .unwrap();
            let solver = Solver::new(design, Some(&w), PreconditionerConfig::Diagonal).unwrap();
            let audit = output(&solver, &y, None, 50, ResidualOutput::WeightedAudit);
            let model = output(&solver, &y, None, 50, ResidualOutput::PreparedModel);
            same_solve(&audit, &model);
            assert!(model.solution.converged);
            assert_eq!(model.solution.unidentified.len(), 3);
            let corrected = output(
                &solver,
                &y,
                Some(&model.solution.x),
                50,
                ResidualOutput::PreparedModel,
            );
            let audit_corrected = output(
                &solver,
                &y,
                Some(&model.solution.x),
                50,
                ResidualOutput::WeightedAudit,
            );
            same_solve(&audit_corrected, &corrected);
            // Independent centered two-column weighted least squares for each group.
            for level in 0..3 {
                let rows: Vec<_> = (0..n).filter(|&i| f[i] == level).collect();
                let mass: f64 = rows.iter().map(|&i| w[i]).sum();
                let zm = rows.iter().map(|&i| w[i] * z[i]).sum::<f64>() / mass;
                let ym = rows.iter().map(|&i| w[i] * y[i]).sum::<f64>() / mass;
                let slope = rows
                    .iter()
                    .map(|&i| w[i] * (z[i] - zm) * (y[i] - ym))
                    .sum::<f64>()
                    / rows
                        .iter()
                        .map(|&i| w[i] * (z[i] - zm).powi(2))
                        .sum::<f64>();
                for i in rows {
                    let expected = y[i] - (ym + slope * (z[i] - zm));
                    assert!((model.solution.demeaned[i] - expected).abs() < 2e-11);
                    assert!((corrected.solution.demeaned[i] - expected).abs() < 2e-11);
                }
            }
        }
    }
}

#[test]
fn captured_rectifier_output_reconstructs_without_resolving_or_unweighting() {
    // Synthetic PPML separation rectifier input: tiny positive weights made
    // division of the weighted audit residual cross a downstream FE certificate.
    // Freeze the already obtained coefficients to isolate output arithmetic.
    let solver = Solver::new(
        vec![
            Effect::new(&rectifier::F, true, []).unwrap(),
            Effect::new(&rectifier::G, true, []).unwrap(),
        ],
        Some(&rectifier::W),
        PreconditionerConfig::Diagonal,
    )
    .unwrap();
    let audit = output(
        &solver,
        &rectifier::Y,
        Some(&rectifier::X),
        0,
        ResidualOutput::WeightedAudit,
    );
    let model = output(
        &solver,
        &rectifier::Y,
        Some(&rectifier::X),
        0,
        ResidualOutput::PreparedModel,
    );
    same_solve(&audit, &model);
    assert_eq!(model.solution.x, rectifier::X);
    assert_eq!(model.solution.iterations, 0);
    let expected: Vec<_> = (0..rectifier::Y.len())
        .map(|i| {
            rectifier::Y[i]
                - (rectifier::X[rectifier::F[i] as usize]
                    + rectifier::X[29 + rectifier::G[i] as usize])
        })
        .collect();
    assert_eq!(model.solution.demeaned, expected);
    let reference = rectifier::Y
        .iter()
        .zip(rectifier::W)
        .map(|(y, w)| w * y * y)
        .sum::<f64>()
        .sqrt();
    let mut worst = 0.0f64;
    for (levels, count) in [(&rectifier::F, 29), (&rectifier::G, 20)] {
        for level in 0..count {
            let rows: Vec<_> = (0..levels.len()).filter(|&i| levels[i] == level).collect();
            let mass = compensated(rows.iter().map(|&i| rectifier::W[i]));
            // Independently evaluate the shifted group-mean certificate used by
            // the caller, preserving its sensitivity to finite output arithmetic.
            let anchor = expected[rows[0]];
            let mean = anchor
                + compensated(
                    rows.iter()
                        .map(|&i| rectifier::W[i] * (expected[i] - anchor)),
                ) / mass;
            worst = worst.max(mean.abs() * mass.sqrt() / reference);
        }
    }
    assert!(worst < 1e-8, "original downstream certificate: {worst:e}");
}

fn compensated(values: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut correction) = (0.0f64, 0.0f64);
    for value in values {
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + correction
}

#[test]
fn parallel_reconstruction_repeats_and_restores_original_row_order() {
    let n = 16_384;
    let f: Vec<_> = (0..n).map(|i| (i * 7 % 31) as u32).collect();
    let g: Vec<_> = (0..n).map(|i| ((i / 3 + i % 7) % 23) as u32).collect();
    let y: Vec<_> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
    let w: Vec<_> = (0..n)
        .map(|i| {
            if i % 13 == 0 {
                1e-18
            } else {
                1.0 + (i % 9) as f64
            }
        })
        .collect();
    let seed: Vec<_> = (0..54).map(|i| (i as f64 - 27.0) / 8.0).collect();
    let expected: Vec<_> = (0..n)
        .map(|i| y[i] - (seed[f[i] as usize] + seed[31 + g[i] as usize]))
        .collect();
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            let solver = Solver::new(
                vec![
                    Effect::new(&f, true, []).unwrap(),
                    Effect::new(&g, true, []).unwrap(),
                ],
                Some(&w),
                PreconditionerConfig::Diagonal,
            )
            .unwrap();
            for _ in 0..2 {
                let result = output(&solver, &y, Some(&seed), 0, ResidualOutput::PreparedModel);
                assert_eq!(result.solution.x, seed);
                assert_eq!(result.solution.demeaned, expected);
            }
        });
    }
}
