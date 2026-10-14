//! Screening is diagnostic: both policies retain exact whitening and solves.
//!
//! Integration witness against revision 0a828c5038552eede16a12b057aeb7b6be261923:
//! 48 same-Mac fits (one/four workers, intercept/slope-only, weighted/unweighted,
//! degenerate/regular, Off/Diagonal/prebuilt) matched the Disabled policy exactly.
//! The comparison included coefficient/residual bits, convergence, iterations,
//! normal-residual diagnostic bits, unidentified ordering and warnings. Both
//! serialized scientific captures had SHA256
//! d17d3b0bd394679fe0b92fa622fabdc7b580e61d82a59a1a3c88b2dbf3d3d8d1.
//! This is a recorded same-host merge witness, not a cross-platform bit assertion.
//! Maintained tests below additionally check original-loading normal equations;
//! tests/slopes.rs supplies independent weighted closed-form projection witnesses.
use within::{BuildWarning, Effect, LsmrOptions, PreconditionerConfig, SlopeScreening, Solver};

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn disabled_screening_retains_slope_math_for_weighted_degenerate_off_and_prebuilt_solves() {
    for threads in [1, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let n = 120;
                let a: Vec<u32> = (0..n).map(|i| (i % 6) as u32).collect();
                let b: Vec<u32> = (0..n).map(|i| ((i / 6) % 5) as u32).collect();
                for intercept in [false, true] {
                    for weighted in [false, true] {
                        for degenerate in [false, true] {
                            let z: Vec<f64> = (0..n)
                                .map(|i| {
                                    if degenerate && a[i] == 0 {
                                        2.0
                                    } else {
                                        ((i * 7) % 19) as f64 / 11.0
                                    }
                                })
                                .collect();
                            let y: Vec<f64> = (0..n)
                                .map(|i| {
                                    1.3 + 0.2 * a[i] as f64
                                        + z[i] * (1.0 + b[i] as f64)
                                        + ((i * 11) % 7) as f64 / 20.0
                                })
                                .collect();
                            let w = weighted.then(|| {
                                (0..n)
                                    .map(|i| {
                                        if degenerate && a[i] == 5 {
                                            0.0
                                        } else {
                                            0.2 + (i % 7) as f64
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            });
                            let effects = || {
                                vec![
                                    Effect::new(&a, intercept, [&z[..]]).unwrap(),
                                    Effect::new(&b, intercept, [&z[..]]).unwrap(),
                                ]
                            };
                            for config in
                                [PreconditionerConfig::Off, PreconditionerConfig::Diagonal]
                            {
                                let default = Solver::new(effects(), w.clone(), &config)
                                    .unwrap()
                                    .with_reproducible()
                                    .unwrap();
                                let disabled = Solver::new_with_screening(
                                    effects(),
                                    w.clone(),
                                    &config,
                                    SlopeScreening::Disabled,
                                )
                                .unwrap()
                                .with_reproducible()
                                .unwrap();
                                assert!(default.warnings().iter().any(|warning| matches!(
                                    warning,
                                    BuildWarning::CollinearSlopeCovariate { .. }
                                )));
                                assert!(disabled.warnings().is_empty());
                                let options = LsmrOptions {
                                    tol: 1e-11,
                                    maxiter: 200,
                                    local_size: Some(16),
                                };
                                let baseline = default.solve(&y, &options).unwrap();
                                let actual = disabled.solve(&y, &options).unwrap();
                                assert!(actual.converged);
                                assert_eq!(bits(&baseline.x), bits(&actual.x));
                                assert_eq!(bits(&baseline.demeaned), bits(&actual.demeaned));
                                assert_eq!(baseline.iterations, actual.iterations);
                                assert_eq!(baseline.residual.to_bits(), actual.residual.to_bits());
                                assert_eq!(baseline.unidentified, actual.unidentified);
                                // Independent original loading moments, including zero-weight levels.
                                for levels in [&a, &b] {
                                    for level in 0..=levels.iter().copied().max().unwrap() {
                                        let mut moment = 0.0;
                                        let mut intercept_moment = 0.0;
                                        for i in 0..n {
                                            if levels[i] == level {
                                                let wr = w.as_ref().map_or(1.0, |ws| ws[i])
                                                    * actual.demeaned[i];
                                                moment += wr * z[i];
                                                intercept_moment += wr;
                                            }
                                        }
                                        assert!(moment.abs() < 1e-7);
                                        if intercept {
                                            assert!(intercept_moment.abs() < 1e-7);
                                        }
                                    }
                                }
                                if let Some(prebuilt) = default.preconditioner() {
                                    let reused = Solver::new_with_screening(
                                        effects(),
                                        w.clone(),
                                        prebuilt,
                                        SlopeScreening::Disabled,
                                    )
                                    .unwrap()
                                    .with_reproducible()
                                    .unwrap()
                                    .solve(&y, &options)
                                    .unwrap();
                                    assert!(reused.warnings.is_empty());
                                    assert_eq!(bits(&reused.x), bits(&actual.x));
                                    assert_eq!(bits(&reused.demeaned), bits(&actual.demeaned));
                                    let diagnostic =
                                        Solver::new(effects(), w.clone(), prebuilt).unwrap();
                                    assert!(diagnostic.warnings().iter().any(|warning| matches!(
                                        warning,
                                        BuildWarning::CollinearSlopeCovariate { .. }
                                    )));
                                }
                            }
                        }
                    }
                }
            });
    }
}

#[test]
fn one_shot_batch_policy_keeps_default_warnings_and_explicit_disabled_outputs() {
    let a = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1];
    let b = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
    let z = [1., 2., 3., 4., 5., 6., 2., 4., 6., 8., 10., 12.];
    let y = [3., 1., 4., 1., 5., 9., 2., 6., 5., 3., 5., 8.];
    let effects = || {
        vec![
            Effect::new(&a, true, [&z[..]]).unwrap(),
            Effect::new(&b, true, [&z[..]]).unwrap(),
        ]
    };
    let options = LsmrOptions {
        tol: 1e-11,
        maxiter: 100,
        local_size: Some(12),
    };
    let default = within::solve_batch(
        effects(),
        &[&y[..]],
        None,
        &options,
        PreconditionerConfig::Off,
    )
    .unwrap();
    let explicit = within::solve_batch_with_screening(
        effects(),
        &[&y[..]],
        None,
        &options,
        PreconditionerConfig::Off,
        SlopeScreening::Enabled,
    )
    .unwrap();
    let disabled = within::solve_batch_with_screening(
        effects(),
        &[&y[..]],
        None,
        &options,
        PreconditionerConfig::Off,
        SlopeScreening::Disabled,
    )
    .unwrap();
    assert!(!default.warnings.is_empty());
    assert_eq!(default.warnings, explicit.warnings);
    assert!(disabled.warnings.is_empty());
    assert_eq!(bits(&default.x), bits(&explicit.x));
    assert_eq!(bits(&default.x), bits(&disabled.x));
    assert_eq!(bits(&default.demeaned), bits(&disabled.demeaned));
    assert_eq!(default.iterations, disabled.iterations);
    assert_eq!(default.unidentified, disabled.unidentified);
}
