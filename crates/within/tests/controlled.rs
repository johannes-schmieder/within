//! The caller's staging policy, coefficient frame, and independent least-squares witnesses.
use std::sync::{Arc, Mutex};
use within::{
    Effect, EscalationHandler, EscalationPolicy, LsmrOptions, LsmrStopReason, PreconditionerConfig,
    Progress, SolveControl, Solver,
};

struct StopAfter {
    iterations: usize,
    seen: Arc<Mutex<Vec<usize>>>,
}
impl EscalationPolicy for StopAfter {
    fn handler(&self) -> Box<dyn EscalationHandler> {
        Box::new(Self {
            iterations: self.iterations,
            seen: self.seen.clone(),
        })
    }
}
impl EscalationHandler for StopAfter {
    fn should_escalate(&mut self, progress: Progress) -> bool {
        self.seen.lock().unwrap().push(progress.iteration);
        progress.iteration >= self.iterations
    }
}

fn options(maxiter: usize) -> LsmrOptions {
    LsmrOptions {
        tol: 1e-12,
        maxiter,
        local_size: None,
    }
}
fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 3e-9 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn caller_handoff_keeps_shared_budget_original_response_and_total_coefficients() {
    let n = 900;
    let f: Vec<_> = (0..n).map(|i| (i * 7 % 31) as u32).collect();
    let g: Vec<_> = (0..n).map(|i| ((i / 3 + i % 7) % 23) as u32).collect();
    let y: Vec<_> = (0..n)
        .map(|i| (i as f64 * 0.7).sin() + (i % 19) as f64)
        .collect();
    let w: Vec<_> = (0..n).map(|i| 0.1 + (i % 9) as f64).collect();
    let effects = || {
        vec![
            Effect::new(&f, true, []).unwrap(),
            Effect::new(&g, true, []).unwrap(),
        ]
    };
    let diagonal = Solver::new(effects(), Some(&w), PreconditionerConfig::Diagonal).unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let stop = StopAfter {
        iterations: 2,
        seen: seen.clone(),
    };
    let first = diagonal
        .solve_controlled(
            &y,
            &options(100),
            SolveControl {
                initial_coefficients: None,
                escalation: Some(&stop),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(first.stop_reason, LsmrStopReason::Escalated);
    assert_eq!(first.solution.iterations, 2);
    assert!(!first.solution.converged);
    assert_eq!(*seen.lock().unwrap(), [1, 2]);
    let additive = Solver::new(
        effects(),
        Some(&w),
        PreconditionerConfig::Additive {
            local_solver: Default::default(),
            reduction: within::ReductionStrategy::ParallelReduction,
        },
    )
    .unwrap();
    let last = additive
        .solve_controlled(
            &y,
            &options(100 - first.solution.iterations),
            SolveControl {
                initial_coefficients: Some(&first.solution.x),
                escalation: None,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(last.solution.converged);
    assert!(last.solution.iterations + first.solution.iterations <= 100);
    assert_ne!(last.stop_reason, LsmrStopReason::Escalated);
    // Independent original-coordinate reconstruction and weighted normal equations.
    let mut gradient = vec![0.0; 54];
    for i in 0..n {
        let fit = last.solution.x[f[i] as usize] + last.solution.x[31 + g[i] as usize];
        close(last.solution.demeaned[i], y[i] - fit);
        gradient[f[i] as usize] += w[i] * (y[i] - fit);
        gradient[31 + g[i] as usize] += w[i] * (y[i] - fit);
    }
    assert!(gradient.iter().all(|v| v.abs() < 1e-7), "{gradient:?}");
    let cold = additive.solve(&y, &options(100)).unwrap();
    for (&a, &b) in last.solution.demeaned.iter().zip(&cold.demeaned) {
        close(a, b);
    }
    // The same API serves a tighter correction after the caller's own certificate.
    let corrected = additive
        .solve_controlled(
            &y,
            &options(100 - first.solution.iterations - last.solution.iterations),
            SolveControl {
                initial_coefficients: Some(&last.solution.x),
                escalation: None,
                ..Default::default()
            },
        )
        .unwrap();
    for (&a, &b) in corrected.solution.demeaned.iter().zip(&cold.demeaned) {
        close(a, b);
    }
}

#[test]
fn zero_budget_reports_original_rhs_reference_and_preserves_nonunique_seed_gauge() {
    let f = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1];
    let g = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    let y = [2.0, 1.0, 5.0, 9.0, 4.0, 1.0, 3.0, 7.0, 3.0, 9.0, 2.0, 5.0];
    let w = [1.0, 4.0, 1.0, 9.0, 4.0, 1.0, 1.0, 4.0, 9.0, 4.0, 1.0, 9.0];
    let solver = Solver::new(
        vec![
            Effect::new(&f, true, []).unwrap(),
            Effect::new(&g, true, []).unwrap(),
        ],
        Some(&w),
        PreconditionerConfig::Off,
    )
    .unwrap();
    let seed = [8.0, 9.0, 10.0, -7.0, -6.0];
    let result = solver
        .solve_controlled(
            &y,
            &options(0),
            SolveControl {
                initial_coefficients: Some(&seed),
                escalation: None,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(result.solution.x, seed);
    assert_eq!(result.solution.iterations, 0);
    assert_eq!(result.stop_reason, LsmrStopReason::MaxIterations);
    let mut numerator = [0.0f64; 5];
    let mut denominator = [0.0f64; 5];
    for i in 0..y.len() {
        let r = y[i] - seed[f[i] as usize] - seed[3 + g[i] as usize];
        close(result.solution.demeaned[i], r);
        for slot in [f[i] as usize, 3 + g[i] as usize] {
            numerator[slot] += w[i] * r;
            denominator[slot] += w[i] * y[i];
        }
    }
    let norm = |x: &[f64]| x.iter().map(|x| x * x).sum::<f64>().sqrt();
    close(
        result.solution.residual,
        norm(&numerator) / norm(&denominator),
    );
}

// Independent dense normal equations for a small, well-conditioned 3-column regression.
fn dense_reference(rows: &[[f64; 3]], y: &[f64], w: &[f64]) -> [f64; 3] {
    let mut a = [[0.0; 4]; 3];
    for ((row, y), w) in rows.iter().zip(y).zip(w) {
        for j in 0..3 {
            for k in 0..3 {
                a[j][k] += w * row[j] * row[k];
            }
            a[j][3] += w * row[j] * y;
        }
    }
    for j in 0..3 {
        let pivot = (j..3)
            .max_by(|&i, &k| a[i][j].abs().total_cmp(&a[k][j].abs()))
            .unwrap();
        a.swap(j, pivot);
        let diagonal = a[j][j];
        for k in j..4 {
            a[j][k] /= diagonal;
        }
        for i in 0..3 {
            if i != j {
                let factor = a[i][j];
                for k in j..4 {
                    a[i][k] -= factor * a[j][k];
                }
            }
        }
    }
    [a[0][3], a[1][3], a[2][3]]
}

#[test]
fn weighted_two_slope_seeds_round_trip_whitening_pivot_order_and_locality_sort() {
    let n = 120;
    let f: Vec<_> = (0..n).map(|i| ((i * 7) % 3) as u32).collect();
    let z0: Vec<_> = (0..n).map(|i| 2.0 + (i as f64 * 0.7).sin()).collect();
    let z1: Vec<_> = (0..n).map(|i| 5.0 + 3.0 * (i as f64 * 1.3).cos()).collect();
    let y: Vec<_> = (0..n)
        .map(|i| (i as f64 * 0.1).sin() + 0.2 * z0[i] - z1[i])
        .collect();
    let weights: Vec<_> = (0..n)
        .map(|i| {
            if i % 11 == 0 {
                0.0
            } else {
                0.5 + (i % 5) as f64
            }
        })
        .collect();
    for w in [vec![1.0; n], weights] {
        for config in [PreconditionerConfig::Off, PreconditionerConfig::Diagonal] {
            let solver = Solver::new(
                vec![Effect::new(&f, true, [&z0[..], &z1[..]]).unwrap()],
                Some(&w),
                config,
            )
            .unwrap();
            let seed = [0.2, -1.0, 3.0, 0.5, -0.2, 0.1, -2.0, 0.7, 0.9];
            let unchanged = solver
                .solve_controlled(
                    &y,
                    &options(0),
                    SolveControl {
                        initial_coefficients: Some(&seed),
                        escalation: None,
                        ..Default::default()
                    },
                )
                .unwrap();
            for (&a, &b) in unchanged.solution.x.iter().zip(&seed) {
                close(a, b);
            }
            for i in 0..n {
                let l = f[i] as usize;
                close(
                    unchanged.solution.demeaned[i],
                    y[i] - seed[l] - z0[i] * seed[3 + l] - z1[i] * seed[6 + l],
                );
            }
            let result = solver
                .solve_controlled(
                    &y,
                    &options(100),
                    SolveControl {
                        initial_coefficients: Some(&seed),
                        escalation: None,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(result.solution.converged);
            for l in 0..3 {
                let indices: Vec<_> = (0..n).filter(|&i| f[i] == l as u32).collect();
                let rows: Vec<_> = indices.iter().map(|&i| [1.0, z0[i], z1[i]]).collect();
                let yl: Vec<_> = indices.iter().map(|&i| y[i]).collect();
                let wl: Vec<_> = indices.iter().map(|&i| w[i]).collect();
                let reference = dense_reference(&rows, &yl, &wl);
                for (j, &value) in reference.iter().enumerate() {
                    close(result.solution.x[j * 3 + l], value);
                }
            }
        }
    }
}

#[test]
fn invalid_seed_and_implicit_adaptation_fail_without_spending_iterations() {
    let f = [0, 1, 0, 1, 0, 1];
    let g = [0, 0, 1, 1, 0, 1];
    let z = [2.0; 6]; // Every slope is unidentified with an intercept.
    let y = [1.0, 2.0, 3.0, 0.0, 1.0, 2.0];
    let solver = Solver::new(
        vec![Effect::new(&f, true, [&z[..]]).unwrap()],
        None,
        PreconditionerConfig::Off,
    )
    .unwrap();
    for seed in [vec![0.0], vec![f64::NAN; 4], vec![0.0, 0.0, 1.0, 0.0]] {
        let error = solver
            .solve_controlled(
                &y,
                None,
                SolveControl {
                    initial_coefficients: Some(&seed),
                    escalation: None,
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert_eq!(error.iterations, 0);
    }
    let allowed = solver
        .solve_controlled(
            &y,
            None,
            SolveControl {
                initial_coefficients: Some(&[1.0, 2.0, 0.0, 0.0]),
                escalation: None,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(&allowed.solution.x[2..], &[0.0, 0.0]);
    let adaptive = Solver::new(
        vec![
            Effect::new(&f, true, []).unwrap(),
            Effect::new(&g, true, []).unwrap(),
        ],
        None,
        None,
    )
    .unwrap();
    let error = adaptive
        .solve_controlled(&y, None, SolveControl::default())
        .unwrap_err();
    assert_eq!(error.iterations, 0);
    assert!(!adaptive.has_escalated());
}

#[test]
fn controlled_cold_calls_preserve_ordinary_science_bits() {
    let n = 3000;
    let f: Vec<_> = (0..n).map(|i| (i * 7 % 41) as u32).collect();
    let g: Vec<_> = (0..n).map(|i| ((i / 3 + i % 5) % 29) as u32).collect();
    let y: Vec<_> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
    let weights: Vec<_> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            for w in [None, Some(&weights[..])] {
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
                        ],
                        w,
                        config,
                    )
                    .unwrap();
                    let plain = solver.solve(&y, &options(100)).unwrap();
                    for _ in 0..2 {
                        let result = solver
                            .solve_controlled(&y, &options(100), SolveControl::default())
                            .unwrap()
                            .solution;
                        assert_eq!(plain.x, result.x);
                        assert_eq!(plain.demeaned, result.demeaned);
                        assert_eq!(plain.iterations, result.iterations);
                        assert_eq!(plain.residual.to_bits(), result.residual.to_bits());
                        assert_eq!(plain.converged, result.converged);
                    }
                }
            }
        });
    }
}

#[test]
fn zero_response_corrects_warm_seed_and_exact_seed_keeps_gauge() {
    let f = [0, 1, 0, 1];
    let g = [0, 0, 1, 1];
    let solver = Solver::new(
        vec![
            Effect::new(&f, true, []).unwrap(),
            Effect::new(&g, true, []).unwrap(),
        ],
        None,
        PreconditionerConfig::Diagonal,
    )
    .unwrap();
    let exact = [3.0, 4.0, -2.0, -1.0];
    let y = [1.0, 2.0, 2.0, 3.0];
    let result = solver
        .solve_controlled(
            &y,
            &options(10),
            SolveControl {
                initial_coefficients: Some(&exact),
                escalation: None,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(result.stop_reason, LsmrStopReason::WarmStartExact);
    assert_eq!(result.solution.x, exact);
    assert_eq!(result.solution.iterations, 0);
    let result = solver
        .solve_controlled(
            &[0.0; 4],
            &options(10),
            SolveControl {
                initial_coefficients: Some(&exact),
                escalation: None,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(result.solution.converged);
    assert!(result.solution.demeaned.iter().all(|v| v.abs() < 1e-12));
}

#[test]
fn slope_only_seed_uses_original_units() {
    let f = [1, 0, 1, 0, 1, 0];
    let z = [2.0, 0.5, 3.0, 1.0, 4.0, 2.0];
    let w = [1.0, 4.0, 9.0, 1.0, 4.0, 1.0];
    let y = [1.0, 3.0, 2.0, -1.0, 4.0, 2.0];
    let solver = Solver::new(
        vec![Effect::new(&f, false, [&z[..]]).unwrap()],
        Some(&w),
        PreconditionerConfig::Diagonal,
    )
    .unwrap();
    let result = solver
        .solve_controlled(
            &y,
            &options(10),
            SolveControl {
                initial_coefficients: Some(&[7.0, -3.0]),
                escalation: None,
                ..Default::default()
            },
        )
        .unwrap()
        .solution;
    for l in 0..2 {
        let numerator: f64 = (0..6)
            .filter(|&i| f[i] == l)
            .map(|i| w[i] * z[i] * y[i])
            .sum();
        let denominator: f64 = (0..6)
            .filter(|&i| f[i] == l)
            .map(|i| w[i] * z[i] * z[i])
            .sum();
        close(result.x[l as usize], numerator / denominator);
    }
}
