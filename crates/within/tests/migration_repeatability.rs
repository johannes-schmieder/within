//! Downstream witness for the pending #472 stack; this is not timing evidence.
use rayon::ThreadPoolBuilder;
use within::{Effect, LocalSolverConfig, LsmrOptions, PreconditionerConfig, SolveResult, Solver};

fn noise(i: usize, stream: u64) -> f64 {
    let mut x = (i as u64).wrapping_add(stream.wrapping_mul(0x9e3779b97f4a7c15));
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((x ^ (x >> 31)) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

#[derive(PartialEq)]
struct Fingerprint {
    floats: Vec<u64>,
    iterations: usize,
    converged: bool,
    discrete: String,
}

fn fingerprint(r: &SolveResult) -> Fingerprint {
    Fingerprint {
        floats: r
            .x
            .iter()
            .chain(&r.demeaned)
            .chain([&r.residual])
            .map(|x| x.to_bits())
            .collect(),
        iterations: r.iterations,
        converged: r.converged,
        discrete: format!("{:?}|{:?}", r.unidentified, r.warnings),
    }
}

#[test]
fn fixed_worker_repeats_cross_worker_decisions_and_reused_batch_agree() {
    let n = 70_000;
    let levels = [
        (0..n).map(|i| (i / 10) as u32).collect::<Vec<_>>(),
        (0..n)
            .map(|i| ((i / 10 * 7919 + i % 10 * 101) % 997) as u32)
            .collect(),
        (0..n).map(|i| (i % 10) as u32).collect(),
    ];
    let z: Vec<Vec<f64>> = (1..=3)
        .map(|s| (0..n).map(|i| noise(i, s)).collect())
        .collect();
    let y: Vec<_> = (0..n).map(|i| noise(i, 5)).collect();
    let other: Vec<_> = (0..n).map(|i| noise(i, 6)).collect();
    let weights: Vec<_> = (0..n).map(|i| 1.0 + noise(i, 7)).collect();
    let options = LsmrOptions {
        tol: 1e-12,
        maxiter: 16,
        local_size: Some(4),
    };
    // A sorted term and both dense and sparse unsorted terms; both fused slope arities.
    for sloped_term in 0..3 {
        for slopes in [1, 3] {
            for preconditioner in [
                PreconditionerConfig::Off,
                PreconditionerConfig::Diagonal,
                PreconditionerConfig::Additive {
                    local_solver: LocalSolverConfig::default(),
                },
            ] {
                let weighted = (sloped_term + slopes) % 2 == 0;
                let label = format!("term={sloped_term}, slopes={slopes}, weighted={weighted}, preconditioner={preconditioner:?}");
                let mut reference = None;
                for workers in [1, 2, 3, 8] {
                    let pool = ThreadPoolBuilder::new()
                        .num_threads(workers)
                        .build()
                        .unwrap();
                    let first = pool.install(|| {
                        let effects = (0..3)
                            .map(|term| {
                                let columns = if term == sloped_term {
                                    z[..slopes].iter().map(Vec::as_slice).collect()
                                } else {
                                    Vec::new()
                                };
                                Effect::new(&levels[term], true, columns).unwrap()
                            })
                            .collect::<Vec<_>>();
                        let solver = Solver::new(
                            effects,
                            weighted.then_some(weights.as_slice()),
                            &preconditioner,
                        )
                        .unwrap();
                        let first = solver.solve(&y, &options).unwrap();
                        let middle = solver.solve(&other, &options).unwrap();
                        assert!(middle
                            .x
                            .iter()
                            .chain(&middle.demeaned)
                            .all(|v| v.is_finite()));
                        let again = solver.solve(&y, &options).unwrap();
                        assert!(
                            fingerprint(&first) == fingerprint(&again),
                            "fixed-worker mismatch: {label}, workers={workers}"
                        );
                        let batch = solver.solve_batch(&[&y, &other, &y], &options).unwrap();
                        for col in [0, 2] {
                            assert!(
                                batch.x[col * first.x.len()..(col + 1) * first.x.len()]
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .eq(first.x.iter().map(|v| v.to_bits())),
                                "batch coefficients: {label}, workers={workers}"
                            );
                            assert!(
                                batch.demeaned[col * n..(col + 1) * n]
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .eq(first.demeaned.iter().map(|v| v.to_bits())),
                                "batch residuals: {label}, workers={workers}"
                            );
                            assert_eq!(batch.iterations[col], first.iterations);
                            assert_eq!(batch.converged[col], first.converged);
                            assert_eq!(batch.residual[col].to_bits(), first.residual.to_bits());
                        }
                        fingerprint(&first)
                    });
                    if let Some(ref previous) = reference {
                        assert!(
                            previous == &first,
                            "cross-worker mismatch: {label}, workers={workers}"
                        );
                    } else {
                        reference = Some(first);
                    }
                }
            }
        }
    }
}
