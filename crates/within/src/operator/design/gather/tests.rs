use std::ops::Range;

use crate::config::LocalSolverConfig;
use crate::domain::{Design, Effect, PreparedDesign};
use crate::operator::schwarz::{build_diagonal, build_schwarz};
use crate::operator::DesignOperator;
use schwarz_precond::{lsmr, mlsmr, LsmrResult, MlsmrOptions, Operator, SolveError};

fn assert_bits(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "row or coefficient {i}"
        );
    }
}

/// An explicit matrix, with term groups retaining the specified product order.
/// It has no gather kernel, chunking, or destination-initialization dependency.
struct DenseForward<'p, 'a> {
    prepared: &'p PreparedDesign<'a>,
    matrix: Vec<f64>,
    groups: Vec<Range<usize>>,
}

impl<'p, 'a> DenseForward<'p, 'a> {
    fn new(prepared: &'p PreparedDesign<'a>) -> Self {
        let cols = prepared.design.n_dofs;
        let mut matrix = vec![0.0; prepared.design.n_obs * cols];
        let mut groups = Vec::new();
        for term in prepared.terms() {
            groups.push(term.term.dofs());
            for c in 0..term.term.n_columns() {
                let base = term.term.column_dofs(c).start;
                for (i, &level) in term.term.levels().iter().enumerate() {
                    matrix[i * cols + base + level as usize] =
                        term.loading(c).map_or(1.0, |z| z[i]);
                }
            }
        }
        Self {
            prepared,
            matrix,
            groups,
        }
    }
}

impl Operator for DenseForward<'_, '_> {
    fn nrows(&self) -> usize {
        self.prepared.design.n_obs
    }
    fn ncols(&self) -> usize {
        self.prepared.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        for (i, (row, out)) in self.matrix.chunks(self.ncols()).zip(y).enumerate() {
            let mut value = 0.0;
            for group in &self.groups {
                value += row[group.clone()]
                    .iter()
                    .zip(&x[group.clone()])
                    .fold(-0.0, |sum, (&a, &b)| sum + a * b);
            }
            *out = match self.prepared.sqrt_weights() {
                Some(weights) => value * weights[i],
                None => value,
            };
        }
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        DesignOperator::new(self.prepared).apply_adjoint(x, y)
    }
}

#[test]
fn chunk_zero_matches_independent_integer_model_at_boundaries() {
    for n in [
        1, 4095, 4096, 4097, 9999, 10_000, 10_001, 12_288, 12_289, 65_537,
    ] {
        let a: Vec<u32> = (0..n).map(|i| (i * 17 % 11) as u32).collect();
        let b: Vec<u32> = (0..n).map(|i| (i * 7 % 13) as u32).collect();
        let scales: Vec<f64> = (0..n).map(|i| [0.0, 0.5, 1.0, 2.0][i % 4]).collect();
        let weights: Vec<f64> = scales.iter().map(|s| s * s).collect();
        for weighted in [false, true] {
            let design = Design::new(vec![
                Effect::new(&a, true, []).unwrap(),
                Effect::new(&b, true, []).unwrap(),
            ])
            .unwrap();
            let prepared = PreparedDesign::new(design, weighted.then_some(&weights)).unwrap();
            let mut x = Vec::new();
            for (term, t) in prepared.terms().enumerate() {
                x.extend((0..t.term.n_levels()).map(|level| {
                    if term == 0 {
                        (level as i64 * 3 - 17) as f64
                    } else {
                        (level as i64 * 5 - 23) as f64
                    }
                }));
            }
            // Integer arithmetic gives the model before its exact dyadic weight.
            let expected: Vec<f64> = (0..n)
                .map(|i| {
                    let sum = (a[i] as i64 * 3 - 17) + (b[i] as i64 * 5 - 23);
                    sum as f64 * if weighted { scales[i] } else { 1.0 }
                })
                .collect();
            for workers in [1, 4] {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap()
                    .install(|| {
                        for poison in [f64::NAN, f64::INFINITY] {
                            let mut actual = vec![poison; n];
                            DesignOperator::new(&prepared)
                                .apply(&x, &mut actual)
                                .unwrap();
                            assert_bits(&prepared.design.permute_obs_out(actual), &expected);
                        }
                    });
            }
        }
    }
}

#[test]
fn chunk_zero_matches_dense_model_for_all_slope_arities() {
    for n in [9999, 10_000, 10_001, 12_288, 12_289] {
        let levels: Vec<u32> = (0..n).map(|i| (i * 7 % 3) as u32).collect();
        let z: Vec<Vec<f64>> = (0..4)
            .map(|k| {
                (0..n)
                    .map(|i| (((i * (7 + 2 * k) + i / 5) % 37) as f64 - 18.0) / 8.0)
                    .collect()
            })
            .collect();
        let weights: Vec<f64> = (0..n).map(|i| (i % 5) as f64 / 4.0).collect();
        for weighted in [false, true] {
            let mut effects = vec![Effect::new(&levels, true, []).unwrap()];
            for intercept in [false, true] {
                for slopes in 1..=4 {
                    effects.push(
                        Effect::new(&levels, intercept, z[..slopes].iter().map(Vec::as_slice))
                            .unwrap(),
                    );
                }
            }
            let prepared =
                PreparedDesign::new(Design::new(effects).unwrap(), weighted.then_some(&weights))
                    .unwrap();
            let reference = DenseForward::new(&prepared);
            let x: Vec<f64> = (0..reference.ncols())
                .map(|j| [1e16, 1.0, -1e16, -0.0, 0.25, -0.5][j % 6])
                .collect();
            let mut expected = vec![f64::NAN; n];
            reference.apply(&x, &mut expected).unwrap();
            assert!(expected.iter().all(|v| v.is_finite()));
            for workers in [1, 4] {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap()
                    .install(|| {
                        for _ in 0..2 {
                            let mut actual = vec![f64::NAN; n];
                            DesignOperator::new(&prepared)
                                .apply(&x, &mut actual)
                                .unwrap();
                            assert_bits(&actual, &expected);
                        }
                    });
            }
        }
    }
}

fn assert_same_solve(actual: &LsmrResult, expected: &LsmrResult) {
    assert_bits(&actual.x, &expected.x);
    assert_eq!(actual.converged, expected.converged);
    assert_eq!(actual.iterations, expected.iterations);
    assert_eq!(actual.stop_reason, expected.stop_reason);
    assert_bits(
        &[actual.residual_norm, actual.normal_eq_residual],
        &[expected.residual_norm, expected.normal_eq_residual],
    );
    match (&actual.true_residual, &expected.true_residual) {
        (Some(a), Some(b)) => assert_bits(a, b),
        (None, None) => {}
        _ => panic!("true-residual availability changed"),
    }
}

#[test]
fn chunk_zero_preserves_solve_signatures_against_dense_forward() {
    let n = 10_017;
    let a: Vec<u32> = (0..n).map(|i| (i * 7 / n) as u32).collect();
    let b: Vec<u32> = (0..n).map(|i| (i % 5) as u32).collect();
    let c: Vec<u32> = (0..n).map(|i| (i * 7 % 3) as u32).collect();
    let z: Vec<f64> = (0..n).map(|i| ((i % 17) as f64 - 8.0) / 8.0).collect();
    let weights: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
    for weighted in [false, true] {
        for sloped in [false, true] {
            let prepared = PreparedDesign::new(
                Design::new(vec![
                    Effect::new(&a, true, []).unwrap(),
                    Effect::new(&b, true, []).unwrap(),
                    Effect::new(&c, true, sloped.then_some(z.as_slice())).unwrap(),
                ])
                .unwrap(),
                weighted.then_some(&weights),
            )
            .unwrap();
            let operator = DesignOperator::new(&prepared);
            let reference = DenseForward::new(&prepared);
            let y: Vec<f64> = (0..n)
                .map(|i| ((i * 29 % 97) as f64 - 48.0) / 16.0)
                .collect();
            let rhs = operator.weighted_rhs(&y);
            let mut across_workers: Option<Vec<LsmrResult>> = None;
            for workers in [1, 4] {
                let results = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap()
                    .install(|| {
                        let diagonal = build_diagonal(&prepared).unwrap();
                        let (schwarz, _) = build_schwarz(
                            &prepared,
                            &LocalSolverConfig::default(),
                            crate::Cancellation::default(),
                        )
                        .unwrap();
                        let schwarz = schwarz.expect("three-factor Schwarz");
                        let mut results = Vec::new();
                        for preconditioner in [None, Some(&diagonal), Some(&schwarz)] {
                            let solve = |op: &dyn Operator| match preconditioner {
                                None => lsmr(op, &rhs, 1e-10, 100, Some(2)),
                                Some(p) => mlsmr(
                                    op,
                                    &rhs,
                                    p,
                                    1e-10,
                                    100,
                                    MlsmrOptions {
                                        local_size: Some(2),
                                        ..Default::default()
                                    },
                                ),
                            };
                            let expected = solve(&reference).unwrap();
                            let actual = solve(&operator).unwrap();
                            assert!(actual.converged, "{weighted} {sloped} {workers}");
                            assert_same_solve(&actual, &expected);
                            assert_same_solve(&solve(&operator).unwrap(), &actual);
                            results.push(actual);
                        }
                        results
                    });
                if let Some(expected) = &across_workers {
                    for (actual, expected) in results.iter().zip(expected) {
                        assert_same_solve(actual, expected);
                    }
                } else {
                    across_workers = Some(results);
                }
            }
        }
    }
}
