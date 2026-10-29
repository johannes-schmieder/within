use super::scatter_apply;
use crate::config::LocalSolverConfig;
use crate::domain::{AdjointSum, Design, Effect, PreparedDesign};
use crate::operator::schwarz::{build_diagonal, build_schwarz};
use crate::operator::DesignOperator;
use schwarz_precond::{lsmr, mlsmr, LsmrResult, MlsmrOptions, Operator, SolveError};

fn assert_bits(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "slot {i}");
    }
}

/// Retain the production generic branch as the arithmetic-order reference.
fn generic_adjoint(prepared: &PreparedDesign<'_>, x: &[f64], out: &mut [f64]) {
    match prepared.sqrt_weights() {
        Some(sw) => scatter_apply(prepared, out, &|i| sw[i] * x[i], None),
        None => scatter_apply(prepared, out, &|i| x[i], None),
    }
}

fn panel(n: usize, sloped: bool, weighted: bool) -> PreparedDesign<'static> {
    let owner: Vec<u32> = (0..n).map(|i| (i * 512 / n) as u32).collect();
    let fold: Vec<u32> = (0..n).map(|i| ((i * 17) % 257) as u32).collect();
    let other: Vec<u32> = (0..n).map(|i| ((i * 7) % 17) as u32).collect();
    let slope: Vec<f64> = (0..n).map(|i| ((i % 19) as f64 - 9.0) / 8.0).collect();
    let mut effects = vec![
        Effect::new(&owner, true, []).unwrap(),
        Effect::new(&fold, true, []).unwrap(),
    ];
    effects.push(if sloped {
        Effect::new(&other, true, [&slope[..]]).unwrap()
    } else {
        Effect::new(&other, true, []).unwrap()
    });
    let design = Design::new(effects).unwrap().into_owned();
    let weights: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let prepared = PreparedDesign::new(design, weighted.then_some(&weights)).unwrap();
    assert!(prepared.design.obs_perm.is_none());
    assert!(matches!(
        prepared.term(0).term.adjoint(),
        AdjointSum::Owner(_)
    ));
    assert!(matches!(prepared.term(1).term.adjoint(), AdjointSum::Fold));
    assert!(matches!(prepared.term(2).term.adjoint(), AdjointSum::Fold));
    prepared
}

#[test]
fn categorical_fold_slices_match_generic_at_leaf_boundaries() {
    for n in [65_535, 65_536, 65_537, 131_071, 131_072, 131_073, 196_621] {
        let prepared = panel(n, false, false);
        // Mixed magnitudes make a changed row order or tree observable.
        let x: Vec<f64> = (0..n)
            .map(|i| match (i / 257) % 6 {
                0 => 1e16,
                1 => 1.0,
                2 => -1e16,
                3 => -0.0,
                4 => -0.25,
                _ => 0.5,
            })
            .collect();
        let mut reference = vec![0.0; prepared.design.n_dofs];
        generic_adjoint(&prepared, &x, &mut reference);
        for workers in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    let operator = DesignOperator::new(&prepared);
                    for _ in 0..2 {
                        let mut actual = vec![f64::NAN; reference.len()];
                        operator.apply_adjoint(&x, &mut actual).unwrap();
                        assert_bits(&actual, &reference);
                    }
                });
        }
    }
}

#[test]
fn categorical_fold_slices_match_independent_integer_group_sums() {
    let prepared = panel(196_621, false, false);
    let numerators: Vec<i64> = (0..prepared.design.n_obs)
        .map(|i| (i % 127) as i64 - 63)
        .collect();
    let x: Vec<f64> = numerators.iter().map(|&v| v as f64 / 8.0).collect();
    // Integer accumulation is independent of the floating-point reduction tree.
    let mut expected = vec![0.0; prepared.design.n_dofs];
    for term in prepared.terms() {
        let mut sums = vec![0i64; term.term.n_levels()];
        for (&level, &value) in term.term.levels().iter().zip(&numerators) {
            sums[level as usize] += value;
        }
        for (slot, sum) in expected[term.term.dofs()].iter_mut().zip(sums) {
            *slot = sum as f64 / 8.0;
        }
    }
    let mut actual = vec![0.0; expected.len()];
    DesignOperator::new(&prepared)
        .apply_adjoint(&x, &mut actual)
        .unwrap();
    assert_bits(&actual, &expected);
}

struct GenericOperator<'a>(&'a PreparedDesign<'a>);

impl Operator for GenericOperator<'_> {
    fn nrows(&self) -> usize {
        self.0.design.n_obs
    }

    fn ncols(&self) -> usize {
        self.0.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        DesignOperator::new(self.0).apply(x, y)
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        generic_adjoint(self.0, x, y);
        Ok(())
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
        (Some(actual), Some(expected)) => assert_bits(actual, expected),
        (None, None) => {}
        _ => panic!("true-residual availability changed"),
    }
}

#[test]
fn categorical_fold_slices_preserve_mixed_design_solve_results() {
    for weighted in [false, true] {
        for sloped in [false, true] {
            let prepared = panel(131_089, sloped, weighted);
            let operator = DesignOperator::new(&prepared);
            let reference = GenericOperator(&prepared);
            let y: Vec<f64> = (0..prepared.design.n_obs)
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
                        let mut direct = vec![0.0; operator.ncols()];
                        let mut generic = vec![0.0; operator.ncols()];
                        operator.apply_adjoint(&rhs, &mut direct).unwrap();
                        reference.apply_adjoint(&rhs, &mut generic).unwrap();
                        assert_bits(&direct, &generic);

                        let diagonal = build_diagonal(&prepared).unwrap();
                        let (schwarz, _) =
                            build_schwarz(&prepared, &LocalSolverConfig::default()).unwrap();
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
                if let Some(reference) = &across_workers {
                    for (actual, expected) in results.iter().zip(reference) {
                        assert_same_solve(actual, expected);
                    }
                } else {
                    across_workers = Some(results);
                }
            }
        }
    }
}
