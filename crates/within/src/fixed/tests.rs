use super::*;
use crate::domain::{build_local_domains, Design};
use crate::{Effect, LocalSolverConfig};
use schwarz_precond::SubdomainCore;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) struct ProjectedDiagonal(pub usize);
impl FixedReducedInverse for ProjectedDiagonal {
    fn dimension(&self) -> usize {
        self.0
    }
    fn scratch_len(&self) -> usize {
        0
    }
    fn apply_work_bound(&self) -> Option<u64> {
        (self.0 as u64).checked_mul(16)
    }
    fn apply_in_place(&self, rhs: &mut [f64], _: &mut [f64]) -> Result<(), LocalSolveError> {
        let mean = rhs.iter().sum::<f64>() / rhs.len() as f64;
        for (i, value) in rhs.iter_mut().enumerate() {
            *value = (*value - mean) * (1.0 + i as f64 / 4.0);
        }
        let mean = rhs.iter().sum::<f64>() / rhs.len() as f64;
        for value in rhs {
            *value -= mean;
        }
        Ok(())
    }
}

pub(crate) struct ProjectedFactory;
impl FixedReducedInverseFactory for ProjectedFactory {
    fn build(
        &self,
        component: PairComponent<'_>,
    ) -> Result<Arc<dyn FixedReducedInverse>, BuildError> {
        Ok(Arc::new(ProjectedDiagonal(component.n_retained)))
    }
}

fn multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..a.len())
        .map(|i| {
            (0..b[0].len())
                .map(|j| (0..b.len()).map(|k| a[i][k] * b[k][j]).sum())
                .collect()
        })
        .collect()
}
fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..a[0].len())
        .map(|j| a.iter().map(|row| row[j]).collect())
        .collect()
}
fn eye(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| f64::from(i == j)).collect())
        .collect()
}
fn projector(n: usize) -> Vec<Vec<f64>> {
    eye(n)
        .into_iter()
        .map(|r| r.into_iter().map(|v| v - 1.0 / n as f64).collect())
        .collect()
}
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(x, y)| x * y).sum()
}
fn apply(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, x)).collect()
}

#[test]
fn wrapped_action_matches_dense_congruence_with_uneven_weights_and_partition_metric() {
    // Complete positive 4-by-3 graph with deliberately uneven observation weights.
    let a: Vec<u32> = (0..4).flat_map(|i| [i; 3]).collect();
    let b: Vec<u32> = (0..4).flat_map(|_| 0..3).collect();
    let weights: Vec<f64> = (0..12).map(|i| 0.2 + (i * i + 3) as f64 / 7.0).collect();
    let design = Design::new(vec![
        Effect::new(&a, true, []).unwrap(),
        Effect::new(&b, true, []).unwrap(),
    ])
    .unwrap();
    let (mut domains, _) =
        build_local_domains(&design, Some(&weights), &LocalSolverConfig::default()).unwrap();
    let domain = domains.pop().unwrap();
    let n = 7;
    let c = &domain.component.matrix.cross_tab.c;
    let diagonal = &domain.component.matrix.diagonal;
    let mut t = eye(n);
    for i in 0..4 {
        for (j, w) in c.row(i) {
            t[i][4 + j] = w / diagonal[i];
        }
    }
    let mut block = vec![vec![0.0; n]; n];
    for i in 0..4 {
        block[i][i] = 1.0 / diagonal[i];
    }
    let mut d = eye(3);
    for (i, row) in d.iter_mut().enumerate() {
        row[i] = 1.0 + i as f64 / 4.0;
    }
    let reduced = multiply(&multiply(&projector(3), &d), &projector(3));
    for i in 0..3 {
        for j in 0..3 {
            block[4 + i][4 + j] = reduced[i][j];
        }
    }
    let g = multiply(&multiply(&t, &block), &transpose(&t));
    let p = projector(n);
    let mut wrap = eye(n);
    let partition: Vec<f64> = (0..n).map(|i| 0.5 + i as f64 / 5.0).collect();
    for i in 0..n {
        wrap[i][i] = partition[i] * if i < 4 { 1.0 } else { -1.0 };
    }
    let oracle = multiply(&multiply(&multiply(&multiply(&wrap, &p), &g), &p), &wrap);
    let solver =
        crate::block_elim::BlockElimSolver::build(domain.component, &LocalSolverConfig::default())
            .unwrap()
            .with_fixed_inverse(Arc::new(ProjectedDiagonal(3)), &ProjectedFactory)
            .unwrap();
    let core = SubdomainCore::with_partition_weights(
        domain.core.global_indices().to_vec(),
        schwarz_precond::PartitionWeights::NonUniform(partition.clone()),
    )
    .unwrap();
    let entry = SubdomainEntry::try_new(core, solver).unwrap();
    let local_apply = |local: &[f64]| {
        let mut global = vec![0.0; n];
        for (i, &g) in entry.global_indices().iter().enumerate() {
            global[g as usize] = local[i];
        }
        let mut output = vec![0.0; n];
        entry
            .apply_weighted_into_with_scratch(
                &global,
                &mut output,
                &mut vec![0.0; entry.scratch_size()],
                &mut vec![0.0; entry.scratch_size()],
                false,
            )
            .unwrap();
        entry
            .global_indices()
            .iter()
            .map(|&g| output[g as usize])
            .collect::<Vec<_>>()
    };
    let x = vec![0.2, -1.0, 0.7, 0.1, -0.4, 1.3, 0.8];
    let y = vec![0.9, 0.3, -0.4, 1.2, 0.0, -0.1, 0.2];
    let mx = local_apply(&x);
    let my = local_apply(&y);
    for (got, want) in mx.iter().zip(apply(&oracle, &x)) {
        assert!((got - want).abs() < 2e-13);
    }
    assert!((dot(&x, &my) - dot(&mx, &y)).abs() < 2e-13);
    assert!(dot(&x, &mx) > 0.0);
    let combined: Vec<_> = x.iter().zip(&y).map(|(a, b)| 0.6 * a - 0.3 * b).collect();
    for (i, value) in local_apply(&combined).iter().enumerate() {
        assert!((value - (0.6 * mx[i] - 0.3 * my[i])).abs() < 2e-13);
    }
    let kernel: Vec<_> = partition
        .iter()
        .enumerate()
        .map(|(i, w)| if i < 4 { 1.0 / w } else { -1.0 / w })
        .collect();
    assert!(local_apply(&kernel).iter().all(|x| x.abs() < 2e-13));
    // Explicit basis confirms strictly positive energy on every independent
    // direction modulo the one-dimensional transformed kernel.
    for j in 1..n {
        let mut v = vec![0.0; n];
        v[j] = 1.0;
        v[0] = -kernel[j] / kernel[0];
        assert!(dot(&v, &local_apply(&v)) > 0.0);
    }
}

#[test]
fn nomination_tie_uses_total_retained_dimension_not_total_local_size() {
    // A has ten levels, B six, C seven. AB has two complete 5-by-3
    // components; AC has components 5-by-3, 1-by-3, and 4-by-1;
    // BC has components 3-by-3 and 3-by-4. All primary scores are 3.
    // Retained sums are respectively 6, 5, 6, so AB wins the index tie.
    // Total local sizes instead rank AC (17) above AB (16) and BC (13).
    let mut columns = vec![Vec::<u32>::new(); 3];
    for a in 0..5 {
        for b in 0..3 {
            for c in 0..3 {
                columns[0].push(a);
                columns[1].push(b);
                columns[2].push(c);
            }
        }
    }
    for b in 3..6 {
        for c in 3..6 {
            columns[0].push(5);
            columns[1].push(b);
            columns[2].push(c);
        }
        for a in 6..10 {
            columns[0].push(a);
            columns[1].push(b);
            columns[2].push(6);
        }
    }
    let solver = Solver::new(
        Design::from_levels_for_test(columns),
        None,
        PreconditionerConfig::Additive {
            local_solver: LocalSolverConfig {
                dense_threshold: 2,
                ..Default::default()
            },
            reduction: crate::ReductionStrategy::Ordered,
        },
    )
    .unwrap();
    let nomination = solver.pair_nomination().unwrap();
    assert_eq!(nomination.factors, (0, 1));
    assert_eq!(nomination.max_retained, 3);
    assert_eq!(nomination.total_retained, 6);
    assert_eq!(nomination.replacement_components, 2);
    let replacement = solver
        .build_pair_replacement(&ProjectedFactory)
        .unwrap()
        .unwrap();
    assert_eq!(replacement.nomination(), nomination);
}

#[test]
fn nomination_preserves_small_components_and_is_transactional() {
    // Two large connected components and one tiny component. Both factors tie
    // with the third redundant factor; original indices decide the nomination.
    let mut columns = vec![Vec::<u32>::new(); 3];
    for (offset, size) in [(0, 4), (4, 4), (8, 1)] {
        for a in 0..size {
            for b in 0..size {
                columns[0].push(offset + a);
                columns[1].push(offset + b);
                columns[2].push(offset + b);
            }
        }
    }
    let design = Design::from_levels_for_test(columns);
    let config = PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig {
            dense_threshold: 2,
            ..Default::default()
        },
        reduction: crate::ReductionStrategy::Ordered,
    };
    let solver = Solver::new(design, None, config).unwrap();
    let nomination = solver.pair_nomination().unwrap();
    assert_eq!(nomination.factors, (0, 1));
    assert_eq!(nomination.replacement_components, 2);
    let incumbent = postcard::to_stdvec(solver.preconditioner().unwrap()).unwrap();
    struct FailsSecond(AtomicUsize);
    impl FixedReducedInverseFactory for FailsSecond {
        fn build(&self, c: PairComponent<'_>) -> Result<Arc<dyn FixedReducedInverse>, BuildError> {
            if self.0.fetch_add(1, Ordering::Relaxed) == 1 {
                return Err(BuildError::LocalSolverBuild(
                    "injected factory rejection".into(),
                ));
            }
            Ok(Arc::new(ProjectedDiagonal(c.n_retained)))
        }
    }
    let factory = FailsSecond(AtomicUsize::new(0));
    assert!(solver.build_pair_replacement(&factory).is_err());
    assert_eq!(factory.0.load(Ordering::Relaxed), 2);
    assert_eq!(
        incumbent,
        postcard::to_stdvec(solver.preconditioner().unwrap()).unwrap()
    );
    let replacement = solver
        .build_pair_replacement(&ProjectedFactory)
        .unwrap()
        .unwrap();
    let Variant::Additive(base) = &solver.preconditioner().unwrap().inner else {
        panic!()
    };
    let new = &replacement.preconditioner;
    for (a, b) in base.inner.subdomains().iter().zip(new.subdomains()) {
        if matches!(b.solver(), PairLocalSolver::Incumbent { .. }) {
            assert!(std::ptr::eq(a.solver(), b.solver().inner()));
        }
        if a.solver().n_kept() <= 2 {
            assert_eq!(
                postcard::to_stdvec(a.solver()).unwrap(),
                postcard::to_stdvec(b.solver().inner()).unwrap()
            );
        }
    }
    assert!(replacement
        .preconditioner
        .subdomains()
        .iter()
        .any(|entry| postcard::to_stdvec(entry.solver().inner()).is_err()));
    let _: crate::Preconditioner = postcard::from_bytes(&incumbent).unwrap();
}

#[test]
fn default_dense_boundary_is_retained_and_same_pair_ties_are_stable() {
    let design = Design::from_levels_for_test(vec![
        (0..24).flat_map(|i| [i; 24]).collect(),
        (0..24).flat_map(|_| 0..24).collect(),
    ]);
    let solver = Solver::new(design, None, None).unwrap();
    assert!(solver.pair_nomination().is_none());
    assert!(solver
        .build_pair_replacement(&ProjectedFactory)
        .unwrap()
        .is_none());
}

#[test]
fn metadata_budget_rejection_precedes_factory_and_leaves_solver_exact() {
    struct Budget {
        units: AtomicUsize,
        calls: AtomicUsize,
    }
    impl FixedReducedInverseFactory for Budget {
        fn checkpoint(&self, work: u64) -> Result<(), BuildError> {
            let total = self.units.fetch_add(work as usize, Ordering::Relaxed) + work as usize;
            if total > 256 {
                Err(BuildError::LocalSolverBuild("metadata budget".into()))
            } else {
                Ok(())
            }
        }
        fn build(&self, c: PairComponent<'_>) -> Result<Arc<dyn FixedReducedInverse>, BuildError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Arc::new(ProjectedDiagonal(c.n_retained)))
        }
    }
    let design = Design::from_levels_for_test(vec![
        (0..4).flat_map(|i| [i; 4]).collect(),
        (0..4).flat_map(|_| 0..4).collect(),
        (0..4).flat_map(|_| 0..4).collect(),
    ]);
    let config = PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig {
            dense_threshold: 2,
            ..Default::default()
        },
        reduction: crate::ReductionStrategy::Ordered,
    };
    let solver = Solver::new(design, None, config).unwrap();
    let before = postcard::to_stdvec(solver.preconditioner().unwrap()).unwrap();
    let budget = Budget {
        units: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
    };
    assert!(solver.build_pair_replacement(&budget).is_err());
    assert_eq!(budget.calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        before,
        postcard::to_stdvec(solver.preconditioner().unwrap()).unwrap()
    );
}
