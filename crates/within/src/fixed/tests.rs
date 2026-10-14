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

#[test]
fn owned_plan_survives_borrowed_inputs_and_preserves_checkpoints_and_application() {
    use std::cell::RefCell;
    #[derive(Debug, PartialEq)]
    enum Boundary {
        Work(u64),
        Allocation(&'static str, usize),
    }
    struct Recording(RefCell<Vec<Boundary>>);
    impl FixedReducedInverseFactory for Recording {
        fn checkpoint(&self, work: u64) -> Result<(), BuildError> {
            self.0.borrow_mut().push(Boundary::Work(work));
            Ok(())
        }
        fn allocation(&self, site: &'static str, bytes: usize) -> Result<(), BuildError> {
            self.0.borrow_mut().push(Boundary::Allocation(site, bytes));
            Ok(())
        }
        fn build(&self, c: PairComponent<'_>) -> Result<Arc<dyn FixedReducedInverse>, BuildError> {
            ProjectedFactory.build(c)
        }
    }
    let build = |separate: bool| {
        let recorder = Recording(RefCell::new(Vec::new()));
        let a: Vec<u32> = (0..4).flat_map(|i| [i; 4]).collect();
        let b: Vec<u32> = (0..4).flat_map(|_| 0..4).collect();
        let solver = Solver::new(
            vec![
                Effect::new(&a, true, []).unwrap(),
                Effect::new(&b, true, []).unwrap(),
            ],
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
        let replacement = if separate {
            let plan = solver.prepare_pair_replacement(&recorder).unwrap().unwrap();
            assert_eq!(plan.nomination(), solver.pair_nomination().unwrap());
            drop(solver);
            drop(a);
            drop(b);
            // The owned plan can cross a worker boundary after all input borrows end.
            let before = recorder.0.into_inner();
            let (replacement, after) = std::thread::spawn(move || {
                let recorder = Recording(RefCell::new(Vec::new()));
                (plan.build(&recorder).unwrap(), recorder.0.into_inner())
            })
            .join()
            .unwrap();
            let mut all = before;
            all.extend(after);
            return (replacement, all);
        } else {
            solver.build_pair_replacement(&recorder).unwrap().unwrap()
        };
        (replacement, recorder.0.into_inner())
    };
    let (sync, sync_work) = build(false);
    let (owned, owned_work) = build(true);
    assert_eq!(sync_work, owned_work);
    let rhs: Vec<f64> = (0..8).map(|i| i as f64 / 7.0).collect();
    let mut a = vec![0.0; 8];
    let mut b = vec![0.0; 8];
    sync.apply(&rhs, &mut a).unwrap();
    owned.apply(&rhs, &mut b).unwrap();
    assert_eq!(a, b);
    assert_eq!(sync.apply_work_bound(), owned.apply_work_bound());
    assert_eq!(sync.apply_storage_bound(), owned.apply_storage_bound());
}

#[test]
fn every_owned_allocation_gate_rejects_transactionally_with_checked_sizes() {
    use std::cell::RefCell;
    #[derive(Default)]
    struct Allocations {
        reject: Option<usize>,
        seen: RefCell<Vec<(&'static str, usize)>>,
    }
    impl FixedReducedInverseFactory for Allocations {
        fn allocation(&self, site: &'static str, bytes: usize) -> Result<(), BuildError> {
            let mut seen = self.seen.borrow_mut();
            let index = seen.len();
            seen.push((site, bytes));
            if self.reject == Some(index) {
                Err(BuildError::Allocation { context: site })
            } else {
                Ok(())
            }
        }
        fn build(&self, c: PairComponent<'_>) -> Result<Arc<dyn FixedReducedInverse>, BuildError> {
            ProjectedFactory.build(c)
        }
    }
    let solver = || {
        let design = || {
            Design::from_levels_for_test(vec![
                (0..4).flat_map(|i| [i; 4]).collect(),
                (0..4).flat_map(|_| 0..4).collect(),
            ])
        };
        let original = Solver::new(
            design(),
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
        // A supported prebuilt action exercises the nonuniform partition copy.
        let mut preconditioner = original.preconditioner().unwrap().clone();
        let Variant::Additive(base) = &mut preconditioner.inner else {
            panic!()
        };
        let entries = base
            .inner
            .subdomains()
            .iter()
            .map(|entry| {
                let weights = (0..entry.global_indices().len())
                    .map(|i| 0.5 + i as f64 / 8.0)
                    .collect();
                let core = SubdomainCore::with_partition_weights(
                    entry.global_indices().to_vec(),
                    PartitionWeights::NonUniform(weights),
                )
                .unwrap();
                SubdomainEntry::try_new(core, entry.solver().clone()).unwrap()
            })
            .collect();
        base.inner =
            SchwarzPreconditioner::with_n_dofs(entries, 8, crate::ReductionStrategy::Ordered);
        Solver::new(design(), None, preconditioner).unwrap()
    };
    let accepted = Allocations::default();
    let reference_solver = solver();
    let replacement = reference_solver
        .build_pair_replacement(&accepted)
        .unwrap()
        .unwrap();
    let boundaries = accepted.seen.into_inner();
    assert_eq!(
        boundaries.iter().map(|(site, _)| *site).collect::<Vec<_>>(),
        vec![
            "pair nomination metadata",
            "pair nomination components",
            "pair plan owner",
            "pair source identity",
            "pair replacement entries",
            "pair eliminated diagonal",
            "pair global indices",
            "pair partition weights",
            "schwarz entries owner",
            "schwarz buffer pool",
            "pair replacement identity",
        ]
    );
    assert!(boundaries.iter().all(|(_, bytes)| *bytes > 0));
    assert_eq!(boundaries[5].1, 4 * std::mem::size_of::<f64>());
    assert_eq!(boundaries[6].1, 8 * std::mem::size_of::<u32>());
    assert_eq!(boundaries[7].1, 8 * std::mem::size_of::<f64>());
    let rhs = [0.3, -1.0, 1.2, 0.5, -0.7, 0.1, 0.2, 0.8];
    let mut expected = [0.0; 8];
    replacement.apply(&rhs, &mut expected).unwrap();
    for (reject, &(site, _)) in boundaries.iter().enumerate() {
        let solver = solver();
        let incumbent = postcard::to_stdvec(solver.preconditioner().unwrap()).unwrap();
        let factory = Allocations {
            reject: Some(reject),
            ..Default::default()
        };
        let error = solver.build_pair_replacement(&factory).err().unwrap();
        assert!(matches!(error, BuildError::Allocation { context } if context == site));
        assert_eq!(*factory.seen.borrow(), boundaries[..=reject]);
        assert_eq!(
            incumbent,
            postcard::to_stdvec(solver.preconditioner().unwrap()).unwrap()
        );
        if let Some(source) = solver.replacement_source.get() {
            assert_eq!(
                Arc::strong_count(source),
                1,
                "failed candidate must release its source"
            );
        }
        let retry = solver
            .build_pair_replacement(&ProjectedFactory)
            .unwrap()
            .unwrap();
        let mut actual = [0.0; 8];
        retry.apply(&rhs, &mut actual).unwrap();
        assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
    }
    let factory = Allocations::default();
    let error = reserve_admitted(
        &mut Vec::<f64>::new(),
        usize::MAX,
        Some(&factory),
        "overflow",
    )
    .unwrap_err();
    assert!(matches!(
        error,
        BuildError::Allocation {
            context: "overflow"
        }
    ));
    assert!(
        factory.seen.borrow().is_empty(),
        "overflow must fail before callback or allocation"
    );
}
