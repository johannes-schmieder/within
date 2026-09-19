//! Independent dense action, ordered cancellation, error publication and binding contracts.
use schwarz_precond::{
    LocalSolveError, LocalSolver, OperatorMut, PartitionWeights, ReductionStrategy,
    SchwarzPreconditioner, SolveError, SubdomainCore, SubdomainEntry,
};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
struct Dense {
    n: usize,
    scratch: usize,
    matrix: Vec<f64>,
    fail: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
}
impl LocalSolver for Dense {
    fn n_local(&self) -> usize {
        self.n
    }
    fn scratch_size(&self) -> usize {
        self.scratch
    }
    fn solve_local(
        &self,
        r: &mut [f64],
        z: &mut [f64],
        parallel: bool,
    ) -> Result<(), LocalSolveError> {
        assert!(!parallel);
        assert_eq!(r.len(), self.scratch);
        assert_eq!(z.len(), self.scratch);
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.fail.load(Ordering::Relaxed) {
            r.fill(f64::NAN);
            z.fill(f64::NAN);
            return Err(LocalSolveError::BackendFailed {
                context: "injected",
                message: String::new(),
            });
        }
        for (i, out) in z[..self.n].iter_mut().enumerate() {
            *out = self.matrix[i * self.n..(i + 1) * self.n]
                .iter()
                .zip(&r[..self.n])
                .map(|(a, b)| a * b)
                .sum();
        }
        // Poison augmentation: another domain must not accidentally read it.
        r[self.n..].fill(f64::NAN);
        z[self.n..].fill(f64::NAN);
        Ok(())
    }
}
fn entry(
    indices: Vec<u32>,
    weights: Vec<f64>,
    matrix: Vec<f64>,
    scratch: usize,
    fail: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
) -> SubdomainEntry<Dense> {
    let n = indices.len();
    let core =
        SubdomainCore::with_partition_weights(indices, PartitionWeights::NonUniform(weights))
            .unwrap();
    SubdomainEntry::try_new(
        core,
        Dense {
            n,
            scratch,
            matrix,
            fail,
            calls,
        },
    )
    .unwrap()
}
#[test]
fn dense_reference_transactional_failure_shapes_and_reuse() {
    let fail = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let entries = vec![
        entry(
            vec![0, 2],
            vec![1., 0.5],
            vec![2., 1., 1., 3.],
            4,
            Arc::new(AtomicBool::new(false)),
            calls.clone(),
        ),
        entry(
            vec![1, 2, 4],
            vec![0.5, 1., 2.],
            vec![2., 0., 0.5, 0., 4., -1., 0.5, -1., 3.],
            5,
            fail.clone(),
            calls.clone(),
        ),
    ];
    let owner = SchwarzPreconditioner::with_n_dofs(entries, 6, ReductionStrategy::Auto);
    assert_eq!(
        owner.serial_workspace_required_payload_bytes().unwrap(),
        128
    );
    assert!(matches!(
        owner.try_serial_workspace(127),
        Err(SolveError::SerialWorkspaceBudget {
            required: 128,
            limit: 127
        })
    ));
    let mut w = owner.try_serial_workspace(128).unwrap();
    assert_eq!(w.retained_payload_bytes(), 128);
    let mut out = [-99.; 6];
    for rhs in [
        [1., 2., 3., 4., 5., 6.],
        [-2., 0.5, 4., 3., -1., 0.],
        [0.; 6],
    ] {
        // Hand-assembled global matrix includes two-sided weights and overlap.
        let expected = [
            2. * rhs[0] + 0.5 * rhs[2],
            0.5 * rhs[1] + 0.5 * rhs[4],
            0.5 * rhs[0] + 4.75 * rhs[2] - 2. * rhs[4],
            0.,
            0.5 * rhs[1] - 2. * rhs[2] + 12. * rhs[4],
            0.,
        ];
        w.apply(&rhs, &mut out).unwrap();
        assert_eq!(out, expected);
        OperatorMut::apply_adjoint(&mut w, &rhs, &mut out).unwrap();
        assert_eq!(out, expected);
        let before = calls.load(Ordering::Relaxed);
        out.fill(7.);
        assert!(matches!(
            w.apply(&rhs[..5], &mut out),
            Err(SolveError::SerialWorkspaceShape { .. })
        ));
        assert!(matches!(
            w.apply(&rhs, &mut out[..5]),
            Err(SolveError::SerialWorkspaceShape { .. })
        ));
        assert_eq!(out, [7.; 6]);
        assert_eq!(calls.load(Ordering::Relaxed), before);
        fail.store(true, Ordering::Relaxed);
        assert!(matches!(
            w.apply(&rhs, &mut out),
            Err(SolveError::LocalSolveFailed { subdomain: 1, .. })
        ));
        assert_eq!(out, [7.; 6]);
        fail.store(false, Ordering::Relaxed);
        w.apply(&rhs, &mut out).unwrap();
        assert_eq!(out, expected);
    }
}
#[test]
fn stored_domain_order_is_observable_and_empty_domains_are_safe() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut entries = Vec::new();
    for a in [1e16, -1e16, 1.] {
        entries.push(entry(
            vec![0],
            vec![1.],
            vec![a],
            1,
            Arc::new(AtomicBool::new(false)),
            calls.clone(),
        ));
    }
    entries.push(entry(
        vec![],
        vec![],
        vec![],
        0,
        Arc::new(AtomicBool::new(true)),
        calls.clone(),
    ));
    let owner =
        SchwarzPreconditioner::with_n_dofs(entries, 2, ReductionStrategy::ParallelReduction);
    let mut w = owner.try_serial_workspace(32).unwrap();
    let mut out = [8.; 2];
    for _ in 0..32 {
        w.apply(&[1., 3.], &mut out).unwrap();
        assert_eq!(out, [1., 0.]);
    }
    assert_eq!(calls.load(Ordering::Relaxed), 96);
    let owner = SchwarzPreconditioner::<Dense>::with_n_dofs(vec![], 0, ReductionStrategy::Auto);
    let mut w = owner.try_serial_workspace(0).unwrap();
    w.apply(&[], &mut []).unwrap();
    assert_eq!(w.retained_payload_bytes(), 0);
}

#[test]
fn changed_local_dimensions_reject_before_output_and_recover() {
    struct Changing {
        n: Arc<AtomicUsize>,
        scratch: Arc<AtomicUsize>,
    }
    impl LocalSolver for Changing {
        fn n_local(&self) -> usize {
            self.n.load(Ordering::Relaxed)
        }
        fn scratch_size(&self) -> usize {
            self.scratch.load(Ordering::Relaxed)
        }
        fn solve_local(
            &self,
            r: &mut [f64],
            z: &mut [f64],
            _: bool,
        ) -> Result<(), LocalSolveError> {
            z.copy_from_slice(r);
            Ok(())
        }
    }
    let n = Arc::new(AtomicUsize::new(2));
    let scratch = Arc::new(AtomicUsize::new(2));
    let entry = SubdomainEntry::try_new(
        SubdomainCore::uniform(vec![0, 1]),
        Changing {
            n: n.clone(),
            scratch: scratch.clone(),
        },
    )
    .unwrap();
    let owner = SchwarzPreconditioner::new(vec![entry], ReductionStrategy::Auto);
    let mut w = owner.try_serial_workspace(48).unwrap();
    let mut out = [7.; 2];
    scratch.store(3, Ordering::Relaxed);
    assert!(matches!(
        w.apply(&[1., 2.], &mut out),
        Err(SolveError::SerialWorkspaceChanged)
    ));
    assert_eq!(out, [7.; 2]);
    scratch.store(2, Ordering::Relaxed);
    n.store(1, Ordering::Relaxed);
    assert!(matches!(
        w.apply(&[1., 2.], &mut out),
        Err(SolveError::SerialWorkspaceInvalidDomain { subdomain: 0 })
    ));
    assert_eq!(out, [7.; 2]);
    assert!(matches!(
        owner.try_serial_workspace(0),
        Err(SolveError::SerialWorkspaceInvalidDomain { subdomain: 0 })
    ));
    n.store(2, Ordering::Relaxed);
    w.apply(&[1., 2.], &mut out).unwrap();
    assert_eq!(out, [1., 2.]);
}
