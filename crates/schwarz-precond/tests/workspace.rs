//! Compare against the frozen upstream executable code, not a second copied recurrence.
use schwarz_precond::{
    mlsmr, mlsmr_with_workspace, MlsmrOptions, MlsmrWorkspace, MlsmrWorkspaceOptions, OperatorMut,
    SolveError,
};
use schwarz_precond_baseline as baseline;

struct Dense {
    m: usize,
    n: usize,
    data: Vec<f64>,
}
impl Dense {
    fn new(m: usize, n: usize, rank_deficient: bool) -> Self {
        let data = (0..m * n)
            .map(|k| {
                let row = k / n;
                let mut col = k % n;
                if rank_deficient && col == n - 1 {
                    col = 0;
                }
                ((row * 13 + col * 7 + 3) as f64 * 0.17).sin() + if row == col { 2.0 } else { 0.0 }
            })
            .collect();
        Self { m, n, data }
    }
    fn apply_raw(&self, x: &[f64], y: &mut [f64]) {
        for (row, out) in self.data.chunks(self.n).zip(y) {
            *out = row.iter().zip(x).map(|(a, b)| a * b).sum();
        }
    }
    fn adjoint_raw(&self, x: &[f64], y: &mut [f64]) {
        for (col, out) in y.iter_mut().enumerate() {
            *out = (0..self.m)
                .map(|row| self.data[row * self.n + col] * x[row])
                .sum();
        }
    }
}
macro_rules! dense_operator {
    ($backend:ident) => {
        impl $backend::Operator for Dense {
            fn nrows(&self) -> usize {
                self.m
            }
            fn ncols(&self) -> usize {
                self.n
            }
            fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), $backend::SolveError> {
                self.apply_raw(x, y);
                Ok(())
            }
            fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), $backend::SolveError> {
                self.adjoint_raw(x, y);
                Ok(())
            }
        }
    };
}
dense_operator!(schwarz_precond);
dense_operator!(baseline);
struct Identity(usize);
macro_rules! identity_operator {
    ($backend:ident) => {
        impl $backend::Operator for Identity {
            fn nrows(&self) -> usize {
                self.0
            }
            fn ncols(&self) -> usize {
                self.0
            }
            fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), $backend::SolveError> {
                y.copy_from_slice(x);
                Ok(())
            }
            fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), $backend::SolveError> {
                y.copy_from_slice(x);
                Ok(())
            }
        }
    };
}
identity_operator!(schwarz_precond);
identity_operator!(baseline);
fn bits(a: &[f64], b: &[f64]) {
    assert_eq!(
        a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn serial_reuse_and_allocating_wrapper_match_frozen_upstream_bits() {
    for (m, n, rank_deficient) in [(9, 5, false), (5, 9, false), (8, 6, true)] {
        let op = Dense::new(m, n, rank_deficient);
        let pre = Identity(n);
        for local in [None, Some(0), Some(1), Some(3), Some(8), Some(usize::MAX)] {
            let mut w = MlsmrWorkspace::try_new(m, n, local).unwrap();
            assert_eq!(
                w.retained_payload_bytes().unwrap(),
                MlsmrWorkspace::required_payload_bytes(m, n, local).unwrap()
            );
            for repeat in 0..4 {
                let b: Vec<_> = (0..m).map(|i| ((i * 3 + repeat) as f64).cos()).collect();
                for maxiter in [0, 1, 200] {
                    for warm in [false, true] {
                        let x0 = vec![0.125; n];
                        let warm_start = warm.then_some(&x0[..]);
                        let frozen = baseline::mlsmr(
                            &op,
                            &b,
                            &pre,
                            1e-10,
                            maxiter,
                            baseline::MlsmrOptions {
                                warm_start,
                                escalation: None,
                                local_size: local,
                            },
                        )
                        .unwrap();
                        let owned = mlsmr(
                            &op,
                            &b,
                            &pre,
                            1e-10,
                            maxiter,
                            MlsmrOptions {
                                warm_start,
                                escalation: None,
                                local_size: local,
                            },
                        )
                        .unwrap();
                        let prepared = mlsmr_with_workspace(
                            &mut &op,
                            &b,
                            &mut &pre,
                            1e-10,
                            maxiter,
                            MlsmrWorkspaceOptions {
                                warm_start,
                                escalation: None,
                            },
                            &mut w,
                        )
                        .unwrap();
                        bits(&frozen.x, &owned.x);
                        bits(&frozen.x, prepared.x);
                        assert_eq!(frozen.converged, prepared.diagnostics.converged);
                        assert_eq!(frozen.iterations, prepared.diagnostics.iterations);
                        assert_eq!(
                            frozen.residual_norm.to_bits(),
                            prepared.diagnostics.residual_norm.to_bits()
                        );
                        assert_eq!(
                            frozen.normal_eq_residual.to_bits(),
                            prepared.diagnostics.normal_eq_residual.to_bits()
                        );
                        assert_eq!(
                            format!("{:?}", frozen.stop_reason),
                            format!("{:?}", prepared.diagnostics.stop_reason)
                        );
                    }
                }
            }
        }
    }
}

struct MutableAction {
    n: usize,
    scale: f64,
    fail: bool,
    calls: usize,
}
impl OperatorMut for MutableAction {
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn apply(&mut self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.calls += 1;
        if self.fail {
            return Err(SolveError::Synchronization {
                context: "injected action failure",
            });
        }
        for (out, &v) in y.iter_mut().zip(x) {
            *out = v * self.scale;
        }
        Ok(())
    }
    fn apply_adjoint(&mut self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}
#[test]
fn mutable_actions_change_between_calls_and_recover_after_failure() {
    let mut op = MutableAction {
        n: 3,
        scale: 2.0,
        fail: false,
        calls: 0,
    };
    let pre = Identity(3);
    let b = [2.0, 4.0, 6.0];
    let mut w = MlsmrWorkspace::try_new(3, 3, Some(8)).unwrap();
    for scale in [2.0, 4.0, 1.0] {
        op.scale = scale;
        let result = mlsmr_with_workspace(
            &mut op,
            &b,
            &mut &pre,
            1e-10,
            30,
            MlsmrWorkspaceOptions::default(),
            &mut w,
        )
        .unwrap();
        for (&x, &target) in result.x.iter().zip(&b) {
            assert!((scale * x - target).abs() < 1e-10);
        }
        op.fail = true;
        assert!(mlsmr_with_workspace(
            &mut op,
            &b,
            &mut &pre,
            1e-10,
            30,
            MlsmrWorkspaceOptions::default(),
            &mut w
        )
        .is_err());
        op.fail = false;
    }
    let before = op.calls;
    let mut wrong = MlsmrWorkspace::try_new(4, 3, Some(8)).unwrap();
    assert!(matches!(
        mlsmr_with_workspace(
            &mut op,
            &b,
            &mut &pre,
            1e-10,
            30,
            MlsmrWorkspaceOptions::default(),
            &mut wrong
        ),
        Err(SolveError::WorkspaceMismatch)
    ));
    assert_eq!(before, op.calls);
    let zero = mlsmr_with_workspace(
        &mut op,
        &[0.0; 3],
        &mut &pre,
        1e-10,
        30,
        MlsmrWorkspaceOptions::default(),
        &mut w,
    )
    .unwrap();
    assert!(zero.diagnostics.converged);
    bits(zero.x, &[0.0; 3]);
    let exact = mlsmr_with_workspace(
        &mut op,
        &b,
        &mut &pre,
        1e-10,
        30,
        MlsmrWorkspaceOptions {
            warm_start: Some(&b),
            escalation: None,
        },
        &mut w,
    )
    .unwrap();
    assert_eq!(
        exact.diagnostics.stop_reason,
        schwarz_precond::LsmrStopReason::WarmStartExact
    );
    bits(exact.x, &b);
}

#[test]
fn huge_workspace_sizes_reject_before_allocation() {
    for (m, n, k) in [
        (usize::MAX, 1, None),
        (1, usize::MAX, None),
        (usize::MAX / 8, usize::MAX / 8, Some(8)),
    ] {
        assert!(matches!(
            MlsmrWorkspace::required_payload_bytes(m, n, k),
            Err(SolveError::WorkspaceSizeOverflow)
        ));
        assert!(matches!(
            MlsmrWorkspace::try_new(m, n, k),
            Err(SolveError::WorkspaceSizeOverflow)
        ));
    }
}

struct StopAfterOne;
impl schwarz_precond::EscalationHandler for StopAfterOne {
    fn should_escalate(&mut self, progress: schwarz_precond::Progress) -> bool {
        progress.iteration >= 1
    }
}
impl baseline::EscalationHandler for StopAfterOne {
    fn should_escalate(&mut self, progress: baseline::Progress) -> bool {
        progress.iteration >= 1
    }
}
impl baseline::EscalationPolicy for StopAfterOne {
    fn handler(&self) -> Box<dyn baseline::EscalationHandler> {
        Box::new(StopAfterOne)
    }
}
#[test]
fn caller_owned_escalation_matches_upstream_handoff() {
    let op = Dense::new(9, 5, false);
    let pre = Identity(5);
    let b: Vec<_> = (0..9).map(|i| (i as f64 * 0.31).cos()).collect();
    let policy = StopAfterOne;
    let frozen = baseline::mlsmr(
        &op,
        &b,
        &pre,
        1e-14,
        100,
        baseline::MlsmrOptions {
            warm_start: None,
            escalation: Some(&policy),
            local_size: Some(3),
        },
    )
    .unwrap();
    let mut handler = StopAfterOne;
    let mut w = MlsmrWorkspace::try_new(9, 5, Some(3)).unwrap();
    let actual = mlsmr_with_workspace(
        &mut &op,
        &b,
        &mut &pre,
        1e-14,
        100,
        MlsmrWorkspaceOptions {
            warm_start: None,
            escalation: Some(&mut handler),
        },
        &mut w,
    )
    .unwrap();
    assert_eq!(
        actual.diagnostics.stop_reason,
        schwarz_precond::LsmrStopReason::Escalated
    );
    assert_eq!(actual.diagnostics.iterations, frozen.iterations);
    bits(actual.x, &frozen.x);
    assert_eq!(
        actual.diagnostics.normal_eq_residual.to_bits(),
        frozen.normal_eq_residual.to_bits()
    );
}

#[test]
fn invalid_inputs_do_not_call_mutable_actions_and_do_not_poison_reuse() {
    let mut op = MutableAction {
        n: 3,
        scale: 1.0,
        fail: false,
        calls: 0,
    };
    let pre = Identity(3);
    let bad_pre = Identity(2);
    let b = [1.0, 2.0, 3.0];
    let mut w = MlsmrWorkspace::try_new(3, 3, Some(3)).unwrap();
    for tol in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(mlsmr_with_workspace(
            &mut op,
            &b,
            &mut &pre,
            tol,
            30,
            MlsmrWorkspaceOptions::default(),
            &mut w
        )
        .is_err());
    }
    assert!(mlsmr_with_workspace(
        &mut op,
        &b[..2],
        &mut &pre,
        1e-10,
        30,
        MlsmrWorkspaceOptions::default(),
        &mut w
    )
    .is_err());
    assert!(mlsmr_with_workspace(
        &mut op,
        &b,
        &mut &bad_pre,
        1e-10,
        30,
        MlsmrWorkspaceOptions::default(),
        &mut w
    )
    .is_err());
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(mlsmr_with_workspace(
            &mut op,
            &[bad; 3],
            &mut &pre,
            1e-10,
            30,
            MlsmrWorkspaceOptions::default(),
            &mut w
        )
        .is_err());
        assert!(mlsmr_with_workspace(
            &mut op,
            &b,
            &mut &pre,
            1e-10,
            30,
            MlsmrWorkspaceOptions {
                warm_start: Some(&[bad; 3]),
                escalation: None
            },
            &mut w
        )
        .is_err());
    }
    assert!(mlsmr_with_workspace(
        &mut op,
        &b,
        &mut &pre,
        1e-10,
        30,
        MlsmrWorkspaceOptions {
            warm_start: Some(&[0.0; 2]),
            escalation: None
        },
        &mut w
    )
    .is_err());
    assert_eq!(op.calls, 0);
    // A computed warm residual has its own guard; recovery must discard it.
    op.scale = f64::INFINITY;
    assert!(mlsmr_with_workspace(
        &mut op,
        &b,
        &mut &pre,
        1e-10,
        30,
        MlsmrWorkspaceOptions {
            warm_start: Some(&[0.0; 3]),
            escalation: None
        },
        &mut w
    )
    .is_err());
    op.scale = 1.0;
    let result = mlsmr_with_workspace(
        &mut op,
        &b,
        &mut &pre,
        1e-10,
        30,
        MlsmrWorkspaceOptions::default(),
        &mut w,
    )
    .unwrap();
    for (&x, &target) in result.x.iter().zip(&b) {
        assert!((x - target).abs() < 1e-10);
    }
}
