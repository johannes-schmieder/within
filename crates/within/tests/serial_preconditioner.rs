//! Real dense/approximate pair kernels, disconnected labels, fixed scheduling and public ownership.
mod serial_support;
use schwarz_precond::OperatorMut;
use within::{Effect, PreconditionerConfig, Solver};
#[test]
fn real_pair_actions_match_one_worker_pooled_reference_and_repeat_across_pools() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let four = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for case in ["latin", "nested", "disconnected-interleaved", "unbalanced"] {
        for approximate in [false, true] {
            for weighted in [false, true] {
                let owner = serial_support::build(case, approximate, weighted);
                let n = owner.nrows();
                let bytes = owner.serial_workspace_required_payload_bytes().unwrap();
                let mut serial = owner.try_serial_workspace(bytes).unwrap();
                let mut out = vec![0.; n];
                let mut pooled = vec![0.; n];
                for seed in 0..8 {
                    let rhs: Vec<_> = (0..n)
                        .map(|i| ((i * 7 + seed * 13) % 23) as f64 / 8. - 1.)
                        .collect();
                    serial.apply(&rhs, &mut out).unwrap();
                    one.install(|| owner.apply(&rhs, &mut pooled)).unwrap();
                    for (&a, &b) in out.iter().zip(&pooled) {
                        assert!(a.is_finite() && b.is_finite());
                        assert!(
                            (a - b).abs() <= 1e-12 * (1. + b.abs()),
                            "{case} {approximate} {weighted}: {a} {b}"
                        );
                    }
                    let first = out.clone();
                    four.install(|| serial.apply(&rhs, &mut out)).unwrap();
                    assert_eq!(
                        out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        first.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                    OperatorMut::apply_adjoint(&mut serial, &rhs, &mut out).unwrap();
                    assert_eq!(out, first);
                    // Independent concurrent workspaces share only the immutable owner.
                    std::thread::scope(|scope| {
                        let a = scope.spawn(|| {
                            let mut w = owner.try_serial_workspace(bytes).unwrap();
                            let mut y = vec![0.; n];
                            w.apply(&rhs, &mut y).unwrap();
                            y
                        });
                        let b = scope.spawn(|| {
                            let mut w = owner.try_serial_workspace(bytes).unwrap();
                            let mut y = vec![0.; n];
                            w.apply(&rhs, &mut y).unwrap();
                            y
                        });
                        assert_eq!(a.join().unwrap(), first);
                        assert_eq!(b.join().unwrap(), first);
                    });
                }
            }
        }
    }
}
#[test]
fn diagonal_action_uses_no_workspace_array_and_validates_before_output() {
    let codes = [0, 1, 0, 1];
    let effects = vec![Effect::new(&codes, true, std::iter::empty::<&[f64]>()).unwrap()];
    let solver = Solver::new(
        effects,
        Some(vec![1., 2., 3., 4.]),
        PreconditionerConfig::Diagonal,
    )
    .unwrap();
    let owner = solver.preconditioner().unwrap();
    assert_eq!(owner.serial_workspace_required_payload_bytes().unwrap(), 0);
    let mut w = owner.try_serial_workspace(0).unwrap();
    assert_eq!(w.retained_payload_bytes(), 0);
    let mut out = [7.; 2];
    assert!(w.apply(&[1.], &mut out).is_err());
    assert_eq!(out, [7.; 2]);
    w.apply(&[8., 18.], &mut out).unwrap();
    assert_eq!(out, [2., 3.]);
}
