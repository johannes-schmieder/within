//! White-box checks on the bidiagonalization itself.

use super::{dot, Bidiagonalization, GolubKahan};
use crate::lsmr::fixtures::DenseOp;
use rayon::prelude::*;

/// Window smaller than the iteration count: the ring must wrap correctly.
/// We re-run the bidiagonalization manually with the same window and
/// verify the last `local_size` `v` vectors are mutually orthogonal to
/// tighter tolerance than they would be without reorthogonalization.
#[test]
fn local_reorth_keeps_the_window_vectors_orthogonal() {
    let op = DenseOp::vandermonde(30, 12);
    let b: Vec<f64> = (0..op.rows)
        .map(|i| {
            let x = i as f64 / (op.rows - 1) as f64;
            (1.0 + x).ln()
        })
        .collect();

    let local_size = 3;
    let n_iters = 10;

    // Run the bidiagonalization directly so we can capture v_k after each
    // step. Mirrors the body of `lsmr_from_bidiag` minus the recurrence.
    let collect_vs = |window_size: usize| -> Vec<Vec<f64>> {
        let (mut bidiag, _) = GolubKahan::init(&op, &b, window_size).expect("init");
        let mut vs = vec![bidiag.v().to_vec()];
        for _ in 0..n_iters {
            bidiag.step().expect("step");
            vs.push(bidiag.v().to_vec());
        }
        vs
    };

    let vs_no_reorth = collect_vs(0);
    let vs_windowed = collect_vs(local_size);

    // Compare the maximum |⟨v_i, v_j⟩| over the last `local_size` vectors.
    let max_off_diag = |vs: &[Vec<f64>]| -> f64 {
        let n = vs.len();
        let start = n.saturating_sub(local_size);
        let mut worst: f64 = 0.0;
        for i in start..n {
            for j in (i + 1)..n {
                worst = worst.max(dot(&vs[i], &vs[j]).abs());
            }
        }
        worst
    };

    let drift_no = max_off_diag(&vs_no_reorth);
    let drift_yes = max_off_diag(&vs_windowed);
    assert!(
        drift_yes < drift_no,
        "windowed drift ({drift_yes:e}) should be smaller than \
         unwindowed drift ({drift_no:e})"
    );
    assert!(
        drift_yes < 1e-10,
        "last {local_size} v's not mutually orthogonal: {drift_yes:e}"
    );
}

fn reduction_vectors(n: usize) -> (Vec<f64>, Vec<f64>) {
    // A large first partial and small later partials expose reassociation.
    let x: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 {
                1e8
            } else if i % 4096 == 0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    (x.clone(), x)
}

#[test]
fn parallel_fused_norm_repeats_exactly() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(7)
        .build()
        .unwrap();
    for n in [9_999, 10_000, 10_001, 65_537, 100_000] {
        let (x, y) = reduction_vectors(n);
        let expected: Vec<f64> = x.iter().zip(&y).map(|(&x, &y)| x - 0.73 * y).collect();
        let oracle: f64 = expected.iter().map(|v| v * v).sum();
        let run = || {
            let mut work = y.clone();
            let norm = pool.install(|| super::axpy_with_sq_norm(&mut work, &x, -0.73));
            assert_eq!(work, expected);
            assert!((norm - oracle).abs() <= 1e-12 * oracle.max(1.0));
            norm.to_bits()
        };
        let first = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                let mut work = y.clone();
                super::axpy_with_sq_norm(&mut work, &x, -0.73).to_bits()
            });
        let repeated: Vec<_> = pool.install(|| (0..32).into_par_iter().map(|_| run()).collect());
        for actual in repeated {
            assert_eq!(actual, first, "norm changed for {n} rows");
        }
    }
}

#[test]
fn parallel_dot_repeats_exactly() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(7)
        .build()
        .unwrap();
    for n in [9_999, 10_000, 10_001, 65_537, 100_000] {
        let (x, y) = reduction_vectors(n);
        let oracle = dot(&x, &y);
        let run = || {
            let actual = pool.install(|| super::par_dot(&x, &y));
            assert!((actual - oracle).abs() <= 1e-12 * oracle.abs().max(1.0));
            actual.to_bits()
        };
        let first = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| super::par_dot(&x, &y))
            .to_bits();
        let repeated: Vec<_> = pool.install(|| (0..32).into_par_iter().map(|_| run()).collect());
        for actual in repeated {
            assert_eq!(actual, first, "dot changed for {n} rows");
        }
    }
}

/// Chunk partials from `lsmr_norm_chunk` combined by `lsmr_norm_reduce` are
/// bitwise the separate update's norm at every worker count.
#[test]
fn published_norm_partials_reproduce_the_separate_update_norm() {
    for n in [
        1,
        17,
        4_096,
        9_999,
        10_000,
        10_001,
        3 * 4_096,
        65_537,
        100_003,
    ] {
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 7_919) % 1_009) as f64 / 37.0 - 13.0 + if i == 0 { 1e7 } else { 0.0 })
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| ((i * 31) % 97) as f64 / 11.0 - 4.0)
            .collect();
        let mut updated = y.clone();
        for (u, &a) in updated.iter_mut().zip(&x) {
            *u = a + -0.37 * *u;
        }
        let chunk = super::lsmr_norm_chunk(n);
        let partials: Vec<f64> = updated
            .chunks(chunk)
            .map(|c| {
                let mut s = 0.0;
                for v in c {
                    s += v * v;
                }
                s
            })
            .collect();
        let published = super::lsmr_norm_reduce(&partials).to_bits();
        for threads in [1, 2, 3, 8] {
            let separate = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut work = y.clone();
                    let norm = super::axpy_with_sq_norm(&mut work, &x, -0.37);
                    assert_eq!(work, updated);
                    norm.to_bits()
                });
            assert_eq!(separate, published, "n={n} threads={threads}");
        }
    }
}

/// An operator whose fused hook reproduces the separate passes exactly leaves
/// modified LSMR bitwise unchanged, including certification and breakdown.
#[test]
fn exact_fused_hook_leaves_modified_lsmr_bitwise_unchanged() {
    use crate::lsmr::fixtures::{jacobi, DiagOp};
    use crate::{mlsmr, MlsmrOptions, Operator, SolveError};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Fused<'a> {
        inner: &'a DenseOp,
        calls: AtomicUsize,
    }
    impl Operator for Fused<'_> {
        fn nrows(&self) -> usize {
            self.inner.nrows()
        }
        fn ncols(&self) -> usize {
            self.inner.ncols()
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            self.inner.apply(x, y)
        }
        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            self.inner.apply_adjoint(x, y)
        }
        fn apply_fused_update(
            &self,
            x: &[f64],
            c: f64,
            u: &mut [f64],
            y: &mut [f64],
        ) -> Option<Result<f64, SolveError>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut ax = vec![0.0; u.len()];
            if let Err(error) = self.inner.apply(x, &mut ax) {
                return Some(Err(error));
            }
            let chunk = super::lsmr_norm_chunk(u.len());
            let mut partials = Vec::new();
            for (u, ax) in u.chunks_mut(chunk).zip(ax.chunks(chunk)) {
                let mut s = 0.0;
                for (ui, &ai) in u.iter_mut().zip(ax) {
                    *ui = ai + c * *ui;
                    s += *ui * *ui;
                }
                partials.push(s);
            }
            Some(
                self.inner
                    .apply_adjoint(u, y)
                    .map(|()| super::lsmr_norm_reduce(&partials)),
            )
        }
    }

    let op = DenseOp::vandermonde(30, 12);
    let b: Vec<f64> = (0..op.rows).map(|i| ((i * 5) % 11) as f64 - 4.5).collect();
    let fused = Fused {
        inner: &op,
        calls: AtomicUsize::new(0),
    };
    let unit = DiagOp(vec![1.0; op.cols]);
    for local_size in [None, Some(4)] {
        let options = MlsmrOptions {
            local_size,
            ..Default::default()
        };
        for m in [&jacobi(&op), &unit] {
            let expected = mlsmr(&op, &b, m, 1e-10, 200, options).unwrap();
            let actual = mlsmr(&fused, &b, m, 1e-10, 200, options).unwrap();
            assert_eq!(expected.iterations, actual.iterations);
            assert_eq!(expected.stop_reason, actual.stop_reason);
            assert_eq!(expected.x, actual.x);
            assert_eq!(
                expected.residual_norm.to_bits(),
                actual.residual_norm.to_bits()
            );
            assert_eq!(
                expected.normal_eq_residual.to_bits(),
                actual.normal_eq_residual.to_bits()
            );
        }
    }
    assert!(fused.calls.load(Ordering::Relaxed) > 0);
}
