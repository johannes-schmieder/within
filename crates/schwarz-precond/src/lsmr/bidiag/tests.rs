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
