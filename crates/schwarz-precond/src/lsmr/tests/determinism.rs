//! Thread-count independence: the same solve returns the same bits in any pool.

use super::super::*;
use crate::{Operator, SolveError};

/// `A x` with `y[i] = x[i mod n] + ½ x[(7i + 3) mod n]`: large enough that every reduction splits.
struct Wrapped {
    m: usize,
    n: usize,
}

impl Wrapped {
    fn cols(&self, i: usize) -> (usize, usize) {
        (i % self.n, (7 * i + 3) % self.n)
    }
}

impl Operator for Wrapped {
    fn nrows(&self) -> usize {
        self.m
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        for (i, yi) in y.iter_mut().enumerate() {
            let (a, b) = self.cols(i);
            *yi = x[a] + 0.5 * x[b];
        }
        Ok(())
    }
    fn apply_adjoint(&self, u: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
        x.fill(0.0);
        for (i, &ui) in u.iter().enumerate() {
            let (a, b) = self.cols(i);
            x[a] += ui;
            x[b] += 0.5 * ui;
        }
        Ok(())
    }
}

/// `M = diag(d)` with entries spread over three decades, so `⟨v, M⁻¹p̃⟩` is not a rescaled norm.
struct Diagonal(Vec<f64>);

impl Operator for Diagonal {
    fn nrows(&self) -> usize {
        self.0.len()
    }
    fn ncols(&self) -> usize {
        self.0.len()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        for ((yi, &xi), &di) in y.iter_mut().zip(x).zip(&self.0) {
            *yi = di * xi;
        }
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}

fn spread(len: usize, scale: f64) -> Vec<f64> {
    (0..len)
        .map(|i| ((i as u64 * 2_654_435_761) % 1_000) as f64 * scale / 1_000.0 - 0.5 * scale)
        .collect()
}

/// Everything a caller can read off a result, as bits.
fn fingerprint(r: &LsmrResult) -> (Vec<u64>, usize, u64, u64, LsmrStopReason) {
    (
        r.x.iter().map(|v| v.to_bits()).collect(),
        r.iterations,
        r.residual_norm.to_bits(),
        r.normal_eq_residual.to_bits(),
        r.stop_reason,
    )
}

fn in_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool")
        .install(f)
}

#[test]
fn lsmr_returns_the_same_bits_at_any_thread_count() {
    let op = Wrapped {
        m: 300_000,
        n: 60_000,
    };
    let b = spread(op.m, 1.0);
    let solve = || fingerprint(&lsmr(&op, &b, 1e-14, 40, None).expect("lsmr"));
    let reference = in_pool(1, solve);
    for threads in [2, 3, 8] {
        assert!(in_pool(threads, solve) == reference, "{threads} threads");
    }
}

#[test]
fn mlsmr_returns_the_same_bits_at_any_thread_count() {
    let op = Wrapped {
        m: 300_000,
        n: 60_000,
    };
    let b = spread(op.m, 1.0);
    let m = Diagonal(
        spread(op.n, 1.0)
            .iter()
            .map(|v| 10f64.powf(3.0 * (v + 0.5)))
            .collect(),
    );
    let solve = || {
        let r = mlsmr(&op, &b, &m, 1e-14, 40, MlsmrOptions::default()).expect("mlsmr");
        fingerprint(&r)
    };
    let reference = in_pool(1, solve);
    for threads in [2, 3, 8] {
        assert!(in_pool(threads, solve) == reference, "{threads} threads");
    }
}
