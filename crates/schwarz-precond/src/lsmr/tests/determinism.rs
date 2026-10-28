//! Thread-count independence: the same solve returns the same bits in any pool.

use super::super::*;
use crate::fold::assert_pool_independent;
use crate::lsmr::fixtures::DiagOp;
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

#[test]
fn lsmr_returns_the_same_bits_at_any_thread_count() {
    let op = Wrapped {
        m: 300_000,
        n: 60_000,
    };
    let b = spread(op.m, 1.0);
    assert_pool_independent(|| fingerprint(&lsmr(&op, &b, 1e-14, 40, None).expect("lsmr")));
}

#[test]
fn mlsmr_returns_the_same_bits_at_any_thread_count() {
    let op = Wrapped {
        m: 300_000,
        n: 60_000,
    };
    let b = spread(op.m, 1.0);
    // Entries over three decades, so `⟨v, M⁻¹p̃⟩` is not a rescaled norm.
    let m = DiagOp(
        spread(op.n, 1.0)
            .iter()
            .map(|v| 10f64.powf(3.0 * (v + 0.5)))
            .collect(),
    );
    assert_pool_independent(|| {
        let r = mlsmr(&op, &b, &m, 1e-14, 40, MlsmrOptions::default()).expect("mlsmr");
        fingerprint(&r)
    });
}
