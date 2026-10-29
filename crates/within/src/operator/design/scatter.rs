//! Scatter kernel: observation space → coefficient space (`Dᵀ x`), summed per term in an
//! order its [`AdjointSum`] fixes, so the thread count never changes the bits.

use std::ops::Range;

use rayon::prelude::*;

use super::PAR_THRESHOLD;
use crate::domain::{AdjointSum, LevelRows, PreparedDesign, PreparedTerm};

/// Adjoint scatter over all terms; `base(i)` is the row value each column scales by its loading.
pub(super) fn scatter_apply(
    prepared: &PreparedDesign<'_>,
    dst: &mut [f64],
    base: &(impl Fn(usize) -> f64 + Sync),
) {
    let design = &prepared.design;
    debug_assert_eq!(dst.len(), design.n_dofs);
    let parallel = design.n_obs > PAR_THRESHOLD;

    for t in prepared.terms() {
        let term = t.term;
        let block = &mut dst[term.dofs()];
        match (term.intercept, t.slopes) {
            (true, []) => scatter_term::<1>(block, &t, parallel, |i| [base(i)]),
            (true, [z0]) => {
                let z0 = &z0[..];
                scatter_term::<2>(block, &t, parallel, |i| {
                    let b = base(i);
                    [b, z0[i] * b]
                })
            }
            (true, [z0, z1]) => {
                let (z0, z1) = (&z0[..], &z1[..]);
                scatter_term::<3>(block, &t, parallel, |i| {
                    let b = base(i);
                    [b, z0[i] * b, z1[i] * b]
                })
            }
            (true, [z0, z1, z2]) => {
                let (z0, z1, z2) = (&z0[..], &z1[..], &z2[..]);
                scatter_term::<4>(block, &t, parallel, |i| {
                    let b = base(i);
                    [b, z0[i] * b, z1[i] * b, z2[i] * b]
                })
            }
            (false, [z0, z1]) => {
                let (z0, z1) = (&z0[..], &z1[..]);
                scatter_term::<2>(block, &t, parallel, |i| {
                    let b = base(i);
                    [z0[i] * b, z1[i] * b]
                })
            }
            (false, [z0, z1, z2]) => {
                let (z0, z1, z2) = (&z0[..], &z1[..], &z2[..]);
                scatter_term::<3>(block, &t, parallel, |i| {
                    let b = base(i);
                    [z0[i] * b, z1[i] * b, z2[i] * b]
                })
            }
            _ => {
                for c in 0..term.n_columns() {
                    let start = c * term.n_levels();
                    let slot = &mut block[start..start + term.n_levels()];
                    match t.loading(c) {
                        None => scatter_term::<1>(slot, &t, parallel, |i| [base(i)]),
                        Some(z) => scatter_term::<1>(slot, &t, parallel, move |i| [z[i] * base(i)]),
                    }
                }
            }
        }
    }
}

/// Write one term's coefficient block: `block[c·L + l] = Σ_{i : level(i) = l} values(i)[c]`.
fn scatter_term<const C: usize>(
    block: &mut [f64],
    term: &PreparedTerm<'_>,
    parallel: bool,
    values: impl Fn(usize) -> [f64; C] + Sync,
) {
    let (n_levels, levels) = (term.term.n_levels(), term.term.levels());
    debug_assert_eq!(block.len(), C * n_levels);
    match term.term.adjoint() {
        AdjointSum::Owner(rows) => owner_sums::<C>(block, rows, parallel, &values),
        AdjointSum::Fold => {
            // Leaves of ≥ 4·C·L rows keep the combine pass under a quarter of the row pass.
            let leaf = (4 * C * n_levels).max(65_536);
            let scatter = |rows: Range<usize>| {
                let mut sums = vec![0.0; C * n_levels];
                for i in rows {
                    let level = levels[i] as usize;
                    for (c, v) in values(i).into_iter().enumerate() {
                        sums[c * n_levels + level] += v;
                    }
                }
                sums
            };
            block.copy_from_slice(&fixed_tree(0..levels.len(), leaf, &scatter, &add));
        }
    }
}

/// Levels per parallel task in [`owner_sums`].
const LEVEL_GROUP: usize = 1024;

/// Rows per leaf of one level's sum; a leaf adds its rows in order.
const LEVEL_LEAF: usize = 4096;

/// Each level's `C` column sums over its own rows, written to `block[c·L + l]`.
fn owner_sums<const C: usize>(
    block: &mut [f64],
    rows: &LevelRows,
    parallel: bool,
    values: &(impl Fn(usize) -> [f64; C] + Sync),
) {
    let level_sum = |level: usize| {
        let leaf = |positions: Range<usize>| {
            let mut sums = [0.0; C];
            for p in positions {
                for (s, v) in sums.iter_mut().zip(values(rows.row(p))) {
                    *s += v;
                }
            }
            sums
        };
        fixed_tree(rows.positions(level), LEVEL_LEAF, &leaf, &add)
    };
    let mut columns = block
        .chunks_mut(rows.n_levels())
        .map(|c| c.chunks_mut(LEVEL_GROUP));
    let mut columns: [_; C] = std::array::from_fn(|_| columns.next().expect("C columns"));
    let groups = (0..rows.n_levels().div_ceil(LEVEL_GROUP)).map(|_| {
        columns
            .each_mut()
            .map(|c| c.next().expect("a group per column"))
    });
    let fill = |(g, mut group): (usize, [&mut [f64]; C])| {
        for k in 0..group[0].len() {
            let sums = level_sum(g * LEVEL_GROUP + k);
            for (column, s) in group.iter_mut().zip(sums) {
                column[k] = s;
            }
        }
    };
    if parallel {
        groups
            .collect::<Vec<_>>()
            .into_par_iter()
            .enumerate()
            .for_each(fill);
    } else {
        groups.enumerate().for_each(fill);
    }
}

fn add<S: AsMut<[f64]>>(mut a: S, mut b: S) -> S {
    for (x, y) in a.as_mut().iter_mut().zip(b.as_mut()) {
        *x += *y;
    }
    a
}

/// `leaf` over `block`-row pieces of `rows`, added in a binary tree fixed by the row count alone.
fn fixed_tree<T: Send>(
    rows: Range<usize>,
    block: usize,
    leaf: &(impl Fn(Range<usize>) -> T + Sync),
    add: &(impl Fn(T, T) -> T + Sync),
) -> T {
    if rows.len() <= block {
        return leaf(rows);
    }
    let mid = rows.start + (rows.len() / 2 / block).max(1) * block;
    let (left, right) = rayon::join(
        || fixed_tree(rows.start..mid, block, leaf, add),
        || fixed_tree(mid..rows.end, block, leaf, add),
    );
    add(left, right)
}
