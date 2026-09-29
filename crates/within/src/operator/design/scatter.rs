//! Observation space → coefficient space with disjoint per-level destinations.

use crate::domain::{Design, LevelMembership, Loading};
use rayon::prelude::*;
use std::ops::Range;

/// Every level sums stable row blocks, then combines them with a fixed binary
/// tree. Scheduling can change, but neither row order nor arithmetic can.
fn level_sum(
    membership: &LevelMembership,
    range: Range<usize>,
    value: &(impl Fn(usize) -> f64 + Sync),
) -> f64 {
    const CHUNK: usize = 4096;
    if range.len() <= CHUNK {
        return range
            .map(|position| value(membership.row(position)))
            .fold(0.0, |sum, value| sum + value);
    }
    let mid = range.start + (range.len() / 2 / CHUNK).max(1) * CHUNK;
    let (left, right) = rayon::join(
        || level_sum(membership, range.start..mid, value),
        || level_sum(membership, mid..range.end, value),
    );
    left + right
}

pub(super) fn scatter_apply(
    design: &Design<'_>,
    dst: &mut [f64],
    base: &(impl Fn(usize) -> f64 + Sync),
) {
    debug_assert_eq!(dst.len(), design.n_dofs);
    for (q, term) in design.terms.iter().enumerate() {
        let membership = &design.membership[q];
        let block = &mut dst[term.offset..term.offset + term.n_dofs()];
        for (column, loading) in term.columns.iter().enumerate() {
            let slot = &mut block[column * term.n_levels..(column + 1) * term.n_levels];
            let value = |row| match loading {
                Loading::Constant => base(row),
                Loading::Covariate(k) => design.frame.loading_column(*k as usize)[row] * base(row),
            };
            let apply = |(level, output): (usize, &mut f64)| {
                *output = level_sum(membership, membership.level_range(level), &value);
            };
            if design.n_obs > super::PAR_THRESHOLD {
                slot.par_iter_mut().enumerate().for_each(apply);
            } else {
                slot.iter_mut().enumerate().for_each(apply);
            }
        }
    }
}
