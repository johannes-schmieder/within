//! How the adjoint sums a term's rows into its coefficients, fixed by the rows alone.

use std::ops::Range;

/// Rows per coefficient slot from which streaming rows into partial blocks beats walking each level.
const FOLD_ROWS_PER_SLOT: usize = 12;

/// The summation order of `Dᵀ` on one term; the thread count only runs its fixed pieces.
#[derive(Clone, Debug)]
pub(crate) enum AdjointSum {
    /// Each level sums its own rows in row order.
    Owner(LevelRows),
    /// Fixed row blocks scatter into partial blocks that a fixed binary tree adds up.
    Fold,
}

impl AdjointSum {
    pub(crate) fn new(levels: &[u32], n_levels: usize, n_columns: usize, sorted: bool) -> Self {
        let n = levels.len();
        // An unsorted term's row index stores u32 rows.
        if !sorted && (n >= FOLD_ROWS_PER_SLOT * n_columns * n_levels || u32::try_from(n).is_err())
        {
            Self::Fold
        } else {
            Self::Owner(LevelRows::new(levels, n_levels, sorted))
        }
    }
}

/// The rows of each level, ascending: the inverse of the row → level map.
#[derive(Clone, Debug)]
pub(crate) struct LevelRows {
    /// Level `l` owns positions `offsets[l]..offsets[l + 1]`.
    offsets: Vec<usize>,
    /// Row at each position; `None` when the levels are sorted, so positions are rows.
    rows: Option<Box<[u32]>>,
}

impl LevelRows {
    fn new(levels: &[u32], n_levels: usize, sorted: bool) -> Self {
        let mut offsets = vec![0; n_levels + 1];
        if sorted {
            // One store per run: a per-row increment chains each run through one counter.
            for run in levels.chunk_by(|a, b| a == b) {
                offsets[run[0] as usize + 1] = run.len();
            }
        } else {
            for &level in levels {
                offsets[level as usize + 1] += 1;
            }
        }
        for level in 0..n_levels {
            offsets[level + 1] += offsets[level];
        }
        let rows = (!sorted).then(|| {
            let mut rows = vec![0; levels.len()].into_boxed_slice();
            let mut cursors = offsets[..n_levels].to_vec();
            for (row, &level) in (0u32..).zip(levels) {
                let cursor = &mut cursors[level as usize];
                rows[*cursor] = row;
                *cursor += 1;
            }
            rows
        });
        Self { offsets, rows }
    }

    pub(crate) fn n_levels(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Positions of `level`'s rows, for [`row`](Self::row).
    pub(crate) fn positions(&self, level: usize) -> Range<usize> {
        self.offsets[level]..self.offsets[level + 1]
    }

    pub(crate) fn row(&self, position: usize) -> usize {
        self.rows
            .as_ref()
            .map_or(position, |rows| rows[position] as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows_of(rows: &LevelRows) -> Vec<Vec<usize>> {
        (0..rows.n_levels())
            .map(|l| rows.positions(l).map(|p| rows.row(p)).collect())
            .collect()
    }

    #[test]
    fn level_rows_list_each_levels_rows_ascending() {
        let rows = LevelRows::new(&[2, 0, 2, 1, 0], 4, false);
        assert_eq!(rows_of(&rows), [vec![1, 4], vec![3], vec![0, 2], vec![]]);
        let sorted = LevelRows::new(&[0, 0, 2, 2, 2], 3, true);
        assert_eq!(rows_of(&sorted), [vec![0, 1], vec![], vec![2, 3, 4]]);
    }

    #[test]
    fn fold_needs_unsorted_rows_and_enough_rows_per_slot() {
        let levels: Vec<u32> = (0..48).map(|i| i % 2).collect();
        assert!(matches!(
            AdjointSum::new(&levels, 2, 2, false),
            AdjointSum::Fold
        ));
        assert!(matches!(
            AdjointSum::new(&levels, 2, 2, true),
            AdjointSum::Owner(_)
        ));
        assert!(matches!(
            AdjointSum::new(&levels[..47], 2, 2, false),
            AdjointSum::Owner(_)
        ));
    }
}
