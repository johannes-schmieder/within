//! Coefficient ownership fixes summation order without atomics or per-RHS banks.
use rayon::prelude::*;

use crate::domain::{Design, Loading};
use crate::BuildError;

pub(crate) struct StableAdjoint {
    terms: Vec<LevelRows>,
}

struct LevelRows {
    offsets: Vec<usize>,
    // Sorted terms use their contiguous row positions without a second index.
    rows: Option<Vec<usize>>,
}

fn zeros(length: usize) -> Result<Vec<usize>, BuildError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| BuildError::Allocation {
            context: "reproducible adjoint indices",
        })?;
    values.resize(length, 0);
    Ok(values)
}

impl StableAdjoint {
    pub(crate) fn new(design: &Design<'_>) -> Result<Self, BuildError> {
        let mut terms = Vec::new();
        terms
            .try_reserve_exact(design.terms.len())
            .map_err(|_| BuildError::Allocation {
                context: "reproducible adjoint terms",
            })?;
        for (q, term) in design.terms.iter().enumerate() {
            let levels = design.frame.level_column(q);
            let mut offsets =
                zeros(term.n_levels.checked_add(1).ok_or(BuildError::Allocation {
                    context: "reproducible adjoint offset count",
                })?)?;
            for &level in levels {
                offsets[level as usize + 1] += 1;
            }
            for level in 0..term.n_levels {
                offsets[level + 1] += offsets[level];
            }
            let rows = if term.sorted {
                None
            } else {
                let mut rows = zeros(levels.len())?;
                let mut cursors = zeros(term.n_levels)?;
                cursors.copy_from_slice(&offsets[..term.n_levels]);
                for (row, &level) in levels.iter().enumerate() {
                    let cursor = &mut cursors[level as usize];
                    rows[*cursor] = row;
                    *cursor += 1;
                }
                Some(rows)
            };
            terms.push(LevelRows { offsets, rows });
        }
        Ok(Self { terms })
    }

    pub(crate) fn apply(
        &self,
        design: &Design<'_>,
        sqrt_weights: Option<&[f64]>,
        x: &[f64],
        y: &mut [f64],
    ) {
        for (q, term) in design.terms.iter().enumerate() {
            let index = &self.terms[q];
            let block = &mut y[term.offset..term.offset + term.n_dofs()];
            block
                .par_iter_mut()
                .enumerate()
                .for_each(|(coefficient, value)| {
                    let level = coefficient % term.n_levels;
                    let column = coefficient / term.n_levels;
                    let mut sum = 0.0;
                    for position in index.offsets[level]..index.offsets[level + 1] {
                        let row = index.rows.as_ref().map_or(position, |rows| rows[position]);
                        let base = sqrt_weights.map_or(x[row], |sw| sw[row] * x[row]);
                        sum += match term.columns[column] {
                            Loading::Constant => base,
                            Loading::Covariate(c) => {
                                design.frame.loading_column(c as usize)[row] * base
                            }
                        };
                    }
                    *value = sum;
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Effect;

    #[test]
    fn adjoints_match_row_major_witness_across_layouts_and_concurrent_rhs() {
        for rows in [19, 250_001] {
            let levels = if rows == 19 { 7 } else { 120_003 };
            let a: Vec<_> = (0..rows).map(|i| (i * levels / rows) as u32).collect();
            let b: Vec<_> = (0..rows).map(|i| ((i * 7919) % levels) as u32).collect();
            let z: Vec<_> = (0..rows).map(|i| (i % 17) as f64 * 0.1 - 0.7).collect();
            let w: Vec<_> = (0..rows)
                .map(|i| {
                    if i % 19 == 0 {
                        0.0
                    } else {
                        0.3 + (i % 11) as f64
                    }
                })
                .collect();
            let design = Design::new(vec![
                Effect::new(&a, true, [&z[..], &z[..], &z[..]]).unwrap(),
                Effect::new(&b, false, [&z[..], &z[..]]).unwrap(),
            ])
            .unwrap();
            assert!(design.terms[0].sorted);
            assert!(!design.terms[1].sorted);
            let stable = StableAdjoint::new(&design).unwrap();
            assert!(stable.terms[0].rows.is_none());
            for workers in [1, 2, 4, 8, 16] {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap()
                    .install(|| {
                        (0..4).into_par_iter().for_each(|rhs| {
                            let x: Vec<_> = (0..rows)
                                .map(|i| ((i * 37 + rhs * 11) % 101) as f64 / 23.0 - 2.0)
                                .collect();
                            // Independent row-major matrix-transpose action, including empty levels.
                            let mut expected = vec![0.0; design.n_dofs];
                            for i in 0..rows {
                                for (q, term) in design.terms.iter().enumerate() {
                                    for (c, loading) in term.columns.iter().enumerate() {
                                        let base = w[i] * x[i];
                                        let value = match loading {
                                            Loading::Constant => base,
                                            Loading::Covariate(k) => {
                                                design.frame.loading_column(*k as usize)[i] * base
                                            }
                                        };
                                        expected[term.column_base(c)
                                            + design.frame.level_column(q)[i] as usize] += value;
                                    }
                                }
                            }
                            for _ in 0..2 {
                                let mut actual = vec![f64::NAN; design.n_dofs];
                                stable.apply(&design, Some(&w), &x, &mut actual);
                                assert_eq!(
                                    actual, expected,
                                    "rows={rows} workers={workers} rhs={rhs}"
                                );
                            }
                        });
                    });
            }
        }
    }
}
