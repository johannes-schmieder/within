use super::scatter_term;
use crate::domain::{AdjointSum, Design, Effect, PreparedDesign};
use crate::operator::DesignOperator;
use crate::BuildError;
use schwarz_precond::Operator;
use std::ops::Range;
use std::sync::Mutex;

fn assert_bits(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (slot, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "coefficient {slot}");
    }
}

/// Independent coefficient-first reference: collect each level's row positions,
/// then sum each coefficient in row order. Only the required tree partition is
/// shared mathematically; neither production scatter nor fixed_tree is called.
fn coefficient_reference(
    levels: &[u32],
    n_levels: usize,
    columns: &[Vec<f64>],
    rows: Range<usize>,
    leaf: usize,
) -> Vec<f64> {
    if rows.len() > leaf {
        let split = rows.start + (rows.len() / 2 / leaf).max(1) * leaf;
        let mut left = coefficient_reference(levels, n_levels, columns, rows.start..split, leaf);
        let right = coefficient_reference(levels, n_levels, columns, split..rows.end, leaf);
        for (a, b) in left.iter_mut().zip(right) {
            *a += b;
        }
        return left;
    }
    let mut positions = vec![Vec::new(); n_levels];
    for i in rows {
        positions[levels[i] as usize].push(i);
    }
    columns
        .iter()
        .flat_map(|column| {
            positions.iter().map(move |positions| {
                let mut sum = 0.0;
                for &i in positions {
                    sum += column[i];
                }
                sum
            })
        })
        .collect()
}

fn noise(i: usize) -> f64 {
    let mut x = (i as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((x ^ (x >> 31)) % 1024) as f64 / 128.0 - 4.0
}

fn codes(n: usize, n_levels: usize) -> Vec<u32> {
    let mut result: Vec<u32> = (0..n)
        .map(|i| {
            let level = if i < n_levels {
                // Every level is present, and the term is not sorted.
                n_levels - 1 - i
            } else {
                match (i / 4096) % 3 {
                    0 => (i.wrapping_mul(7919) ^ (i / 7)) % n_levels,
                    1 => (i / 257) % n_levels,
                    _ => i % n_levels,
                }
            };
            level as u32
        })
        .collect();
    if n > n_levels + 5 {
        // Revisit a coefficient after another level: a run subtotal from zero
        // would lose the final unit for [1e16, 1, -1e16, 1].
        result[n_levels..n_levels + 5].copy_from_slice(&[0, 0, 1, 0, 0]);
    }
    result
}

fn prepared(
    levels: &[u32],
    intercept: bool,
    slopes: usize,
    weighted: bool,
) -> PreparedDesign<'static> {
    let loadings: Vec<Vec<f64>> = (0..slopes)
        .map(|c| (0..levels.len()).map(|i| noise(31 * i + 997 * c)).collect())
        .collect();
    let effect = Effect::new(levels, intercept, loadings.iter().map(Vec::as_slice)).unwrap();
    let design = Design::new_unsorted([effect]).unwrap().into_owned();
    let weights: Vec<f64> = (0..levels.len())
        .map(|i| [0.0, 0.25, 1.0, 4.0][i % 4])
        .collect();
    let result = PreparedDesign::new(design, weighted.then_some(weights.as_slice())).unwrap();
    assert!(result.design.obs_perm.is_none());
    assert!(matches!(result.term(0).term.adjoint(), AdjointSum::Fold));
    result
}

fn check_arity<const C: usize>() {
    let wide_levels = 65_536 / (4 * C) + 1;
    let wide_leaf = 4 * C * wide_levels;
    let cases = [
        (12 * C * 7, 7),
        (65_535, 7),
        (65_536, 7),
        (65_537, 7),
        (131_073, 7),
        (3 * wide_leaf + 1, wide_levels),
    ];
    for (n, n_levels) in cases {
        let mut levels = codes(n, n_levels);
        let leaf = (4 * C * n_levels).max(65_536);
        for boundary in [leaf, 2 * leaf] {
            if boundary + 3 < n {
                // A run must be split exactly at the existing leaf boundary.
                levels[boundary - 3..boundary + 3].fill(0);
            }
        }
        let design = prepared(&levels, true, C - 1, false);
        assert_eq!(design.term(0).term.n_levels(), n_levels);
        let columns: Vec<Vec<f64>> = (0..C)
            .map(|c| {
                let mut values: Vec<f64> = (0..n)
                    .map(|i| match (i + c) % 8 {
                        0 => 1e150,
                        1 => -1e150,
                        2 => -0.0,
                        3 => f64::from_bits(1),
                        _ => noise(i + 17 * c),
                    })
                    .collect();
                values[n_levels..n_levels + 5].copy_from_slice(&[1e16, 1.0, 3.0, -1e16, 1.0]);
                values
            })
            .collect();
        let expected = coefficient_reference(&levels, n_levels, &columns, 0..n, leaf);
        for workers in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    for _ in 0..2 {
                        let mut actual = vec![f64::NAN; C * n_levels];
                        scatter_term::<C>(&mut actual, &design.term(0), true, |i| {
                            std::array::from_fn(|c| columns[c][i])
                        });
                        assert_bits(&actual, &expected);
                    }
                });
        }
    }
}

#[test]
fn fold_runs_preserve_coefficient_order_all_arities_and_leaf_boundaries() {
    check_arity::<1>();
    check_arity::<2>();
    check_arity::<3>();
    check_arity::<4>();
}

#[test]
fn fold_runs_match_weighted_sloped_and_general_operator_dispatch() {
    let n = 65_549;
    let levels = codes(n, 7);
    let input: Vec<f64> = (0..n).map(|i| noise(i * 13)).collect();
    for weighted in [false, true] {
        // Fused C=1..4, slope-only C=2/3 and all scalar fallback shapes.
        for (intercept, slopes) in [
            (true, 0),
            (true, 1),
            (true, 2),
            (true, 3),
            (true, 4),
            (false, 1),
            (false, 2),
            (false, 3),
            (false, 4),
            (false, 5),
        ] {
            let design = prepared(&levels, intercept, slopes, weighted);
            let term = design.term(0);
            let columns: Vec<Vec<f64>> = (0..term.term.n_columns())
                .map(|c| {
                    let loading = term.loading(c);
                    let column: Vec<f64> = (0..n)
                        .map(|i| {
                            let base = design
                                .sqrt_weights()
                                .map_or(input[i], |sw| sw[i] * input[i]);
                            loading.map_or(base, |z| z[i] * base)
                        })
                        .collect();
                    assert!(
                        column.iter().any(|v| *v != 0.0),
                        "an addressing arm was whitened away"
                    );
                    column
                })
                .collect();
            // Generic fallback calls scalar scatter separately, so C=1 fixes
            // its leaf size even when the term contains more columns.
            let fused = matches!((intercept, slopes), (true, 0..=3) | (false, 2..=3));
            let arity = if fused { columns.len() } else { 1 };
            let leaf = (4 * arity * term.term.n_levels()).max(65_536);
            let expected =
                coefficient_reference(&levels, term.term.n_levels(), &columns, 0..n, leaf);
            for workers in [1, 4] {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap()
                    .install(|| {
                        for _ in 0..2 {
                            let mut actual = vec![f64::NAN; expected.len()];
                            DesignOperator::new(&design)
                                .apply_adjoint(&input, &mut actual)
                                .unwrap();
                            assert_bits(&actual, &expected);
                        }
                    });
            }
        }
    }
}

#[test]
fn fold_runs_preserve_nonfinite_and_signed_zero_kernel_edges() {
    let levels = codes(128, 2);
    let design = prepared(&levels, true, 0, false);
    for edge in [
        0.0,
        -0.0,
        f64::MAX,
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8_0000_0000_0042),
    ] {
        let mut column = vec![-0.0; levels.len()];
        column[2..7].copy_from_slice(&[edge, edge, 0.0, -edge, 1.0]);
        let expected =
            coefficient_reference(&levels, 2, &[column.clone()], 0..levels.len(), 65_536);
        let mut actual = vec![123.0; 2];
        scatter_term::<1>(&mut actual, &design.term(0), false, |i| [column[i]]);
        assert_bits(&actual, &expected);
    }
}

#[test]
fn fold_runs_callback_order_and_panic_leave_output_unpublished() {
    let levels = codes(128, 2);
    let design = prepared(&levels, true, 3, false);
    let visited = Mutex::new(Vec::new());
    let mut output = vec![123.0; 8];
    scatter_term::<4>(&mut output, &design.term(0), false, |i| {
        visited.lock().unwrap().push(i);
        [i as f64; 4]
    });
    assert_eq!(
        *visited.lock().unwrap(),
        (0..levels.len()).collect::<Vec<_>>()
    );
    visited.lock().unwrap().clear();
    output.fill(123.0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scatter_term::<4>(&mut output, &design.term(0), false, |i| {
            visited.lock().unwrap().push(i);
            assert_ne!(i, 25, "intentional callback failure");
            [i as f64; 4]
        });
    }));
    assert!(result.is_err());
    assert_eq!(*visited.lock().unwrap(), (0..=25).collect::<Vec<_>>());
    assert_bits(&output, &[123.0; 8]);

    // Valid designs reject empty observations before any kernel can execute.
    let empty = Effect::new(&[], true, []).unwrap();
    assert!(matches!(
        Design::new_unsorted([empty]),
        Err(BuildError::EmptyObservations)
    ));
}
