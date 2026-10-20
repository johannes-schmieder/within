//! The checked build's Schur complements: serial, with every array reserved
//! through a [`Ledger`], and in the same arithmetic order as [`super::exact`]
//! and [`super::sampled`].
//!
//! Row splits and star folds never change a value: the exact rows are computed
//! independently, and sampled edges are reseeded per star and summed after one
//! total-order sort. So the serial forms reproduce the parallel ones bit for bit.

use approx_chol::low_level::CliqueTreeSampler;

use super::super::compensated_sum;
use super::super::csr_matrix::CsrMatrix;
use super::{star, Edge, UpperSchurRow};
use crate::config::ApproxSchurConfig;
use crate::csr_block::to_u32;
use crate::domain::{Grounding, SddmMatrix};
use crate::setup::Ledger;
use crate::BuildError;

const EXACT: &str = "exact Schur complement";
const SAMPLED: &str = "sampled Schur complement";

/// Refuse a reduced matrix whose `u32` row offsets would overflow.
fn offset_count(nnz: usize) -> Result<usize, BuildError> {
    if u32::try_from(nnz).is_err() {
        return Err(BuildError::CheckedSetupUnsupported {
            reason: "a reduced local matrix with more than u32::MAX nonzeros",
        });
    }
    Ok(nnz)
}

/// [`super::exact_for_factor`], checked.
pub(crate) fn exact_for_factor_checked(
    matrix: &SddmMatrix,
    inv_diagonal_eliminated: &[f64],
    ledger: &mut Ledger,
) -> Result<CsrMatrix, BuildError> {
    let principal = exact(matrix, inv_diagonal_eliminated, ledger)?;
    let surplus = reduced_surplus(matrix, inv_diagonal_eliminated, ledger)?;
    let laplacian = build_explicit_laplacian(&principal, &surplus, matrix.grounding, ledger)?;
    principal.free_checked(ledger);
    ledger.free(surplus);
    Ok(laplacian)
}

/// [`super::compute_schur_row_dense`] with a ledger-grown `touched`: a column whose
/// partial sum cancels to zero is recorded again, so its length is not bounded by
/// the row width.
fn compute_schur_row_dense(
    matrix: &SddmMatrix,
    inv_diagonal_eliminated: &[f64],
    i: usize,
    work: &mut [f64],
    touched: &mut Vec<usize>,
    ledger: &mut Ledger,
) -> Result<(), BuildError> {
    work[i] = matrix.diagonal_kept()[i];
    ledger.push(EXACT, touched, i)?;

    let elim_to_keep = &matrix.cross_tab.c;
    for (k, w) in matrix.cross_tab.ct.row(i) {
        let inv = inv_diagonal_eliminated[k];
        let start = elim_to_keep.indptr[k] as usize;
        let end = elim_to_keep.indptr[k + 1] as usize;
        let columns = &elim_to_keep.indices[start..end];
        let upper_start = columns.partition_point(|&j| (j as usize) < i);
        for position in start + upper_start..end {
            let j = elim_to_keep.indices[position] as usize;
            let v = elim_to_keep.data[position];
            if work[j] == 0.0 && j != i {
                ledger.push(EXACT, touched, j)?;
            }
            work[j] -= w * v * inv;
        }
    }
    Ok(())
}

/// [`super::extract_sparse_row`], checked.
fn extract_sparse_row(
    i: usize,
    work: &mut [f64],
    touched: &mut [usize],
    ledger: &mut Ledger,
) -> Result<UpperSchurRow, BuildError> {
    touched.sort_unstable();
    let mut columns = ledger.with_capacity(EXACT, touched.len())?;
    let mut values = ledger.with_capacity(EXACT, touched.len())?;
    for &j in touched.iter() {
        let v = work[j];
        if v != 0.0 || j == i {
            columns.push(to_u32(j));
            values.push(v);
        }
        work[j] = 0.0;
    }
    Ok(UpperSchurRow { columns, values })
}

/// [`super::exact`] with sequential rows, checked.
fn exact(
    matrix: &SddmMatrix,
    inv_diagonal_eliminated: &[f64],
    ledger: &mut Ledger,
) -> Result<CsrMatrix, BuildError> {
    let n_keep = matrix.n_kept();
    let mut work = ledger.filled(EXACT, n_keep, 0.0f64)?;
    let mut touched = Vec::new();
    let mut rows = ledger.with_capacity(EXACT, n_keep)?;
    for i in 0..n_keep {
        compute_schur_row_dense(
            matrix,
            inv_diagonal_eliminated,
            i,
            &mut work,
            &mut touched,
            ledger,
        )?;
        rows.push(extract_sparse_row(i, &mut work, &mut touched, ledger)?);
        touched.clear();
    }
    ledger.free(work);
    ledger.free(touched);
    assemble_schur_csr(rows, n_keep, ledger)
}

/// [`super::assemble_schur_csr`], checked; releases each row as it is consumed.
fn assemble_schur_csr(
    rows: Vec<UpperSchurRow>,
    n_keep: usize,
    ledger: &mut Ledger,
) -> Result<CsrMatrix, BuildError> {
    // Each row stores its diagonal once and mirrors every strict-upper entry.
    let total_nnz = rows
        .iter()
        .try_fold(0usize, |sum, row| {
            sum.checked_add(2 * row.columns.len() - 1)
        })
        .ok_or(BuildError::SetupAllocationFailed {
            context: EXACT,
            bytes: usize::MAX,
        })?;
    let total_nnz = offset_count(total_nnz)?;
    let mut row_offsets = ledger.filled(EXACT, n_keep + 1, 0u32)?;
    for (i, row) in rows.iter().enumerate() {
        row_offsets[i + 1] += to_u32(row.columns.len());
        for &j in &row.columns {
            if j as usize != i {
                row_offsets[j as usize + 1] += 1;
            }
        }
    }
    for i in 0..n_keep {
        row_offsets[i + 1] += row_offsets[i];
    }
    debug_assert_eq!(row_offsets[n_keep] as usize, total_nnz);
    let mut column_indices = ledger.filled(EXACT, total_nnz, 0u32)?;
    let mut values = ledger.filled(EXACT, total_nnz, 0.0f64)?;

    let mut lower_write_positions = ledger.copied(EXACT, &row_offsets[..n_keep])?;
    let rows_capacity = rows.capacity();
    for (i, row) in rows.into_iter().enumerate() {
        let upper_start = row_offsets[i + 1] as usize - row.columns.len();
        column_indices[upper_start..upper_start + row.columns.len()].copy_from_slice(&row.columns);
        values[upper_start..upper_start + row.values.len()].copy_from_slice(&row.values);
        for (&j, &v) in row.columns.iter().zip(&row.values) {
            if j as usize != i {
                let position = lower_write_positions[j as usize] as usize;
                column_indices[position] = to_u32(i);
                values[position] = v;
                lower_write_positions[j as usize] += 1;
            }
        }
        ledger.free(row.columns);
        ledger.free(row.values);
    }
    ledger.forget::<UpperSchurRow>(rows_capacity);
    ledger.free(lower_write_positions);
    Ok(CsrMatrix::new(row_offsets, column_indices, values, n_keep))
}

/// [`super::reduced_surplus`], checked.
fn reduced_surplus(
    matrix: &SddmMatrix,
    inv_diagonal_eliminated: &[f64],
    ledger: &mut Ledger,
) -> Result<Vec<f64>, BuildError> {
    if matrix.grounding == Grounding::Floating {
        return ledger.filled(EXACT, matrix.n_kept(), 0.0);
    }
    let mut scaled = ledger.with_capacity(EXACT, inv_diagonal_eliminated.len())?;
    scaled.extend(
        inv_diagonal_eliminated
            .iter()
            .zip(matrix.surplus_eliminated())
            .map(|(&inv_diag, &surplus)| inv_diag * surplus),
    );
    let mut surplus = ledger.filled(EXACT, matrix.n_kept(), 0.0)?;
    matrix
        .cross_tab
        .ct
        .spmv_assign_add(&scaled, matrix.surplus_kept(), &mut surplus, false);
    ledger.free(scaled);
    Ok(surplus)
}

/// [`super::build_explicit_laplacian`] with its exact entry count reserved up front.
fn build_explicit_laplacian(
    principal: &CsrMatrix,
    surplus: &[f64],
    grounding: Grounding,
    ledger: &mut Ledger,
) -> Result<CsrMatrix, BuildError> {
    let n_keep = principal.n();
    let ground = to_u32(n_keep);
    let grounded = grounding == Grounding::Grounded;
    let n = n_keep + usize::from(grounded);
    let positive = if grounded {
        surplus.iter().take(n_keep).filter(|&&s| s > 0.0).count()
    } else {
        0
    };
    // Every principal entry, a ground column per positive row, then the ground row.
    let nnz = principal.indices().len() + if grounded { 2 * positive + 1 } else { 0 };
    let nnz = offset_count(nnz)?;
    let mut indptr = ledger.with_capacity(EXACT, n + 1)?;
    let mut indices = ledger.with_capacity(EXACT, nnz)?;
    let mut data = ledger.with_capacity(EXACT, nnz)?;
    indptr.push(0);

    for (i, &row_surplus) in surplus.iter().enumerate().take(n_keep) {
        let start = principal.indptr()[i] as usize;
        let end = principal.indptr()[i + 1] as usize;
        let mut adjacency = 0.0;
        let mut diagonal_position = None;
        for (&j, &value) in principal.indices()[start..end]
            .iter()
            .zip(&principal.data()[start..end])
        {
            if j as usize == i {
                diagonal_position = Some(indices.len());
                indices.push(j);
                data.push(0.0);
            } else {
                adjacency -= value;
                indices.push(j);
                data.push(value);
            }
        }
        let diagonal_position = diagonal_position.expect("exact Schur row must contain a diagonal");
        data[diagonal_position] = adjacency + row_surplus;
        if grounded && row_surplus > 0.0 {
            indices.push(ground);
            data.push(-row_surplus);
        }
        indptr.push(to_u32(indices.len()));
    }

    if grounded {
        for (i, &value) in surplus.iter().enumerate() {
            if value > 0.0 {
                indices.push(to_u32(i));
                data.push(-value);
            }
        }
        indices.push(ground);
        data.push(compensated_sum(surplus));
        indptr.push(to_u32(indices.len()));
    }
    debug_assert_eq!(indices.len(), nnz);
    Ok(CsrMatrix::new(indptr, indices, data, n))
}

/// [`super::sampled`] with one sampler over the stars in order, checked.
pub(crate) fn sampled_checked(
    matrix: &SddmMatrix,
    config: &ApproxSchurConfig,
    ledger: &mut Ledger,
) -> Result<CsrMatrix, BuildError> {
    let edges = emit(matrix, config, ledger)?;
    let n = matrix.n_kept() + usize::from(matrix.grounding == Grounding::Grounded);
    let laplacian = build_laplacian_csr(&edges, n, ledger)?;
    ledger.free(edges);
    Ok(laplacian)
}

/// [`super::par_emit`] serially: the sampler's scratch and every star's fill are
/// reserved before the first star is sampled, so sampling never allocates.
fn emit(
    matrix: &SddmMatrix,
    config: &ApproxSchurConfig,
    ledger: &mut Ledger,
) -> Result<Vec<Edge>, BuildError> {
    let n_kept = matrix.n_kept();
    let surplus_eliminated = matrix.surplus_eliminated();
    let ground_vertex = (matrix.grounding == Grounding::Grounded).then(|| to_u32(n_kept));
    // Entries of eliminated star `k`, counted as `StarWorkspace::sample_star` fills them.
    let entries_of = |k: usize| {
        let degree = star(matrix, k).degree();
        let ground = ground_vertex.is_some() && surplus_eliminated[k] > 0.0;
        (degree, degree + usize::from(ground))
    };

    let mut sampler = CliqueTreeSampler::new(config.seed, Some(config.split));
    // Every star with a neighbor fills `entries`; only those with two or more sample.
    let mut max_star = 0;
    let mut max_entries = 0;
    let mut fill = 0usize;
    for k in 0..matrix.n_eliminated() {
        let (degree, entries) = entries_of(k);
        if degree > 0 {
            max_star = max_star.max(entries);
        }
        if degree > 0 && entries > 1 {
            max_entries = max_entries.max(entries);
            fill = fill.saturating_add(sampler.max_fill_edges(entries));
        }
    }
    if ground_vertex.is_some() {
        let positive = matrix.surplus_kept().iter().filter(|&&s| s > 0.0).count();
        fill = fill.saturating_add(positive);
    }

    let sampler_bytes =
        sampler
            .reservation_bytes(max_entries)
            .ok_or(BuildError::SetupAllocationFailed {
                context: SAMPLED,
                bytes: usize::MAX,
            })?;
    ledger.reserve_external(SAMPLED, sampler_bytes, || sampler.try_reserve(max_entries))?;
    let retained = sampler
        .retained_heap_bytes()
        .ok_or(BuildError::SetupAllocationFailed {
            context: SAMPLED,
            bytes: usize::MAX,
        })?;
    // A fresh sampler owns nothing, so this charges only allocator over-delivery.
    ledger.charge(SAMPLED, retained - sampler_bytes)?;

    let mut entries: Vec<(u32, f64)> = ledger.with_capacity(SAMPLED, max_star)?;
    let entries_capacity = entries.capacity();
    let mut edges: Vec<Edge> = ledger.with_capacity(SAMPLED, fill)?;
    let edges_capacity = edges.capacity();
    for (k, &surplus) in surplus_eliminated.iter().enumerate() {
        let star = star(matrix, k);
        if star.degree() == 0 {
            continue;
        }
        entries.clear();
        for (&col, &w) in star.col_indices.iter().zip(star.weights) {
            entries.push((col, w));
        }
        if let Some(ground) = ground_vertex {
            if surplus > 0.0 {
                entries.push((ground, surplus));
            }
        }
        if entries.len() <= 1 {
            continue;
        }
        sampler.sample(star.index as u64, &entries, &mut edges);
    }
    if let Some(ground) = ground_vertex {
        edges.extend(
            matrix
                .surplus_kept()
                .iter()
                .enumerate()
                .filter(|(_, surplus)| **surplus > 0.0)
                .map(|(i, &surplus)| (i as u32, ground, surplus)),
        );
    }
    debug_assert_eq!(
        edges.capacity(),
        edges_capacity,
        "fill exceeded its reservation"
    );
    debug_assert_eq!(
        entries.capacity(),
        entries_capacity,
        "a star exceeded its reservation"
    );
    drop(sampler);
    ledger.release(retained);
    ledger.free(entries);
    sort_and_dedup(&mut edges, n_kept, ledger)?;
    Ok(edges)
}

/// [`super::sort_and_dedup`] serially; the total order makes it agree with the
/// parallel sorts.
fn sort_and_dedup(
    edges: &mut Vec<Edge>,
    n_kept: usize,
    ledger: &mut Ledger,
) -> Result<(), BuildError> {
    if edges.len() <= 1 {
        return Ok(());
    }
    if edges.len() >= n_kept {
        counting_sort_by_lo(edges, n_kept, ledger)?;
        let by_hi_weight = |a: &Edge, b: &Edge| a.1.cmp(&b.1).then_with(|| a.2.total_cmp(&b.2));
        edges
            .chunk_by_mut(|a, b| a.0 == b.0)
            .for_each(|run| run.sort_unstable_by(by_hi_weight));
    } else {
        edges.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.cmp(&b.1))
                .then_with(|| a.2.total_cmp(&b.2))
        });
    }
    let mut write = 0;
    for read in 1..edges.len() {
        if edges[write].0 == edges[read].0 && edges[write].1 == edges[read].1 {
            edges[write].2 += edges[read].2;
        } else {
            write += 1;
            edges[write] = edges[read];
        }
    }
    edges.truncate(write + 1);
    Ok(())
}

/// [`super::counting_sort_by_lo`], checked.
fn counting_sort_by_lo(
    edges: &mut Vec<Edge>,
    n_kept: usize,
    ledger: &mut Ledger,
) -> Result<(), BuildError> {
    let mut cursors = ledger.filled(SAMPLED, n_kept + 1, 0usize)?;
    for e in edges.iter() {
        cursors[e.0 as usize + 1] += 1;
    }
    for i in 1..cursors.len() {
        cursors[i] += cursors[i - 1];
    }
    let mut out = ledger.filled(SAMPLED, edges.len(), (0u32, 0u32, 0.0f64))?;
    for &e in edges.iter() {
        let cursor = &mut cursors[e.0 as usize];
        out[*cursor] = e;
        *cursor += 1;
    }
    ledger.free(cursors);
    let old = std::mem::replace(edges, out);
    ledger.free(old);
    Ok(())
}

/// [`super::build_laplacian_csr`], checked.
fn build_laplacian_csr(
    edges: &[Edge],
    n: usize,
    ledger: &mut Ledger,
) -> Result<CsrMatrix, BuildError> {
    let mut lower_count = ledger.filled(SAMPLED, n, 0u32)?;
    let mut upper_count = ledger.filled(SAMPLED, n, 0u32)?;
    let mut diag = ledger.filled(SAMPLED, n, 0.0)?;
    for &(lo, hi, w) in edges {
        diag[lo as usize] += w;
        upper_count[lo as usize] += 1;
        lower_count[hi as usize] += 1;
        diag[hi as usize] += w;
    }

    let total_nnz = offset_count(n + 2 * edges.len())?;
    let mut offsets = ledger.filled(SAMPLED, n + 1, 0u32)?;
    for i in 0..n {
        offsets[i + 1] = offsets[i] + lower_count[i] + 1 + upper_count[i];
    }
    debug_assert_eq!(offsets[n] as usize, total_nnz);
    let mut indices = ledger.filled(SAMPLED, total_nnz, 0u32)?;
    let mut data = ledger.filled(SAMPLED, total_nnz, 0.0f64)?;

    let mut lower_cursor = ledger.with_capacity(SAMPLED, n)?;
    lower_cursor.extend((0..n).map(|i| offsets[i]));
    let mut upper_cursor = ledger.with_capacity(SAMPLED, n)?;
    upper_cursor.extend((0..n).map(|i| offsets[i] + lower_count[i] + 1));
    for i in 0..n {
        let pos = (offsets[i] + lower_count[i]) as usize;
        indices[pos] = to_u32(i);
        data[pos] = diag[i];
    }

    for &(lo, hi, w) in edges {
        let lo_idx = lo as usize;
        let hi_idx = hi as usize;
        let pos = upper_cursor[lo_idx] as usize;
        indices[pos] = hi;
        data[pos] = -w;
        upper_cursor[lo_idx] += 1;
        let pos = lower_cursor[hi_idx] as usize;
        indices[pos] = lo;
        data[pos] = -w;
        lower_cursor[hi_idx] += 1;
    }
    ledger.free(lower_count);
    ledger.free(upper_count);
    ledger.free(diag);
    ledger.free(lower_cursor);
    ledger.free(upper_cursor);

    Ok(CsrMatrix::new(offsets, indices, data, n))
}
