//! Fused modified-LSMR step: `u ← A x + c·u`, `‖u‖²` and `y = Aᵀ u` in one
//! pass over the observations.
//!
//! Rows are split into fixed lanes that depend only on the design, never on
//! the worker count. Each lane walks its rows in blocks: it gathers `A x` for
//! the block with the separate path's per-row arithmetic, updates `u`, adds
//! the block's squared norm partial and accumulates the adjoint while the
//! block is cache resident.
//!
//! Adjoint accumulation, by term:
//! - sorted intercept terms: each level's run is summed in row order from
//!   `0.0`; runs inside a lane are written directly, and runs crossing lane
//!   boundaries are combined from their lane partials in lane order;
//! - small unsorted intercept terms: within a block, consecutive rows of one
//!   level are summed in a register and added to the lane's accumulator
//!   (`y` itself for lane 0, a lane buffer otherwise); buffers are added to
//!   `y` in lane order;
//! - remaining terms (slopes, or unsorted terms too large to buffer) are
//!   summed per coefficient in row order through a shared ownership index
//!   after the pass ([`StableAdjoint`]); no adjoint uses atomics.
//!
//! [`fused_adjoint`] applies the same accumulation to a plain `Aᵀ u`, so every
//! adjoint of a fused operator follows one fixed order.
//!
//! `u` and `‖u‖²` are bitwise the separate path's values. The adjoint sums
//! are fixed by the lanes and the ownership index, so results repeat exactly
//! and do not depend on the worker count; they may differ from the separate
//! adjoint in the last bits (a level whose rows span lanes is summed lane by
//! lane).

use rayon::prelude::*;
use schwarz_precond::{lsmr_norm_chunk, lsmr_norm_reduce, SolveError};

use super::gather::gather_rows;
use super::{StableAdjoint, PAR_THRESHOLD};
use crate::domain::{Design, Loading};

/// Fewest rows in a lane.
const LANE_MIN_ROWS: usize = 16_384;
/// Rows gathered, updated and scattered together inside a lane (one norm chunk).
const BLOCK_ROWS: usize = 4_096;
/// A lane spans at least this many rows per buffered coefficient, so all lane
/// buffers together hold at most a quarter of the observation count.
const ROWS_PER_BUFFERED_COEFFICIENT: usize = 4;
/// Buffered coefficients stay within `n_obs / BUFFER_SHARE` (or
/// [`MIN_BUFFERED`]), which keeps at least eight lanes when buffering.
const BUFFER_SHARE: usize = 32;
const MIN_BUFFERED: usize = 1_024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Sorted,
    Buffered { offset: usize },
    Separate,
}

/// Lane layout and per-term routes; a function of the design alone.
#[derive(Debug)]
pub(crate) struct FusedPlan {
    routes: Vec<Route>,
    lane_rows: usize,
    lanes: usize,
    norm_chunk: usize,
    buffered: usize,
}

impl FusedPlan {
    pub(crate) fn new(design: &Design<'_>) -> Self {
        let n = design.n_obs;
        let cap = (n / BUFFER_SHARE).max(MIN_BUFFERED);
        let mut buffered = 0;
        let routes = design
            .terms
            .iter()
            .map(|term| match &*term.columns {
                [Loading::Constant] if term.sorted => Route::Sorted,
                [Loading::Constant] if buffered + term.n_levels <= cap => {
                    let offset = buffered;
                    buffered += term.n_levels;
                    Route::Buffered { offset }
                }
                _ => Route::Separate,
            })
            .collect();
        let norm_chunk = lsmr_norm_chunk(n);
        // Each block must be one norm chunk, or part of the single sequential one.
        assert!(norm_chunk == BLOCK_ROWS || norm_chunk >= n);
        let lane_rows = if norm_chunk >= n {
            n.max(1)
        } else {
            LANE_MIN_ROWS
                .max(ROWS_PER_BUFFERED_COEFFICIENT * buffered)
                .div_ceil(norm_chunk)
                * norm_chunk
        };
        Self {
            routes,
            lane_rows,
            lanes: n.div_ceil(lane_rows).max(1),
            norm_chunk,
            buffered,
        }
    }

    /// Lane-buffer entries the step needs: lane 0 accumulates into `y` itself.
    pub(crate) fn buffer_len(&self) -> usize {
        (self.lanes - 1) * self.buffered
    }

    /// Whether term `q` is summed through the ownership index.
    pub(crate) fn is_separate(&self, q: usize) -> bool {
        self.routes[q] == Route::Separate
    }
}

/// A sorted term's lane partial for one level run.
#[derive(Clone, Copy)]
struct Edge {
    level: usize,
    sum: f64,
}

/// The first and (if different) last run of one sorted term in one lane.
#[derive(Clone, Copy, Default)]
struct LaneEdges {
    head: Option<Edge>,
    tail: Option<Edge>,
}

/// Mutable targets owned by one lane.
struct LaneTargets<'y> {
    /// Per term: a sorted term's interior levels `(first, slice)`, where
    /// `slice[k]` holds level `first + k`.
    interior: Vec<Option<(usize, &'y mut [f64])>>,
    /// Per term: a buffered term's accumulator.
    buffers: Vec<Option<&'y mut [f64]>>,
}

/// Rows `[start, end)` of lane `lane`.
fn lane_rows(plan: &FusedPlan, n: usize, lane: usize) -> (usize, usize) {
    let start = lane * plan.lane_rows;
    (start, (start + plan.lane_rows).min(n))
}

/// Split `y` and the lane buffers into disjoint per-lane targets.
fn carve<'y>(
    design: &Design<'_>,
    plan: &FusedPlan,
    y: &'y mut [f64],
    buffers: &'y mut [f64],
) -> Vec<LaneTargets<'y>> {
    let n = design.n_obs;
    let terms = design.terms.len();
    let mut targets: Vec<LaneTargets<'y>> = (0..plan.lanes)
        .map(|_| LaneTargets {
            interior: (0..terms).map(|_| None).collect(),
            buffers: (0..terms).map(|_| None).collect(),
        })
        .collect();
    // (start in y, length, lane, term, first interior level or None for a buffer)
    let mut pieces: Vec<(usize, usize, usize, usize, Option<usize>)> = Vec::new();
    for (q, (term, route)) in design.terms.iter().zip(&plan.routes).enumerate() {
        match route {
            Route::Sorted => {
                let levels = design.frame.level_column(q);
                for lane in 0..plan.lanes {
                    let (start, end) = lane_rows(plan, n, lane);
                    let first = levels[start] as usize + 1;
                    let last = levels[end - 1] as usize;
                    if last > first {
                        pieces.push((term.offset + first, last - first, lane, q, Some(first)));
                    }
                }
            }
            Route::Buffered { .. } => pieces.push((term.offset, term.n_levels, 0, q, None)),
            Route::Separate => {}
        }
    }
    pieces.sort_unstable_by_key(|piece| piece.0);
    let mut rest = y;
    let mut consumed = 0;
    for (start, len, lane, q, first) in pieces {
        let (_, tail) = std::mem::take(&mut rest).split_at_mut(start - consumed);
        let (piece, tail) = tail.split_at_mut(len);
        rest = tail;
        consumed = start + len;
        match first {
            Some(first) => targets[lane].interior[q] = Some((first, piece)),
            None => targets[lane].buffers[q] = Some(piece),
        }
    }
    if plan.buffered > 0 {
        for (lane, buffer) in buffers.chunks_mut(plan.buffered).enumerate() {
            let mut rest = buffer;
            for (q, route) in plan.routes.iter().enumerate() {
                if let Route::Buffered { .. } = route {
                    let (piece, tail) = rest.split_at_mut(design.terms[q].n_levels);
                    rest = tail;
                    targets[lane + 1].buffers[q] = Some(piece);
                }
            }
        }
    }
    targets
}

/// What a lane reads: the step's update of `u`, or a fixed `u`.
enum LaneRows<'u> {
    /// `u ← A x + c·u` with the lane's squared-norm partials.
    Update {
        x: &'u [f64],
        c: f64,
        u: &'u mut [f64],
        partials: &'u mut [f64],
    },
    /// A plain adjoint of `u`.
    Adjoint { u: &'u [f64] },
}

/// Run one lane starting at row `start`; returns each sorted term's edge partials.
fn run_lane(
    design: &Design<'_>,
    plan: &FusedPlan,
    sqrt_weights: Option<&[f64]>,
    start: usize,
    mut rows: LaneRows<'_>,
    mut targets: LaneTargets<'_>,
) -> Vec<LaneEdges> {
    let terms = design.terms.len();
    let len = match &rows {
        LaneRows::Update { u, .. } => u.len(),
        LaneRows::Adjoint { u } => u.len(),
    };
    // Open run per sorted term: (level, sum, is the lane's first run). Each
    // level's rows are summed in row order from `0.0`; the lane's first and
    // last runs become edge partials, every run in between is written in place.
    let mut edges = vec![LaneEdges::default(); terms];
    let mut runs: Vec<(usize, f64, bool)> = (0..terms)
        .map(|q| (design.frame.level_column(q)[start] as usize, 0.0, true))
        .collect();
    let mut base = vec![0.0; BLOCK_ROWS.min(len)];
    let mut sum_sq = 0.0;
    let mut offset = 0;
    while offset < len {
        let block = BLOCK_ROWS.min(len - offset);
        let first_row = start + offset;
        let base = &mut base[..block];
        match &mut rows {
            LaneRows::Update { x, c, u, partials } => {
                gather_rows(design, x, base, first_row, sqrt_weights);
                // The separate update `u = A x + c·u` and its sequential norm partial.
                for (b, ui) in base.iter_mut().zip(&mut u[offset..offset + block]) {
                    let value = *b + *c * *ui;
                    *ui = value;
                    sum_sq += value * value;
                    *b = value;
                }
                if (offset + block) % plan.norm_chunk == 0 || offset + block == len {
                    partials[(offset + block - 1) / plan.norm_chunk] = sum_sq;
                    sum_sq = 0.0;
                }
            }
            LaneRows::Adjoint { u } => base.copy_from_slice(&u[offset..offset + block]),
        }
        if let Some(sw) = sqrt_weights {
            for (b, &w) in base.iter_mut().zip(&sw[first_row..first_row + block]) {
                *b *= w;
            }
        }
        offset += block;
        for (q, route) in plan.routes.iter().enumerate() {
            let levels = &design.frame.level_column(q)[first_row..first_row + block];
            match route {
                Route::Buffered { .. } => {
                    let buffer = targets.buffers[q].as_deref_mut().expect("lane buffer");
                    // Repeated levels are summed in a register first, so runs do
                    // not chain read-modify-writes through one buffer slot.
                    let mut level = levels[0];
                    let mut sum = 0.0;
                    for (&next, &b) in levels.iter().zip(base.iter()) {
                        if next != level {
                            buffer[level as usize] += sum;
                            level = next;
                            sum = 0.0;
                        }
                        sum += b;
                    }
                    buffer[level as usize] += sum;
                }
                Route::Sorted => {
                    let (level, sum, first) = &mut runs[q];
                    for (&next, &b) in levels.iter().zip(base.iter()) {
                        let next = next as usize;
                        if next != *level {
                            if *first {
                                edges[q].head = Some(Edge {
                                    level: *level,
                                    sum: *sum,
                                });
                                *first = false;
                            } else {
                                let (lowest, interior) =
                                    targets.interior[q].as_mut().expect("interior levels");
                                interior[*level - *lowest] = *sum;
                            }
                            *level = next;
                            *sum = 0.0;
                        }
                        *sum += b;
                    }
                }
                Route::Separate => {}
            }
        }
    }
    for (q, route) in plan.routes.iter().enumerate() {
        if *route == Route::Sorted {
            let (level, sum, first) = runs[q];
            let edge = Some(Edge { level, sum });
            if first {
                edges[q].head = edge;
            } else {
                edges[q].tail = edge;
            }
        }
    }
    edges
}

fn allocation(context: &'static str) -> SolveError {
    SolveError::Allocation { context }
}

/// Zeroed lane buffers of the plan's length, reusing the operator's storage.
fn prepare_buffers(plan: &FusedPlan, buffers: &mut Vec<f64>) -> Result<(), SolveError> {
    let needed = plan.buffer_len();
    if buffers.len() != needed {
        buffers.clear();
        buffers
            .try_reserve_exact(needed)
            .map_err(|_| allocation("fused LSMR lane buffers"))?;
        buffers.resize(needed, 0.0);
    }
    buffers.fill(0.0);
    Ok(())
}

/// Combine lane results into `y` in lane order, then add the separate terms.
#[allow(clippy::too_many_arguments)]
fn finish(
    design: &Design<'_>,
    plan: &FusedPlan,
    buffers: &[f64],
    edges: &[Vec<LaneEdges>],
    owners: &StableAdjoint,
    sqrt_weights: Option<&[f64]>,
    u: &[f64],
    y: &mut [f64],
) {
    for (q, route) in plan.routes.iter().enumerate() {
        let term = &design.terms[q];
        match route {
            Route::Sorted => {
                // Lane edges in row order; equal neighbouring levels are one run.
                let mut pending: Option<Edge> = None;
                for lane in edges {
                    for edge in [lane[q].head, lane[q].tail].into_iter().flatten() {
                        pending = match pending {
                            Some(open) if open.level == edge.level => Some(Edge {
                                level: open.level,
                                sum: open.sum + edge.sum,
                            }),
                            Some(open) => {
                                y[term.offset + open.level] = open.sum;
                                Some(edge)
                            }
                            None => Some(edge),
                        };
                    }
                }
                if let Some(open) = pending {
                    y[term.offset + open.level] = open.sum;
                }
            }
            Route::Buffered { offset } => {
                let levels = term.n_levels;
                let block = &mut y[term.offset..term.offset + levels];
                let (lanes, buffered, offset) = (plan.lanes, plan.buffered, *offset);
                let merge = |(chunk, values): (usize, &mut [f64])| {
                    let first = chunk * BLOCK_ROWS;
                    for lane in 1..lanes {
                        let source = &buffers[(lane - 1) * buffered + offset + first..];
                        for (value, &partial) in values.iter_mut().zip(source) {
                            *value += partial;
                        }
                    }
                };
                if levels * lanes > PAR_THRESHOLD {
                    block.par_chunks_mut(BLOCK_ROWS).enumerate().for_each(merge);
                } else {
                    block.chunks_mut(BLOCK_ROWS).enumerate().for_each(merge);
                }
            }
            Route::Separate => {
                let block = &mut y[term.offset..term.offset + term.n_dofs()];
                owners.apply_term(design, q, sqrt_weights, u, block);
            }
        }
    }
}

/// Lane work items paired with their targets, run in parallel above the threshold.
fn run_lanes<'u>(
    design: &Design<'_>,
    plan: &FusedPlan,
    sqrt_weights: Option<&[f64]>,
    lanes: Vec<(LaneRows<'u>, LaneTargets<'u>)>,
) -> Vec<Vec<LaneEdges>> {
    let lane = |(index, (rows, targets)): (usize, (LaneRows<'u>, LaneTargets<'u>))| {
        run_lane(
            design,
            plan,
            sqrt_weights,
            index * plan.lane_rows,
            rows,
            targets,
        )
    };
    if design.n_obs > PAR_THRESHOLD {
        lanes.into_par_iter().enumerate().map(lane).collect()
    } else {
        lanes.into_iter().enumerate().map(lane).collect()
    }
}

/// One fused step: `u ← A x + c·u`, `y = Aᵀ u`; returns `‖u‖²`. `owners`
/// must index every [`Route::Separate`] term.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_update(
    design: &Design<'_>,
    plan: &FusedPlan,
    buffers: &mut Vec<f64>,
    owners: &StableAdjoint,
    sqrt_weights: Option<&[f64]>,
    x: &[f64],
    c: f64,
    u: &mut [f64],
    y: &mut [f64],
) -> Result<f64, SolveError> {
    let n = design.n_obs;
    debug_assert_eq!(u.len(), n);
    debug_assert_eq!(x.len(), design.n_dofs);
    debug_assert_eq!(y.len(), design.n_dofs);
    prepare_buffers(plan, buffers)?;
    let chunks = n.div_ceil(plan.norm_chunk);
    let mut partials = Vec::new();
    partials
        .try_reserve_exact(chunks)
        .map_err(|_| allocation("fused LSMR norm partials"))?;
    partials.resize(chunks, 0.0);

    y.fill(0.0);
    let edges = {
        let targets = carve(design, plan, y, buffers);
        let lanes = u
            .chunks_mut(plan.lane_rows)
            .zip(partials.chunks_mut(plan.lane_rows / plan.norm_chunk))
            .map(|(u, partials)| LaneRows::Update { x, c, u, partials })
            .zip(targets)
            .collect();
        run_lanes(design, plan, sqrt_weights, lanes)
    };
    finish(design, plan, buffers, &edges, owners, sqrt_weights, u, y);
    Ok(lsmr_norm_reduce(&partials))
}

/// `y = Aᵀ u` in the accumulation order of [`fused_update`].
pub(crate) fn fused_adjoint(
    design: &Design<'_>,
    plan: &FusedPlan,
    buffers: &mut Vec<f64>,
    owners: &StableAdjoint,
    sqrt_weights: Option<&[f64]>,
    u: &[f64],
    y: &mut [f64],
) -> Result<(), SolveError> {
    debug_assert_eq!(u.len(), design.n_obs);
    debug_assert_eq!(y.len(), design.n_dofs);
    prepare_buffers(plan, buffers)?;
    y.fill(0.0);
    let edges = {
        let targets = carve(design, plan, y, buffers);
        let lanes = u
            .chunks(plan.lane_rows)
            .map(|u| LaneRows::Adjoint { u })
            .zip(targets)
            .collect();
        run_lanes(design, plan, sqrt_weights, lanes)
    };
    finish(design, plan, buffers, &edges, owners, sqrt_weights, u, y);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Effect;
    use crate::operator::design::gather::gather_apply;

    struct Case {
        design: Design<'static>,
        weights: Option<Vec<f64>>,
    }

    fn leak<T>(values: Vec<T>) -> &'static [T] {
        Box::leak(values.into_boxed_slice())
    }

    /// Sorted, buffered and separate routes, including a run that spans
    /// several lanes, empty levels, a slope term and zero weights.
    fn cases() -> Vec<Case> {
        let mut out = Vec::new();
        for rows in [19usize, 9_999, 12_289, 70_001] {
            let worker: Vec<u32> = (0..rows)
                .map(|i| if i < rows / 2 { 0 } else { (3 * i / 5) as u32 })
                .collect();
            let firm: Vec<u32> = (0..rows).map(|i| ((i * 7_919) % 301) as u32).collect();
            let wide: Vec<u32> = (0..rows).map(|i| ((i * 104_729) % 5_003) as u32).collect();
            let small: Vec<u32> = (0..rows).map(|i| ((i * 31) % 32) as u32 * 2).collect();
            let slope: Vec<f64> = (0..rows).map(|i| (i % 17) as f64 * 0.1 - 0.7).collect();
            let (worker, firm, wide, small, slope) = (
                leak(worker),
                leak(firm),
                leak(wide),
                leak(small),
                leak(slope),
            );
            let none = std::iter::empty::<&[f64]>;
            let design = Design::new(vec![
                Effect::new(worker, true, none()).unwrap(),
                Effect::new(firm, true, none()).unwrap(),
                Effect::new(wide, true, none()).unwrap(),
                Effect::new(small, true, [slope]).unwrap(),
            ])
            .unwrap();
            let weights: Vec<f64> = (0..rows)
                .map(|i| {
                    if i % 13 == 0 {
                        0.0
                    } else {
                        0.5 + (i % 7) as f64
                    }
                })
                .collect();
            out.push(Case {
                design: design.clone(),
                weights: None,
            });
            out.push(Case {
                design,
                weights: Some(weights),
            });
        }
        // Three sorted levels over three lanes: the first lane holds two
        // adjacent levels and no interior, the last a single run.
        let rows = 40_000usize;
        let first = leak((0..rows).map(|i| (3 * i / rows) as u32).collect());
        let second = leak((0..rows).map(|i| (i % 2) as u32).collect());
        let none = std::iter::empty::<&[f64]>;
        out.push(Case {
            design: Design::new(vec![
                Effect::new(first, true, none()).unwrap(),
                Effect::new(second, true, none()).unwrap(),
            ])
            .unwrap(),
            weights: None,
        });
        out
    }

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    #[test]
    fn plan_routes_terms_and_bounds_lane_buffers() {
        for case in cases() {
            let design = &case.design;
            let plan = FusedPlan::new(design);
            let n = design.n_obs;
            let mut buffered = 0;
            for (q, term) in design.terms.iter().enumerate() {
                let intercept = matches!(&*term.columns, [Loading::Constant]);
                match plan.routes[q] {
                    Route::Sorted => assert!(intercept && term.sorted),
                    Route::Buffered { offset } => {
                        assert!(intercept && !term.sorted && offset == buffered);
                        buffered += term.n_levels;
                    }
                    Route::Separate => assert!(
                        !intercept
                            || buffered + term.n_levels > (n / BUFFER_SHARE).max(MIN_BUFFERED)
                    ),
                }
            }
            if design.terms.len() == 4 {
                assert!(plan.is_separate(3), "slope terms use the ownership index");
            }
            if n == 70_001 {
                // A dominant sorted term with one run across several lanes.
                assert_eq!(plan.routes[0], Route::Sorted);
                assert!(plan.lanes > 2 && plan.is_separate(2));
            }
            assert!(plan.buffer_len() <= n / 4);
            assert_eq!(plan.lanes == 1, n < 16_384);
            if n >= 10_000 {
                assert_eq!(plan.lane_rows % 4_096, 0);
            }
        }
    }

    #[test]
    fn fused_step_matches_separate_update_and_witness_at_every_worker_count() {
        for case in cases() {
            let design = &case.design;
            let n = design.n_obs;
            let sw: Option<Vec<f64>> = case
                .weights
                .as_ref()
                .map(|w| w.iter().map(|v| v.sqrt()).collect());
            let sw = sw.as_deref();
            let plan = FusedPlan::new(design);
            let owners = StableAdjoint::for_terms(design, |q| plan.is_separate(q)).unwrap();
            let x: Vec<f64> = (0..design.n_dofs)
                .map(|j| ((j * 37) % 101) as f64 / 23.0 - 2.0)
                .collect();
            let u0: Vec<f64> = (0..n)
                .map(|i| ((i * 53) % 97) as f64 / 31.0 - 1.5)
                .collect();
            let c = -0.61;

            // Separate path: gather, update, norm partials, row-major adjoint witness.
            let mut ax = vec![0.0; n];
            gather_apply(design, &x, &mut ax, sw);
            let expected_u: Vec<f64> = ax.iter().zip(&u0).map(|(a, u)| a + c * u).collect();
            let partials: Vec<f64> = expected_u
                .chunks(lsmr_norm_chunk(n))
                .map(|chunk| {
                    let mut s = 0.0;
                    for v in chunk {
                        s += v * v;
                    }
                    s
                })
                .collect();
            let expected_norm = lsmr_norm_reduce(&partials);
            let mut witness = vec![0.0; design.n_dofs];
            for i in 0..n {
                let base = sw.map_or(expected_u[i], |sw| sw[i] * expected_u[i]);
                for (q, term) in design.terms.iter().enumerate() {
                    let level = design.frame.level_column(q)[i] as usize;
                    for (column, loading) in term.columns.iter().enumerate() {
                        let value = match loading {
                            Loading::Constant => base,
                            Loading::Covariate(k) => {
                                design.frame.loading_column(*k as usize)[i] * base
                            }
                        };
                        witness[term.column_base(column) + level] += value;
                    }
                }
            }

            let mut reference: Option<Vec<u64>> = None;
            for threads in [1, 2, 4, 8, 8] {
                pool(threads).install(|| {
                    let mut buffers = Vec::new();
                    let mut u = u0.clone();
                    let mut y = vec![f64::NAN; design.n_dofs];
                    let norm = fused_update(
                        design,
                        &plan,
                        &mut buffers,
                        &owners,
                        sw,
                        &x,
                        c,
                        &mut u,
                        &mut y,
                    )
                    .unwrap();
                    assert_eq!(u, expected_u, "n={n}: u must be the separate update");
                    assert_eq!(norm.to_bits(), expected_norm.to_bits(), "n={n}: norm bits");
                    let scale = witness.iter().fold(1.0f64, |m, v| m.max(v.abs()));
                    for (j, (a, b)) in y.iter().zip(&witness).enumerate() {
                        assert!(
                            (a - b).abs() <= 1e-12 * scale,
                            "n={n} coefficient {j}: {a} vs {b}"
                        );
                    }
                    // A plain adjoint of the updated `u` follows the same order.
                    let mut plain = vec![f64::NAN; design.n_dofs];
                    fused_adjoint(design, &plan, &mut buffers, &owners, sw, &u, &mut plain)
                        .unwrap();
                    assert_eq!(
                        plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    let bits: Vec<u64> = y.iter().map(|v| v.to_bits()).collect();
                    match &reference {
                        None => reference = Some(bits),
                        Some(expected) => {
                            assert_eq!(&bits, expected, "n={n} differs at {threads} workers")
                        }
                    }
                });
            }
        }
    }
}
