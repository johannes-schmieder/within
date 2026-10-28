//! [`SchwarzPreconditioner`]: each DOF sums its subdomains' local outputs in subdomain order.

use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::SolveError;
use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::Operator;

use super::buffers::Pool;
use super::prolongation::Prolongation;

/// Persists `n_dofs`, which entries alone cannot recover when some columns are uncovered.
#[cfg(feature = "serde")]
impl<S> serde::Serialize for SchwarzPreconditioner<S>
where
    S: LocalSolver + serde::Serialize,
{
    fn serialize<Ser: serde::Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("SchwarzPreconditioner", 2)?;
        state.serialize_field("subdomains", &**self.subdomains)?;
        state.serialize_field("n_dofs", &self.n_dofs())?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, S> serde::Deserialize<'de> for SchwarzPreconditioner<S>
where
    S: LocalSolver + serde::de::DeserializeOwned,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::Deserialize;

        #[derive(Deserialize)]
        #[serde(bound(deserialize = "S: serde::de::DeserializeOwned"))]
        struct Helper<S: LocalSolver> {
            subdomains: Vec<SubdomainEntry<S>>,
            n_dofs: usize,
        }

        let h: Helper<S> = Helper::deserialize(deserializer)?;
        // An `n_dofs` below the covered span would put a subdomain DOF outside the prolongation.
        let covered_span = covered_span(&h.subdomains);
        if h.n_dofs < covered_span {
            return Err(serde::de::Error::custom(format!(
                "n_dofs ({}) is below the covered subdomain index span ({covered_span})",
                h.n_dofs
            )));
        }
        Ok(SchwarzPreconditioner::with_n_dofs(h.subdomains, h.n_dofs))
    }
}

/// One-level additive Schwarz preconditioner; `apply` reuses buffers pooled across clones.
pub struct SchwarzPreconditioner<S: LocalSolver> {
    subdomains: Arc<Vec<SubdomainEntry<S>>>,
    prolongation: Arc<Prolongation>,
    /// One per concurrent apply; every apply rewrites all slots, so a failure leaves nothing stale.
    local_outputs: Arc<Pool>,
    /// One per running span of subdomains.
    scratch: Arc<Pool>,
    outer_capacity: f64,
}

impl<S: LocalSolver> SchwarzPreconditioner<S> {
    const MIN_INNER_PARALLEL_WORK: usize = 200_000;
    const OUTER_CAPACITY_TARGET: f64 = 0.75;

    /// Infers `n_dofs` from the maximum global index; only valid when every DOF is covered.
    pub fn new(entries: Vec<SubdomainEntry<S>>) -> Self {
        Self::build(entries, None)
    }

    /// Explicit global DOF count; an uncovered column stays in the null space and applies to `0`.
    pub fn with_n_dofs(entries: Vec<SubdomainEntry<S>>, n_dofs: usize) -> Self {
        Self::build(entries, Some(n_dofs))
    }

    fn build(entries: Vec<SubdomainEntry<S>>, n_dofs: Option<usize>) -> Self {
        let n_dofs = n_dofs.unwrap_or_else(|| covered_span(&entries));
        debug_assert!(
            n_dofs >= covered_span(&entries),
            "n_dofs ({n_dofs}) is below the covered index span"
        );
        let prolongation = Prolongation::new(entries.iter().map(|e| e.global_indices()), n_dofs);
        let scratch_len = entries.iter().map(|e| e.scratch_len()).max().unwrap_or(0);
        Self {
            local_outputs: Arc::new(Pool::new(prolongation.n_slots())),
            scratch: Arc::new(Pool::new(scratch_len)),
            outer_capacity: Self::outer_capacity(&entries),
            subdomains: Arc::new(entries),
            prolongation: Arc::new(prolongation),
        }
    }

    /// Total over largest subdomain work; infinite when no subdomain merits inner parallelism.
    fn outer_capacity(entries: &[SubdomainEntry<S>]) -> f64 {
        let work = || {
            entries
                .iter()
                .map(|e| e.solver().inner_parallelism_work_estimate())
        };
        let largest = work().max().unwrap_or(0);
        if largest < Self::MIN_INNER_PARALLEL_WORK {
            return f64::INFINITY;
        }
        let total = work().fold(0usize, usize::saturating_add);
        total as f64 / largest as f64
    }

    /// Access the underlying subdomain entries.
    pub fn subdomains(&self) -> &[SubdomainEntry<S>] {
        &self.subdomains
    }

    fn n_dofs(&self) -> usize {
        self.prolongation.n_dofs()
    }

    /// Operator apply that propagates local-solver failures.
    pub fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), SolveError> {
        self.check_len("r", r.len())?;
        self.check_len("z", z.len())?;
        let sweep = Sweep {
            preconditioner: self,
            r,
            allow_inner_parallelism: self.allow_inner_parallelism(),
        };
        self.local_outputs.lend(|outputs| {
            sweep.solve_all(outputs)?;
            self.prolongation.apply(outputs, z);
            Ok(())
        })
    }

    fn check_len(&self, name: &str, len: usize) -> Result<(), SolveError> {
        let n_dofs = self.n_dofs();
        if len == n_dofs {
            return Ok(());
        }
        Err(SolveError::InvalidInput {
            context: "SchwarzPreconditioner::apply",
            message: format!("{name}.len() ({len}) != n_dofs ({n_dofs})"),
        })
    }

    /// Nested Rayon in local solves, unless the subdomains alone already fill the pool.
    fn allow_inner_parallelism(&self) -> bool {
        let threads = rayon::current_num_threads().max(1);
        self.outer_capacity < threads as f64 * Self::OUTER_CAPACITY_TARGET
    }
}

/// One past the largest DOF any subdomain covers.
fn covered_span<S: LocalSolver>(entries: &[SubdomainEntry<S>]) -> usize {
    entries
        .iter()
        .flat_map(|e| e.global_indices().iter().copied())
        .max()
        .map_or(0, |m| m as usize + 1)
}

/// Consecutive subdomains `ids` and `outputs`, exactly their slots of the local-output buffer.
struct Span<'a> {
    ids: Range<usize>,
    outputs: &'a mut [f64],
}

impl Span<'_> {
    fn halve(self, p: &Prolongation) -> (Self, Option<Self>) {
        if self.ids.len() < 2 {
            return (self, None);
        }
        let mid = self.ids.start + self.ids.len() / 2;
        let (left, right) = self
            .outputs
            .split_at_mut(p.first_slot(mid) - p.first_slot(self.ids.start));
        let left = Span {
            ids: self.ids.start..mid,
            outputs: left,
        };
        let right = Span {
            ids: mid..self.ids.end,
            outputs: right,
        };
        (left, Some(right))
    }

    /// The slots of subdomain `i`, one of `ids`.
    fn outputs_of(&mut self, i: usize, p: &Prolongation) -> &mut [f64] {
        let base = p.first_slot(self.ids.start);
        &mut self.outputs[p.first_slot(i) - base..p.first_slot(i + 1) - base]
    }
}

/// One apply's fixed inputs.
struct Sweep<'a, S: LocalSolver> {
    preconditioner: &'a SchwarzPreconditioner<S>,
    r: &'a [f64],
    allow_inner_parallelism: bool,
}

impl<S: LocalSolver> Sweep<'_, S> {
    /// Solves every subdomain into its slots of `outputs`, in parallel over halved spans.
    fn solve_all(&self, outputs: &mut [f64]) -> Result<(), SolveError> {
        let p = &*self.preconditioner.prolongation;
        let all = Span {
            ids: 0..self.preconditioner.subdomains.len(),
            outputs,
        };
        rayon::iter::split(all, |span| span.halve(p)).try_for_each(|span| {
            self.preconditioner
                .scratch
                .lend(|scratch| self.solve(span, scratch))
        })
    }

    fn solve(&self, mut span: Span<'_>, scratch: &mut [f64]) -> Result<(), SolveError> {
        let p = &*self.preconditioner.prolongation;
        for i in span.ids.clone() {
            self.preconditioner.subdomains[i]
                .apply_weighted_local(
                    self.r,
                    span.outputs_of(i, p),
                    scratch,
                    self.allow_inner_parallelism,
                )
                .map_err(|source| SolveError::LocalSolveFailed {
                    subdomain: i,
                    source,
                })?;
        }
        Ok(())
    }
}

impl<S: LocalSolver> Clone for SchwarzPreconditioner<S> {
    /// Clone shares subdomain data and the buffer pool via `Arc`: O(1) and fully interchangeable.
    fn clone(&self) -> Self {
        Self {
            subdomains: Arc::clone(&self.subdomains),
            prolongation: Arc::clone(&self.prolongation),
            local_outputs: Arc::clone(&self.local_outputs),
            scratch: Arc::clone(&self.scratch),
            outer_capacity: self.outer_capacity,
        }
    }
}

impl<S: LocalSolver> Operator for SchwarzPreconditioner<S> {
    fn nrows(&self) -> usize {
        self.n_dofs()
    }

    fn ncols(&self) -> usize {
        self.n_dofs()
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), SolveError> {
        SchwarzPreconditioner::apply(self, r, z)
    }

    fn apply_adjoint(&self, r: &[f64], z: &mut [f64]) -> Result<(), SolveError> {
        SchwarzPreconditioner::apply(self, r, z)
    }
}
