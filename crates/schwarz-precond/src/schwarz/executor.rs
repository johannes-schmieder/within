//! Additive Schwarz execution. Every subdomain writes its local output into its own slots of one
//! buffer; each DOF then sums its slots in subdomain order, so no thread count changes the bits.

use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::SolveError;
use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::prolongation::Prolongation;

use super::buffers::{LocalSolveScratch, WorkerBufferStack, Workspace, WorkspacePool};

/// Where each subdomain's local output lives, and which outputs each DOF sums.
struct SlotLayout {
    /// Subdomain `i` owns slots `starts[i]..starts[i + 1]`.
    starts: Vec<usize>,
    /// Sums each DOF's slots in subdomain order.
    prolongation: Prolongation,
}

pub(super) struct AdditiveExecutor<S: LocalSolver> {
    subdomains: Arc<Vec<SubdomainEntry<S>>>,
    layout: Arc<SlotLayout>,
    pool: WorkspacePool,
    max_scratch_size: usize,
}

impl<S: LocalSolver> AdditiveExecutor<S> {
    pub(super) fn new(
        subdomains: Arc<Vec<SubdomainEntry<S>>>,
        n_dofs: usize,
        max_scratch_size: usize,
    ) -> Self {
        let mut starts = Vec::with_capacity(subdomains.len() + 1);
        starts.push(0);
        for entry in subdomains.iter() {
            starts.push(starts[starts.len() - 1] + entry.global_indices().len());
        }
        let prolongation = Prolongation::new(
            subdomains
                .iter()
                .flat_map(|e| e.global_indices().iter().copied()),
            n_dofs,
        );
        let pool = WorkspacePool::new(starts[subdomains.len()]);
        Self {
            subdomains,
            layout: Arc::new(SlotLayout {
                starts,
                prolongation,
            }),
            pool,
            max_scratch_size,
        }
    }

    pub(super) fn subdomains(&self) -> &[SubdomainEntry<S>] {
        &self.subdomains
    }

    pub(super) fn n_dofs(&self) -> usize {
        self.layout.prolongation.n_dofs()
    }

    pub(super) fn apply(
        &self,
        allow_inner_parallelism: bool,
        r: &[f64],
        z: &mut [f64],
    ) -> Result<(), SolveError> {
        let mut workspace = self.pool.take()?;
        let result = self.apply_in(&mut workspace, allow_inner_parallelism, r, z);
        self.pool.put(workspace);
        result
    }

    fn apply_in(
        &self,
        workspace: &mut Workspace,
        allow_inner_parallelism: bool,
        r: &[f64],
        z: &mut [f64],
    ) -> Result<(), SolveError> {
        let worker_scratch = WorkerBufferStack::new(
            std::mem::take(&mut workspace.scratch),
            self.max_scratch_size,
        );
        let starts = &self.layout.starts;
        let all = (0..self.subdomains.len(), &mut workspace.local_outputs[..]);
        let solved = rayon::iter::split(all, |(ids, outputs)| {
            if ids.len() < 2 {
                return ((ids, outputs), None);
            }
            let mid = ids.start + ids.len() / 2;
            let (left, right) = outputs.split_at_mut(starts[mid] - starts[ids.start]);
            ((ids.start..mid, left), Some((mid..ids.end, right)))
        })
        .try_for_each(|(ids, outputs)| {
            worker_scratch.with_buffer(|scratch| {
                self.solve_span(ids, outputs, r, scratch, allow_inner_parallelism)
            })
        });

        // A pool-recovery failure must not mask a real `LocalSolveFailed`.
        match worker_scratch.into_pool() {
            Ok(recovered) => workspace.scratch = recovered,
            Err(into_err) => return solved.and(Err(into_err)),
        }
        solved?;

        self.layout.prolongation.apply(&workspace.local_outputs, z);
        Ok(())
    }

    /// Solves subdomains `ids` into `outputs`, their slots of the local-output buffer.
    fn solve_span(
        &self,
        ids: Range<usize>,
        outputs: &mut [f64],
        r: &[f64],
        scratch: &mut LocalSolveScratch,
        allow_inner_parallelism: bool,
    ) -> Result<(), SolveError> {
        let starts = &self.layout.starts;
        let base = starts[ids.start];
        for i in ids {
            self.subdomains[i]
                .apply_weighted_local(
                    r,
                    &mut outputs[starts[i] - base..starts[i + 1] - base],
                    &mut scratch.r_scratch,
                    &mut scratch.z_scratch,
                    allow_inner_parallelism,
                )
                .map_err(|source| SolveError::LocalSolveFailed {
                    subdomain: i,
                    source,
                })?;
        }
        Ok(())
    }
}

impl<S: LocalSolver> Clone for AdditiveExecutor<S> {
    fn clone(&self) -> Self {
        Self {
            subdomains: Arc::clone(&self.subdomains),
            layout: Arc::clone(&self.layout),
            pool: self.pool.clone(),
            max_scratch_size: self.max_scratch_size,
        }
    }
}
