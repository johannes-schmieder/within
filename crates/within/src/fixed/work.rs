//! Conservative logical-work admission for the optional backend path.
//!
//! approx-chol does not expose its factor entry counts. Its borrowed serializer
//! visits every permutation, elimination header/neighbor and dense coefficient.
//! Counting those bytes supplies a storage bound without copying the factor or
//! depending on private layouts. Each stored scalar produces at least one byte;
//! 32 units per byte covers both triangular sweeps, permutations and block gauges.

use crate::block_elim::BlockElimSolver;
use crate::operator::schwarz::Variant;
use crate::{BuildError, Solver};
use approx_chol::Factor;
use postcard::ser_flavors::Flavor;
use schwarz_precond::{LocalSolver, Operator, ReductionStrategy, SchwarzPreconditioner};

pub(crate) fn factor_work(
    factor: &Factor,
    checkpoint: &mut dyn FnMut(u64) -> Result<(), BuildError>,
) -> Result<Option<u64>, BuildError> {
    struct Counter<'a> {
        checkpoint: &'a mut dyn FnMut(u64) -> Result<(), BuildError>,
        error: &'a mut Option<BuildError>,
        bytes: u64,
        left: u8,
    }
    impl Flavor for Counter<'_> {
        type Output = u64;
        fn try_push(&mut self, _: u8) -> postcard::Result<()> {
            if self.left == 0 {
                if let Err(error) = (self.checkpoint)(512) {
                    *self.error = Some(error);
                    return Err(postcard::Error::SerializeBufferFull);
                }
                self.left = 16;
            }
            self.left -= 1;
            self.bytes = self
                .bytes
                .checked_add(1)
                .ok_or(postcard::Error::SerializeBufferFull)?;
            Ok(())
        }
        fn finalize(self) -> postcard::Result<u64> {
            Ok(self.bytes)
        }
    }
    let mut error = None;
    let result = postcard::serialize_with_flavor(
        factor,
        Counter {
            checkpoint,
            error: &mut error,
            bytes: 0,
            left: 0,
        },
    );
    if let Some(error) = error {
        return Err(error);
    }
    match result {
        Ok(bytes) => Ok(bytes.checked_mul(32)),
        Err(error) => Err(BuildError::LocalSolverBuild(format!(
            "factor work metadata: {error}"
        ))),
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ApplyWork {
    local: Option<u64>,
    max_scratch: u64,
    n_dofs: u64,
    entries: usize,
}
impl ApplyWork {
    pub(crate) fn new(n_dofs: usize, entries: usize) -> Self {
        Self {
            local: Some(0),
            max_scratch: 0,
            n_dofs: n_dofs as u64,
            entries,
        }
    }
    pub(crate) fn add(
        &mut self,
        solver: &BlockElimSolver,
        checkpoint: &mut dyn FnMut(u64) -> Result<(), BuildError>,
    ) -> Result<(), BuildError> {
        checkpoint(64)?;
        let bound = solver.apply_work_bound(checkpoint)?;
        self.local = self.local.and_then(|sum| sum.checked_add(bound?));
        self.max_scratch = self.max_scratch.max(solver.scratch_size() as u64);
        Ok(())
    }
    fn slots(self, strategy: ReductionStrategy) -> u64 {
        // Ordered execution fixes slot width. Other strategies may retain one
        // scratch buffer per nested/reentrant subdomain, so use the domain count.
        if strategy == ReductionStrategy::Ordered {
            rayon::current_num_threads().min(self.entries.max(1)) as u64
        } else {
            self.entries.max(1) as u64
        }
    }

    pub(crate) fn bound(self, strategy: ReductionStrategy) -> Option<u64> {
        // Includes first-use zeroed local/global buffers, restriction/prolongation,
        // final readout and worker reductions. Query under the application pool.
        self.local?.checked_add(
            self.n_dofs
                .checked_add(self.max_scratch)?
                .checked_mul(self.slots(strategy).checked_add(1)?)?
                .checked_mul(16)?,
        )
    }

    pub(crate) fn storage_bound(self, strategy: ReductionStrategy) -> Option<u64> {
        // Each bank has two local f64 buffers plus at most one global buffer per
        // slot. The extra global allowance covers incumbent factor permutation
        // temporaries (at most local dimension plus grounding auxiliaries).
        // Metadata includes vector headers, worker bookkeeping and spare capacity.
        // BufferPool retains at most four banks in addition to the active apply.
        let slots = self.slots(strategy).checked_add(1)?;
        let vectors = self
            .n_dofs
            .checked_add(self.max_scratch)?
            .checked_mul(16)?
            .checked_mul(slots)?;
        let metadata = slots
            .checked_mul(256)?
            .checked_add((rayon::current_num_threads() as u64).checked_mul(128)?)?;
        vectors.checked_add(metadata)?.checked_mul(5)
    }
}

pub(crate) fn schwarz_work(
    base: &SchwarzPreconditioner<BlockElimSolver>,
    checkpoint: &mut dyn FnMut(u64) -> Result<(), BuildError>,
) -> Result<ApplyWork, BuildError> {
    let mut work = ApplyWork::new(base.ncols(), base.subdomains().len());
    for entry in base.subdomains() {
        work.add(entry.solver(), checkpoint)?;
    }
    Ok(work)
}

impl Solver<'_> {
    /// Conservative logical operations for one full incumbent inverse-Gram
    /// action, including all factors, transforms and Schwarz reductions. Counts
    /// actual factor storage without allocation; checkpoints bound the scan.
    /// `None` means arithmetic overflow. Query in the same Rayon worker pool as
    /// the intended application. This does not include design-vector products.
    pub fn incumbent_apply_work_bound(
        &self,
        mut checkpoint: impl FnMut(u64) -> Result<(), BuildError>,
    ) -> Result<Option<u64>, BuildError> {
        checkpoint(64)?;
        Ok(match self.preconditioner() {
            Some(preconditioner) => match &preconditioner.inner {
                Variant::Additive(base) => schwarz_work(&base.inner, &mut checkpoint)?
                    .bound(base.inner.reduction_strategy()),
                Variant::Diagonal(_) => (self.n_dofs() as u64).checked_mul(4),
            },
            None => Some(self.n_dofs() as u64),
        })
    }

    /// Conservative logical operations for either one design action or one
    /// adjoint action, including observation weights and coefficient clears.
    /// Counts only term metadata, with bounded checkpoints and no allocations.
    pub fn design_apply_work_bound(
        &self,
        mut checkpoint: impl FnMut(u64) -> Result<(), BuildError>,
    ) -> Result<Option<u64>, BuildError> {
        let mut channels = Some(0u64);
        for term in &self.design.terms {
            checkpoint(32)?;
            channels = channels.and_then(|sum| sum.checked_add(term.columns.len() as u64));
        }
        Ok(channels
            .and_then(|channels| channels.checked_mul(self.n_obs() as u64))
            .and_then(|work| work.checked_add(self.n_dofs() as u64))
            .and_then(|work| work.checked_add(self.n_obs() as u64))
            .and_then(|work| work.checked_mul(16)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed::tests::ProjectedFactory;
    use crate::{Design, LocalSolverConfig, PreconditionerConfig, ReductionStrategy};

    #[test]
    fn bounds_count_real_incumbent_storage_and_candidate_metadata_under_checkpoints() {
        let solver = Solver::new(
            Design::from_levels_for_test(vec![
                (0..60).map(|i| i % 6).collect(),
                (0..60).map(|i| (i / 6) % 5).collect(),
                (0..60).map(|i| i % 4).collect(),
            ]),
            None,
            PreconditionerConfig::Additive {
                local_solver: LocalSolverConfig {
                    dense_threshold: 2,
                    ..Default::default()
                },
                reduction: ReductionStrategy::Ordered,
            },
        )
        .unwrap();
        let before = postcard::to_stdvec(solver.preconditioner().unwrap()).unwrap();
        let mut charged = 0;
        let bound = solver
            .incumbent_apply_work_bound(|units| {
                assert!(units <= 1024);
                charged += units;
                Ok(())
            })
            .unwrap()
            .unwrap();
        assert!(charged > 512, "actual factor entries were inspected");
        assert!(bound > (solver.n_dofs() * 64) as u64);
        let mut calls = 0;
        assert!(solver
            .incumbent_apply_work_bound(|_| {
                calls += 1;
                if calls == 4 {
                    Err(BuildError::LocalSolverBuild("scan cap".into()))
                } else {
                    Ok(())
                }
            })
            .is_err());
        assert_eq!(calls, 4);
        let design = solver
            .design_apply_work_bound(|units| {
                assert!(units <= 1024);
                Ok(())
            })
            .unwrap()
            .unwrap();
        assert_eq!(design, 16 * (60 * 3 + solver.n_dofs() + 60) as u64);
        let candidate = solver
            .build_pair_replacement(&ProjectedFactory)
            .unwrap()
            .unwrap();
        assert!(candidate.apply_work_bound().unwrap() > (solver.n_dofs() * 64) as u64);
        let storage = candidate.apply_storage_bound().unwrap();
        assert!(storage >= candidate.work.max_scratch * 16 * 5);
        let wider = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let narrow = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        assert!(
            wider.install(|| candidate.apply_storage_bound().unwrap())
                >= narrow.install(|| candidate.apply_storage_bound().unwrap())
        );
        assert_eq!(
            postcard::to_stdvec(solver.preconditioner().unwrap()).unwrap(),
            before
        );
    }
}
