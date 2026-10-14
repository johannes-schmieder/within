//! Optional runtime-only fixed reduced inverses for categorical pair components.

use std::sync::Arc;
use std::time::{Duration, Instant};

use schwarz_precond::{
    LocalSolveError, LocalSolver, PartitionWeights, SchwarzPreconditioner, SubdomainCore,
    SubdomainEntry,
};

use crate::block_elim::BlockElimSolver;
use crate::domain::Loading;
use crate::operator::schwarz::{FeSchwarz, Variant};
use crate::{BuildError, Channel, PreconditionerConfig, Solver};

pub(crate) mod work;

/// A fixed linear reduced inverse. Implementations must be symmetric positive
/// semidefinite with kernel exactly the constant vector, in the Euclidean
/// retained-coordinate metric. A fixed symmetric multigrid cycle is suitable;
/// tolerance-stopped or fixed-iteration CG is not a linear operator.
///
/// `within` supplies the full component's sign transform, projection, triangular
/// elimination, recovery and transpose projection. The resulting local action
/// retains the incumbent categorical component kernel. No global full-rank basis
/// or global rank test is required. In canonical Laplacian coordinates the
/// wrapper is `P G P`, where `P = I - 11ᵀ/n` and
/// `G = T diag(D_elim⁻¹, B) Tᵀ`. Although `G` generally has a different kernel
/// from the local Laplacian, `P G P` has exactly its constant kernel: zero energy
/// forces the eliminated part to zero and the retained part to a constant;
/// membership in `1`-perp then forces that constant to zero. Sign and partition
/// weights are applied as two-sided congruences, with the actual transpose on
/// the output. A backend using another metric must supply the required adjoint
/// transformations itself before returning the Euclidean reduced action.
pub trait FixedReducedInverse: Send + Sync {
    /// Number of retained categorical levels.
    fn dimension(&self) -> usize;
    /// Scratch elements required by an application.
    fn scratch_len(&self) -> usize;
    /// Conservative logical-operation bound for one application, including
    /// scratch initialization and the complete fixed cycle. `None` declines
    /// work-capped admission; it does not affect explicit applications.
    fn apply_work_bound(&self) -> Option<u64> {
        None
    }
    /// Apply the same fixed inverse on every call, replacing `rhs` with its image.
    fn apply_in_place(&self, rhs: &mut [f64], scratch: &mut [f64]) -> Result<(), LocalSolveError>;
}

/// Borrowed positive weighted bipartite component in eliminated-major CSR order.
/// Entries are cross-tab cell weights, not negative Laplacian off-diagonals.
/// The component is connected, floating and categorical; its diagonals are the
/// row/column sums. Indices address the retained block.
#[derive(Clone, Copy, Debug)]
pub struct PairComponent<'a> {
    /// Eliminated level count, at least the retained count.
    pub n_eliminated: usize,
    /// Retained level count.
    pub n_retained: usize,
    /// CSR row pointers, length `n_eliminated + 1`.
    pub row_ptr: &'a [u32],
    /// CSR retained-level indices.
    pub columns: &'a [u32],
    /// Positive cross-tab cell weights.
    pub weights: &'a [f64],
}

/// Builds a fixed reduced inverse; an error rejects the entire nominated pair.
/// The caller owns resource admission and any application-specific certificates.
pub trait FixedReducedInverseFactory {
    /// Bounded construction checkpoint, including nomination and metadata copies.
    /// `work_units` is a conservative primitive-work charge: 128 per metadata
    /// entry, 32 before allocation and bytes per copy chunk (at most 1024).
    /// Scheduler construction separately checks each 128-index chunk; factor
    /// work inspection checks each 16 encoded bytes (512 work units). The caller
    /// must admit the metadata memory envelope before starting construction.
    fn checkpoint(&self, _work_units: u64) -> Result<(), BuildError> {
        Ok(())
    }
    /// Build from the original aggregated component, once per nominated component.
    fn build(
        &self,
        component: PairComponent<'_>,
    ) -> Result<Arc<dyn FixedReducedInverse>, BuildError>;
}

/// Deterministic categorical pair selected for optional replacement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairNomination {
    /// Original factor indices, ascending.
    pub factors: (usize, usize),
    /// Largest retained component dimension (the primary score).
    pub max_retained: usize,
    /// Total retained dimension over this pair's components (the secondary score).
    pub total_retained: usize,
    /// Number of components strictly above the incumbent dense threshold.
    pub replacement_components: usize,
}

/// A transactional replacement tied to its originating solver. Other pair
/// components and small direct components retain their incumbent factors.
/// This handle is runtime-only; serializing its external inverse is unsupported.
#[derive(Clone)]
pub struct PairReplacement {
    pub(crate) source: Arc<()>,
    pub(crate) identity: Arc<()>,
    pub(crate) preconditioner: SchwarzPreconditioner<PairLocalSolver>,
    build_duration: Duration,
    nomination: PairNomination,
    work: work::ApplyWork,
}

impl PairReplacement {
    /// Time spent constructing all replacements and metadata.
    pub fn build_duration(&self) -> Duration {
        self.build_duration
    }
    /// Pair and component-size information used for nomination.
    pub fn nomination(&self) -> PairNomination {
        self.nomination
    }

    /// Conservative work for the entire replacement action, including unchanged
    /// local factors and Schwarz buffers/reductions. `None` means a backend did
    /// not declare a bound or arithmetic overflowed. Query under the same Rayon
    /// pool as application; all factor metadata was scanned during construction.
    pub fn apply_work_bound(&self) -> Option<u64> {
        self.work.bound(self.preconditioner.reduction_strategy())
    }

    /// Conservative bytes for one active Schwarz application and its cache of
    /// at most four pooled banks, including local/global buffers and incumbent
    /// permutation scratch. Immutable factors and backend-owned storage are
    /// excluded: the factory must admit them separately. Query under the intended
    /// Rayon pool; multiplying by the concurrent RHS count safely covers shared
    /// cache and active banks. `None` means size arithmetic overflow.
    pub fn apply_storage_bound(&self) -> Option<u64> {
        self.work
            .storage_bound(self.preconditioner.reduction_strategy())
    }

    /// Apply the complete replacement Schwarz action, including all unchanged pairs.
    pub fn apply(&self, rhs: &[f64], out: &mut [f64]) -> Result<(), crate::SolveError> {
        self.preconditioner.apply(rhs, out)
    }
}

/// Owned, runtime-only construction plan tied to one solver's topology and weights.
/// Preparing this plan does not replace any incumbent factors. Dropping a plan
/// or a failed build leaves the originating solver usable without a restart.
pub struct OwnedPairBuild {
    shared: Arc<SchwarzPreconditioner<BlockElimSolver>>,
    selected: Vec<usize>,
    nomination: PairNomination,
    n_dofs: usize,
    reduction: crate::ReductionStrategy,
    source: Arc<()>,
    started: Instant,
}

impl OwnedPairBuild {
    /// Pair and component dimensions fixed during preparation.
    pub fn nomination(&self) -> PairNomination {
        self.nomination
    }

    /// Construct all nominated components transactionally on the calling worker.
    /// Preparation plus construction has exactly the synchronous builder's
    /// checkpoint order. The caller admits factors and holds the factory's
    /// resource/cancellation state for this entire operation.
    pub fn build(
        self,
        factory: &dyn FixedReducedInverseFactory,
    ) -> Result<PairReplacement, BuildError> {
        factory.checkpoint(32)?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(self.shared.subdomains().len())
            .map_err(|_| BuildError::Allocation {
                context: "pair replacement entries",
            })?;
        let mut work = work::ApplyWork::new(self.n_dofs, self.shared.subdomains().len());
        let mut selected = self.selected.into_iter().peekable();
        for (index, entry) in self.shared.subdomains().iter().enumerate() {
            factory.checkpoint(128)?;
            let solver = if selected.peek() == Some(&index) {
                selected.next();
                let cross = entry
                    .solver()
                    .categorical_cross_tab()
                    .expect("nominated categorical component");
                let inverse = factory.build(PairComponent {
                    n_eliminated: cross.c.nrows,
                    n_retained: cross.c.ncols,
                    row_ptr: &cross.c.indptr,
                    columns: &cross.c.indices,
                    weights: &cross.c.data,
                })?;
                PairLocalSolver::Replacement(entry.solver().with_fixed_inverse(inverse, factory)?)
            } else {
                PairLocalSolver::Incumbent {
                    source: self.shared.clone(),
                    index,
                }
            };
            work.add(solver.inner(), &mut |units| factory.checkpoint(units))?;
            let globals = copy_admitted(entry.global_indices(), factory)?;
            let partition = match entry.partition_weights() {
                PartitionWeights::Uniform(n) => PartitionWeights::Uniform(*n),
                PartitionWeights::NonUniform(weights) => {
                    PartitionWeights::NonUniform(copy_admitted(weights, factory)?)
                }
            };
            let core = SubdomainCore::with_partition_weights(globals, partition)
                .map_err(BuildError::Preconditioner)?;
            entries
                .push(SubdomainEntry::try_new(core, solver).map_err(BuildError::Preconditioner)?);
        }
        let preconditioner =
            SchwarzPreconditioner::try_with_n_dofs(entries, self.n_dofs, self.reduction, |work| {
                factory.checkpoint(work)
            })?;
        Ok(PairReplacement {
            source: self.source,
            identity: Arc::new(()),
            preconditioner,
            build_duration: self.started.elapsed(),
            nomination: self.nomination,
            work,
        })
    }
}

/// Runtime-only local action; incumbent factors are shared, never cloned.
pub(crate) enum PairLocalSolver {
    Incumbent {
        source: Arc<SchwarzPreconditioner<BlockElimSolver>>,
        index: usize,
    },
    Replacement(BlockElimSolver),
}
impl PairLocalSolver {
    pub(crate) fn inner(&self) -> &BlockElimSolver {
        match self {
            Self::Incumbent { source, index } => source.subdomains()[*index].solver(),
            Self::Replacement(solver) => solver,
        }
    }
}
impl LocalSolver for PairLocalSolver {
    fn n_local(&self) -> usize {
        self.inner().n_local()
    }
    fn scratch_size(&self) -> usize {
        self.inner().scratch_size()
    }
    fn inner_parallelism_work_estimate(&self) -> usize {
        self.inner().inner_parallelism_work_estimate()
    }
    fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        parallel: bool,
    ) -> Result<(), LocalSolveError> {
        self.inner().solve_local(rhs, sol, parallel)
    }
}

pub(crate) fn copy_admitted<T: Copy>(
    values: &[T],
    factory: &dyn FixedReducedInverseFactory,
) -> Result<Vec<T>, BuildError> {
    factory.checkpoint(32)?;
    let mut copy = Vec::new();
    copy.try_reserve_exact(values.len())
        .map_err(|_| BuildError::Allocation {
            context: "pair replacement metadata",
        })?;
    for chunk in values.chunks(128) {
        factory.checkpoint(std::mem::size_of_val(chunk) as u64)?;
        copy.extend_from_slice(chunk);
    }
    Ok(copy)
}

type NominatedEntries<'a> = (&'a FeSchwarz, PairNomination, Vec<usize>);

impl Solver<'_> {
    /// Nominate the categorical pair with largest `max_c min(L_ac, L_bc)`;
    /// ties prefer larger total retained dimension, then smaller original factor indices.
    /// Components at or below the existing dense threshold are never replaced.
    pub fn pair_nomination(&self) -> Option<PairNomination> {
        self.pair_entries(None)
            .ok()
            .flatten()
            .map(|(_, nomination, _)| nomination)
    }

    /// Build every eligible component of the nominated pair transactionally.
    /// On failure the incumbent solver is unchanged and remains usable.
    pub fn build_pair_replacement(
        &self,
        factory: &dyn FixedReducedInverseFactory,
    ) -> Result<Option<PairReplacement>, BuildError> {
        self.prepare_pair_replacement(factory)?
            .map(|plan| plan.build(factory))
            .transpose()
    }

    /// Prepare an owned nominated-pair plan. The returned plan shares immutable
    /// factors/topology and the exact solver identity without borrowing the
    /// solver or its observation columns, so another worker can own construction.
    /// The factory's admission checkpoints cover nomination and plan metadata.
    pub fn prepare_pair_replacement(
        &self,
        factory: &dyn FixedReducedInverseFactory,
    ) -> Result<Option<OwnedPairBuild>, BuildError> {
        let started = Instant::now();
        let Some((base, nomination, selected)) = self.pair_entries(Some(factory))? else {
            return Ok(None);
        };
        let shared = Arc::new(base.inner.clone());
        let PreconditionerConfig::Additive { reduction, .. } = &base.config else {
            unreachable!("additive variant has additive configuration")
        };
        Ok(Some(OwnedPairBuild {
            shared,
            selected,
            nomination,
            n_dofs: self.n_dofs(),
            reduction: *reduction,
            source: self.replacement_source.get_or_init(|| Arc::new(())).clone(),
            started,
        }))
    }

    fn pair_entries(
        &self,
        factory: Option<&dyn FixedReducedInverseFactory>,
    ) -> Result<Option<NominatedEntries<'_>>, BuildError> {
        let Some(preconditioner) = self.preconditioner() else {
            return Ok(None);
        };
        let Variant::Additive(base) = &preconditioner.inner else {
            return Ok(None);
        };
        let PreconditionerConfig::Additive { local_solver, .. } = &base.config else {
            return Ok(None);
        };
        checkpoint(factory, 32)?;
        let count = base.inner.subdomains().len();
        let mut pairs: Vec<PairNomination> = Vec::new();
        pairs
            .try_reserve_exact(count)
            .map_err(|_| BuildError::Allocation {
                context: "pair nomination metadata",
            })?;
        for entry in base.inner.subdomains() {
            checkpoint(factory, 128)?;
            let Some(factors) = self.entry_pair(entry, factory)? else {
                continue;
            };
            let cross = entry
                .solver()
                .categorical_cross_tab()
                .expect("eligible pair");
            let mut found = None;
            for (index, pair) in pairs.iter().enumerate() {
                checkpoint(factory, 32)?;
                if pair.factors == factors {
                    found = Some(index);
                    break;
                }
            }
            let index = found.unwrap_or_else(|| {
                pairs.push(PairNomination {
                    factors,
                    max_retained: 0,
                    total_retained: 0,
                    replacement_components: 0,
                });
                pairs.len() - 1
            });
            let candidate = &mut pairs[index];
            candidate.max_retained = candidate.max_retained.max(cross.c.ncols);
            candidate.total_retained = candidate
                .total_retained
                .checked_add(cross.c.ncols)
                .ok_or_else(|| BuildError::LocalSolverBuild("pair size overflow".into()))?;
            if cross.c.ncols > local_solver.dense_threshold {
                candidate.replacement_components += 1;
            }
        }
        let mut best: Option<PairNomination> = None;
        for candidate in pairs {
            checkpoint(factory, 64)?;
            if candidate.replacement_components == 0 {
                continue;
            }
            let score = |p: PairNomination| {
                (
                    p.max_retained,
                    p.total_retained,
                    std::cmp::Reverse(p.factors),
                )
            };
            if best.is_none_or(|current| score(candidate) > score(current)) {
                best = Some(candidate);
            }
        }
        let Some(nomination) = best else {
            return Ok(None);
        };
        checkpoint(factory, 32)?;
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(nomination.replacement_components)
            .map_err(|_| BuildError::Allocation {
                context: "pair nomination components",
            })?;
        for (index, entry) in base.inner.subdomains().iter().enumerate() {
            checkpoint(factory, 64)?;
            if self.entry_pair(entry, factory)? == Some(nomination.factors)
                && entry.solver().n_kept() > local_solver.dense_threshold
            {
                selected.push(index);
            }
        }
        Ok(Some((base, nomination, selected)))
    }

    fn entry_pair(
        &self,
        entry: &SubdomainEntry<BlockElimSolver>,
        factory: Option<&dyn FixedReducedInverseFactory>,
    ) -> Result<Option<(usize, usize)>, BuildError> {
        let Some(cross) = entry.solver().categorical_cross_tab() else {
            return Ok(None);
        };
        let indices = entry.global_indices();
        let (Some(&first), Some(&second)) = (indices.first(), indices.get(cross.c.nrows)) else {
            return Ok(None);
        };
        let (Some(a), Some(b)) = (
            self.channel_at(first as usize, factory)?,
            self.channel_at(second as usize, factory)?,
        ) else {
            return Ok(None);
        };
        Ok((a.term != b.term
            && matches!(self.design.loading(a), Loading::Constant)
            && matches!(self.design.loading(b), Loading::Constant))
        .then_some((a.term.min(b.term), a.term.max(b.term))))
    }

    fn channel_at(
        &self,
        index: usize,
        factory: Option<&dyn FixedReducedInverseFactory>,
    ) -> Result<Option<Channel>, BuildError> {
        if index >= self.n_dofs() {
            return Ok(None);
        }
        let (mut low, mut high) = (0, self.design.terms.len());
        while low < high {
            checkpoint(factory, 32)?;
            let middle = low + (high - low) / 2;
            if self.design.terms[middle].offset <= index {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        let Some(term) = low.checked_sub(1) else {
            return Ok(None);
        };
        let meta = &self.design.terms[term];
        Ok(Some(Channel {
            term,
            column: (index - meta.offset) / meta.n_levels,
        }))
    }
}

fn checkpoint(
    factory: Option<&dyn FixedReducedInverseFactory>,
    work: u64,
) -> Result<(), BuildError> {
    factory.map_or(Ok(()), |factory| factory.checkpoint(work))
}

#[cfg(test)]
pub(crate) mod tests;
