//! Optional runtime-only fixed reduced inverses for categorical pair components.

use std::collections::BTreeMap;
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
    /// Copy charges also cover the final scheduler's index scan. The caller
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
    /// Total local dimension over this pair's components (the secondary score).
    pub total_levels: usize,
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

    /// Apply the complete replacement Schwarz action, including all unchanged pairs.
    pub fn apply(&self, rhs: &[f64], out: &mut [f64]) -> Result<(), crate::SolveError> {
        self.preconditioner.apply(rhs, out)
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

impl<'a> Solver<'a> {
    /// Nominate the categorical pair with largest `max_c min(L_ac, L_bc)`;
    /// ties prefer larger total dimension, then smaller original factor indices.
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
        let started = Instant::now();
        let Some((base, nomination, selected)) = self.pair_entries(Some(factory))? else {
            return Ok(None);
        };
        let shared = Arc::new(base.inner.clone());
        factory.checkpoint(32)?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(base.inner.subdomains().len())
            .map_err(|_| BuildError::Allocation {
                context: "pair replacement entries",
            })?;
        for (index, entry) in base.inner.subdomains().iter().enumerate() {
            factory.checkpoint(128)?;
            let solver = if selected.binary_search(&index).is_ok() {
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
                    source: shared.clone(),
                    index,
                }
            };
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
        let PreconditionerConfig::Additive { reduction, .. } = &base.config else {
            unreachable!("additive variant has additive configuration")
        };
        let preconditioner = SchwarzPreconditioner::with_n_dofs(entries, self.n_dofs(), *reduction);
        Ok(Some(PairReplacement {
            source: self.replacement_source.get_or_init(|| Arc::new(())).clone(),
            identity: Arc::new(()),
            preconditioner,
            build_duration: started.elapsed(),
            nomination,
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
        let mut pairs: BTreeMap<(usize, usize), (PairNomination, Vec<usize>)> = BTreeMap::new();
        for (index, entry) in base.inner.subdomains().iter().enumerate() {
            if let Some(factory) = factory {
                factory.checkpoint(128)?;
            }
            let solver = entry.solver();
            let Some(cross) = solver.categorical_cross_tab() else {
                continue;
            };
            let indices = entry.global_indices();
            let (Some(a), Some(b)) = (
                indices.first().and_then(|&i| self.channel_at(i as usize)),
                indices
                    .get(cross.c.nrows)
                    .and_then(|&i| self.channel_at(i as usize)),
            ) else {
                return Ok(None);
            };
            if a.term == b.term
                || !matches!(self.design.loading(a), Loading::Constant)
                || !matches!(self.design.loading(b), Loading::Constant)
            {
                continue;
            }
            let factors = (a.term.min(b.term), a.term.max(b.term));
            let (candidate, selected) = pairs.entry(factors).or_insert((
                PairNomination {
                    factors,
                    max_retained: 0,
                    total_levels: 0,
                    replacement_components: 0,
                },
                Vec::new(),
            ));
            candidate.max_retained = candidate.max_retained.max(cross.c.ncols);
            candidate.total_levels += cross.n_local();
            if cross.c.ncols > local_solver.dense_threshold {
                candidate.replacement_components += 1;
                selected
                    .try_reserve(1)
                    .map_err(|_| BuildError::Allocation {
                        context: "pair nomination components",
                    })?;
                selected.push(index);
            }
        }
        // BTreeMap order makes exact ties choose the lexicographically first pair.
        let mut best: Option<(PairNomination, Vec<usize>)> = None;
        for (candidate, selected) in pairs.into_values() {
            if let Some(factory) = factory {
                factory.checkpoint(128)?;
            }
            if selected.is_empty() {
                continue;
            }
            if best.as_ref().is_none_or(|(current, _)| {
                (candidate.max_retained, candidate.total_levels)
                    > (current.max_retained, current.total_levels)
            }) {
                best = Some((candidate, selected));
            }
        }
        Ok(best.map(|(nomination, selected)| (base, nomination, selected)))
    }

    fn channel_at(&self, index: usize) -> Option<Channel> {
        if index >= self.n_dofs() {
            return None;
        }
        let term = self
            .design
            .terms
            .partition_point(|term| term.offset <= index)
            .checked_sub(1)?;
        let meta = &self.design.terms[term];
        Some(Channel {
            term,
            column: (index - meta.offset) / meta.n_levels,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests;
