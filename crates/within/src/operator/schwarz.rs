//! Schwarz preconditioner: bridges FE domain types to the generic
//! `schwarz-precond` API, plus the opaque public [`Preconditioner`] handle.

use rayon::prelude::*;
use schwarz_precond::{
    Operator, PartitionWeights, SchwarzPreconditioner, SubdomainCore, SubdomainEntry,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::block_elim::BlockElimSolver;
use crate::config::{LocalSolverConfig, PreconditionerConfig};
use crate::domain::Loading;
use crate::domain::{for_each_local_domain_checked, Design, LocalDomain};
use crate::setup::{Ledger, SetupLimit, SetupReport};
use crate::{BuildError, BuildWarning};

mod serial;
pub use serial::SerialPreconditionerWorkspace;

#[cfg(test)]
mod checked_tests;
#[cfg(test)]
mod tests;

/// Concrete additive Schwarz type used in the parent crate.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct FeSchwarz {
    inner: SchwarzPreconditioner<BlockElimSolver>,
    config: PreconditionerConfig,
}

impl FeSchwarz {
    fn config(&self) -> &PreconditionerConfig {
        &self.config
    }
}

impl std::fmt::Debug for FeSchwarz {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeSchwarz")
            .field("n_subdomains", &self.inner.subdomains().len())
            .field("config", &self.config)
            .finish()
    }
}

impl Operator for FeSchwarz {
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.inner.apply(x, y)
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.inner.apply_adjoint(x, y)
    }
}

/// Diagonal/Jacobi preconditioner for the fixed-effects Gramian.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct DiagonalPreconditioner {
    inv_diag: Arc<[f64]>,
}

impl Operator for DiagonalPreconditioner {
    fn nrows(&self) -> usize {
        self.inv_diag.len()
    }

    fn ncols(&self) -> usize {
        self.inv_diag.len()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        let n = self.inv_diag.len();
        if x.len() != n || y.len() != n {
            return Err(schwarz_precond::SolveError::InvalidInput {
                context: "DiagonalPreconditioner::apply",
                message: format!(
                    "x.len()={}, y.len()={}, expected n_dofs={}",
                    x.len(),
                    y.len(),
                    n
                ),
            });
        }
        for ((yi, &xi), &di) in y.iter_mut().zip(x.iter()).zip(self.inv_diag.iter()) {
            *yi = di * xi;
        }
        Ok(())
    }

    /// A diagonal operator is symmetric (`M^T = M`), so the adjoint is `apply`.
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.apply(x, y)
    }
}

/// `n_dofs` may exceed the span of subdomain indices; an uncovered column resolves to `0`.
pub(crate) fn build_additive_with_strategy(
    domains: Vec<LocalDomain>,
    config: &LocalSolverConfig,
    strategy: schwarz_precond::ReductionStrategy,
    n_dofs: usize,
) -> Result<FeSchwarz, BuildError> {
    let entries = domains
        .into_par_iter()
        .map(|domain| build_entry(domain, config))
        .collect::<Result<Vec<_>, BuildError>>()?;
    Ok(FeSchwarz {
        inner: SchwarzPreconditioner::with_n_dofs(entries, n_dofs, strategy),
        config: PreconditionerConfig::Additive {
            local_solver: config.clone(),
            reduction: strategy,
        },
    })
}

/// Build a single `SubdomainEntry<BlockElimSolver>` from a pre-built CrossTab.
pub(crate) fn build_entry(
    domain: LocalDomain,
    config: &LocalSolverConfig,
) -> Result<SubdomainEntry<BlockElimSolver>, BuildError> {
    let LocalDomain { core, component } = domain;
    let solver = BlockElimSolver::build(component, config)?;
    SubdomainEntry::try_new(core, solver).map_err(BuildError::Preconditioner)
}

/// Opaque handle to a pre-built preconditioner; cloning is O(1) via `Arc`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Preconditioner {
    inner: Variant,
    build_duration: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Variant {
    // Keep Additive first: postcard encodes by declaration order and the fixture depends on it.
    Additive(FeSchwarz),
    Diagonal(DiagonalPreconditioner),
}

impl Preconditioner {
    /// Stable display name for the concrete preconditioner variant.
    pub fn variant_name(&self) -> &'static str {
        match &self.inner {
            Variant::Additive(_) => "Additive",
            Variant::Diagonal(_) => "Diagonal",
        }
    }

    /// Complete configuration used to build this preconditioner.
    pub fn config(&self) -> &PreconditionerConfig {
        const DIAGONAL: PreconditionerConfig = PreconditionerConfig::Diagonal;
        match &self.inner {
            Variant::Additive(p) => p.config(),
            Variant::Diagonal(_) => &DIAGONAL,
        }
    }

    // `within` does not re-export `schwarz_precond::Operator`, so these are the only public way.
    /// Number of rows of the underlying linear operator.
    pub fn nrows(&self) -> usize {
        <Self as schwarz_precond::Operator>::nrows(self)
    }

    /// Number of columns of the underlying linear operator.
    pub fn ncols(&self) -> usize {
        <Self as schwarz_precond::Operator>::ncols(self)
    }

    /// Apply the preconditioner: writes `M^{-1} x` into `y`.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        <Self as schwarz_precond::Operator>::apply(self, x, y)
    }

    /// Time spent building this preconditioner.
    pub fn build_duration(&self) -> Duration {
        self.build_duration
    }

    /// Build the additive Schwarz preconditioner serially within a declared live
    /// payload, returning it with its accounting.
    ///
    /// Takes the design and caller-order weights [`Solver::new`](crate::Solver::new)
    /// takes, and the local-solver and reduction settings of
    /// [`PreconditionerConfig::Additive`]. Every array of the build is charged to
    /// one ledger before it is allocated: weight permutation, active-level maps,
    /// pair cross-tabulations, component splits, exact and sampled Schur
    /// complements and their sampler scratch, every `approx-chol` factor through
    /// its own checked builder (including a failed exact attempt's peak),
    /// partition weights and the shared blocks of the returned value. Exceeding
    /// `limit` returns [`BuildError::SetupBudgetExceeded`] without the refused
    /// allocation; a refused reservation returns
    /// [`BuildError::SetupAllocationFailed`]. A successful build returns a
    /// preconditioner whose action and serialized form equal the one
    /// [`Solver::new`](crate::Solver::new) builds from the same input, whatever the
    /// limit. Its retained payload is the report's `retained_payload_bytes`.
    ///
    /// Covers intercept-only designs, whose pairs are known Laplacians; slope
    /// effects return [`BuildError::CheckedSetupUnsupported`] before any
    /// allocation. Fixed-size reference-counted blocks are allocated infallibly
    /// after being charged. `Ok((None, _))` means the design has no pair
    /// subdomain, as an unchecked build returns no preconditioner.
    pub fn build_checked(
        design: &Design<'_>,
        weights: Option<&[f64]>,
        local_solver: &LocalSolverConfig,
        reduction: schwarz_precond::ReductionStrategy,
        limit: SetupLimit,
    ) -> Result<(Option<Self>, SetupReport), BuildError> {
        let mut ledger = Ledger::new(limit);
        let preconditioner =
            Self::build_checked_in(design, weights, local_solver, reduction, &mut ledger)?;
        Ok((preconditioner, ledger.report()))
    }

    /// [`Self::build_checked`] against a caller's ledger.
    pub(crate) fn build_checked_in(
        design: &Design<'_>,
        weights: Option<&[f64]>,
        local_solver: &LocalSolverConfig,
        reduction: schwarz_precond::ReductionStrategy,
        ledger: &mut Ledger,
    ) -> Result<Option<Self>, BuildError> {
        const CONTEXT: &str = "subdomain assembly";
        let build_started = Instant::now();
        if !local_solver.ridge.is_finite() || local_solver.ridge < 0.0 {
            return Err(BuildError::InvalidRidge {
                value: local_solver.ridge,
            });
        }
        design.validate_weights(weights)?;
        if (0..design.n_factors())
            .flat_map(|term| design.channels(term))
            .any(|channel| design.loading(channel).covariate().is_some())
        {
            return Err(BuildError::CheckedSetupUnsupported {
                reason: "slope (covariate) effects",
            });
        }
        if u32::try_from(design.n_obs()).is_err() {
            return Err(BuildError::CheckedSetupUnsupported {
                reason: "more than u32::MAX observations",
            });
        }

        let permuted = match (&design.obs_perm, weights) {
            (Some(perm), Some(weights)) => {
                let mut permuted = ledger.with_capacity("weight permutation", perm.len())?;
                permuted.extend(perm.iter().map(|&i| weights[i as usize]));
                Some(permuted)
            }
            _ => None,
        };
        let internal_weights = match &design.obs_perm {
            Some(_) => permuted.as_deref(),
            None => weights,
        };

        let mut built: Vec<(SubdomainCore, BlockElimSolver)> = Vec::new();
        for_each_local_domain_checked(
            design,
            internal_weights,
            ledger,
            |ledger, globals, component| {
                let solver = BlockElimSolver::build_checked(component, local_solver, ledger)?;
                ledger.push(
                    CONTEXT,
                    &mut built,
                    (SubdomainCore::uniform(globals), solver),
                )
            },
        )?;
        if let Some(permuted) = permuted {
            ledger.free(permuted);
        }
        if built.is_empty() {
            ledger.free(built);
            return Ok(None);
        }

        // Intercept-only channels carry no slope, so the `1/√c` weights always apply.
        let mut counts = ledger.filled("partition weights", design.n_dofs, 0u32)?;
        for (core, _) in &built {
            for &idx in core.global_indices() {
                counts[idx as usize] += 1;
            }
        }
        for (core, _) in &mut built {
            if core
                .global_indices()
                .iter()
                .all(|&idx| counts[idx as usize] <= 1)
            {
                core.set_uniform_partition_weights();
            } else {
                let mut weights = ledger.with_capacity("partition weights", core.n_local())?;
                weights.extend(core.global_indices().iter().map(|&idx| {
                    let c = counts[idx as usize];
                    debug_assert!(c > 0);
                    1.0 / (c as f64).sqrt()
                }));
                core.set_partition_weights(PartitionWeights::NonUniform(weights))
                    .expect("partition weight count must match index count");
            }
        }
        ledger.free(counts);

        let mut entries = ledger.with_capacity(CONTEXT, built.len())?;
        let built_capacity = built.capacity();
        for (core, solver) in built {
            entries
                .push(SubdomainEntry::try_new(core, solver).map_err(BuildError::Preconditioner)?);
        }
        ledger.forget::<(SubdomainCore, BlockElimSolver)>(built_capacity);
        ledger.charge(
            CONTEXT,
            SchwarzPreconditioner::<BlockElimSolver>::assembly_heap_bytes(),
        )?;
        let inner = SchwarzPreconditioner::with_n_dofs(entries, design.n_dofs, reduction);
        let preconditioner = Preconditioner {
            inner: Variant::Additive(FeSchwarz {
                inner,
                config: PreconditionerConfig::Additive {
                    local_solver: local_solver.clone(),
                    reduction,
                },
            }),
            build_duration: build_started.elapsed(),
        };
        Ok(Some(preconditioner))
    }

    /// Checked sum of every local reduced factor's retained heap capacity in bytes.
    ///
    /// Counts each approximate-Cholesky factor's block, fallback, permutation and
    /// nested exact or approximate arrays, including spare capacity. Excludes the
    /// local cross-tabulations, inverse diagonals and coordinate maps, the
    /// restriction indices and weights, inline objects, allocator overhead and every
    /// construction peak. `None` on arithmetic overflow; diagonal variants hold no
    /// factors. A post-build inventory, not construction admission.
    pub fn local_factor_retained_heap_bytes(&self) -> Option<usize> {
        match &self.inner {
            Variant::Additive(p) => p.inner.subdomains().iter().try_fold(0usize, |sum, entry| {
                sum.checked_add(entry.solver().factor_retained_heap_bytes()?)
            }),
            Variant::Diagonal(_) => Some(0),
        }
    }
}

impl Operator for Preconditioner {
    fn nrows(&self) -> usize {
        match &self.inner {
            Variant::Additive(p) => p.nrows(),
            Variant::Diagonal(p) => p.nrows(),
        }
    }

    fn ncols(&self) -> usize {
        match &self.inner {
            Variant::Additive(p) => p.ncols(),
            Variant::Diagonal(p) => p.ncols(),
        }
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        match &self.inner {
            Variant::Additive(p) => p.apply(x, y),
            Variant::Diagonal(p) => p.apply(x, y),
        }
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        match &self.inner {
            Variant::Additive(p) => p.apply_adjoint(x, y),
            Variant::Diagonal(p) => p.apply_adjoint(x, y),
        }
    }
}

fn build_diagonal(
    design: &Design<'_>,
    weights: Option<&[f64]>,
) -> Result<DiagonalPreconditioner, BuildError> {
    let mut diag = vec![0.0; design.n_dofs];

    for (factor_idx, term) in design.terms.iter().enumerate() {
        let levels = design.frame.level_column(factor_idx);
        let w = |uid: usize| weights.map_or(1.0, |ws| ws[uid]);
        for (column, loading) in term.columns.iter().enumerate() {
            let base = term.column_base(column);
            let slice = &mut diag[base..base + term.n_levels];
            match loading {
                Loading::Constant => {
                    for (uid, &level) in levels.iter().enumerate() {
                        slice[level as usize] += w(uid);
                    }
                }
                Loading::Covariate(z_col) => {
                    let z = design.frame.loading_column(*z_col as usize);
                    for (uid, &level) in levels.iter().enumerate() {
                        // Keep `w * z * z` left-to-right: a zero weight kills a huge `z` first.
                        slice[level as usize] += w(uid) * z[uid] * z[uid];
                    }
                }
            }
        }
    }

    // A zero diagonal is an unidentified DOF, so the pseudo-inverse keeps it in the null space.
    for (index, d) in diag.iter_mut().enumerate() {
        if *d == 0.0 {
            continue;
        }
        let inv = 1.0 / *d;
        if !inv.is_finite() {
            return Err(BuildError::SingularDiagonal { index });
        }
        *d = inv;
    }

    Ok(DiagonalPreconditioner {
        inv_diag: Arc::from(diag),
    })
}

/// Build a [`Preconditioner`] from a design and optional weights, plus any warnings.
pub(crate) fn build_preconditioner(
    design: &Design<'_>,
    weights: Option<&[f64]>,
    config: Option<&PreconditionerConfig>,
) -> Result<(Option<Preconditioner>, Vec<BuildWarning>), BuildError> {
    use crate::domain::build_local_domains;

    // Weights are pre-validated by the sole caller, whose permutation preserves length and sign.
    let default_cfg = PreconditionerConfig::default();
    let resolved = config.unwrap_or(&default_cfg);
    // Measure preconditioner build time
    let build_started = Instant::now();
    let (inner, warnings) = match resolved {
        PreconditionerConfig::Off => {
            return Ok((None, Vec::new()));
        }
        PreconditionerConfig::Additive {
            local_solver,
            reduction,
        } => {
            let (domains, warnings) = build_local_domains(design, weights, local_solver)?;
            if domains.is_empty() {
                // No factor-pair subdomains means no useful Schwarz; fall back to plain LSMR.
                return Ok((None, warnings));
            }
            let preconditioner =
                build_additive_with_strategy(domains, local_solver, *reduction, design.n_dofs)?;
            (Variant::Additive(preconditioner), warnings)
        }
        PreconditionerConfig::Diagonal => {
            let preconditioner = build_diagonal(design, weights)?;
            (Variant::Diagonal(preconditioner), Vec::new())
        }
    };
    let build_duration = build_started.elapsed();
    Ok((
        Some(Preconditioner {
            inner,
            build_duration,
        }),
        warnings,
    ))
}
