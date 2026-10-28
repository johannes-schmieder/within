//! Local solver trait and subdomain entry.
//!
//! In the Schwarz formula `M⁻¹ = Σ Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ`, this module
//! provides the two key abstractions:
//!
//! - [`LocalSolver`] — the `Aᵢ⁻¹` operator: given a local right-hand
//!   side, produce an approximate (or exact) local solution. Implementations
//!   are problem-specific (e.g. approximate Cholesky, block elimination).
//!
//! - [`SubdomainEntry`] — bundles a [`SubdomainCore`] (which implements
//!   `Rᵢ` and `D̃ᵢ`) with a `LocalSolver` (which implements `Aᵢ⁻¹`),
//!   giving a single object that can compute the per-subdomain
//!   contribution `D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ r` via [`SubdomainEntry::apply_weighted_local`].

use crate::domain::SubdomainCore;
use crate::error::{BuildError, LocalSolveError};

/// The `Aᵢ⁻¹` operator: solves the restricted system `Aᵢ = Rᵢ A Rᵢᵀ`, exactly or approximately.
pub trait LocalSolver: Send + Sync {
    /// Number of DOFs in the subdomain (before augmentation).
    fn n_local(&self) -> usize;

    /// Required scratch buffer size (may be > n_local for augmented systems).
    fn scratch_size(&self) -> usize;

    /// Solve into `sol` with `scratch_size()` buffers; hint and thread count can't change bits.
    fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError>;

    /// Estimated local work that can benefit from nested Rayon; zero when there is no such region.
    fn inner_parallelism_work_estimate(&self) -> usize {
        0
    }
}

/// One term of the Schwarz sum `Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ`.
#[derive(Clone)]
pub struct SubdomainEntry<S: LocalSolver> {
    /// Subdomain core with restriction indices and partition-of-unity weights.
    core: SubdomainCore,
    /// The local solver for this subdomain.
    solver: S,
}

impl<S: LocalSolver> SubdomainEntry<S> {
    /// Create a validated subdomain entry from a core and a local solver.
    pub fn try_new(core: SubdomainCore, solver: S) -> Result<Self, BuildError> {
        let index_count = core.n_local();
        let solver_n_local = solver.n_local();
        if solver_n_local != index_count {
            return Err(BuildError::LocalDofCountMismatch {
                index_count,
                solver_n_local,
            });
        }

        let scratch_size = solver.scratch_size();
        if scratch_size < index_count {
            return Err(BuildError::ScratchSizeTooSmall {
                scratch_size,
                required_min: index_count,
            });
        }

        Ok(Self { core, solver })
    }

    /// Global DOF indices covered by this subdomain.
    pub fn global_indices(&self) -> &[u32] {
        self.core.global_indices()
    }

    /// Partition-of-unity weights for this subdomain.
    pub fn partition_weights(&self) -> &crate::domain::PartitionWeights {
        self.core.partition_weights()
    }

    /// The local solver for this subdomain.
    pub fn solver(&self) -> &S {
        &self.solver
    }

    /// Returns `true` if this subdomain is empty.
    pub fn is_empty(&self) -> bool {
        self.core.is_empty()
    }

    /// Required scratch buffer size (delegates to solver).
    #[inline]
    pub fn scratch_size(&self) -> usize {
        self.solver.scratch_size()
    }

    /// Writes `D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ r` locally to `out`; `scratch` must hold `scratch_size()`.
    pub(crate) fn apply_weighted_local(
        &self,
        r: &[f64],
        out: &mut [f64],
        scratch: &mut LocalSolveScratch,
        allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        if self.core.is_empty() {
            return Ok(());
        }
        let LocalSolveScratch { rhs, sol } = scratch;
        self.core.restrict_weighted(r, rhs);
        self.solver.solve_local(rhs, sol, allow_inner_parallelism)?;
        self.core.weight(sol, out);
        Ok(())
    }
}

/// The local right-hand side and solution buffers of one [`LocalSolver::solve_local`] call.
pub(crate) struct LocalSolveScratch {
    rhs: Vec<f64>,
    sol: Vec<f64>,
}

impl LocalSolveScratch {
    pub(crate) fn new(scratch_size: usize) -> Self {
        Self {
            rhs: vec![0.0; scratch_size],
            sol: vec![0.0; scratch_size],
        }
    }
}

#[cfg(feature = "serde")]
impl<S> serde::Serialize for SubdomainEntry<S>
where
    S: LocalSolver + serde::Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("SubdomainEntry", 2)?;
        state.serialize_field("core", &self.core)?;
        state.serialize_field("solver", &self.solver)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, S> serde::Deserialize<'de> for SubdomainEntry<S>
where
    S: LocalSolver + serde::de::DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(bound(deserialize = "S: serde::de::DeserializeOwned"))]
        struct Helper<S> {
            core: SubdomainCore,
            solver: S,
        }

        let helper = Helper::<S>::deserialize(deserializer)?;
        Self::try_new(helper.core, helper.solver).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::PartitionWeights;
    use rstest::rstest;

    /// Identity local solver: sol = rhs.
    struct IdentityLocalSolver {
        n: usize,
    }

    impl LocalSolver for IdentityLocalSolver {
        fn n_local(&self) -> usize {
            self.n
        }
        fn scratch_size(&self) -> usize {
            self.n
        }
        fn solve_local(
            &self,
            rhs: &mut [f64],
            sol: &mut [f64],
            _allow_inner_parallelism: bool,
        ) -> Result<(), LocalSolveError> {
            sol[..self.n].copy_from_slice(&rhs[..self.n]);
            Ok(())
        }
    }

    #[rstest]
    #[case::uniform(
        SubdomainCore::uniform(vec![1, 3, 4]),
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        vec![20.0, 40.0, 50.0],
        vec![20.0, 40.0, 50.0]
    )]
    // Two-sided partition of unity: the effective weight is `w²`.
    #[case::nonuniform(
        SubdomainCore::with_partition_weights(
            vec![0, 1, 2],
            PartitionWeights::NonUniform(vec![1.0, 0.5, 0.25]),
        )
        .expect("matching non-uniform weights"),
        vec![4.0, 8.0, 16.0],
        vec![4.0, 4.0, 4.0],
        vec![4.0, 2.0, 1.0]
    )]
    fn test_restrict_and_weight(
        #[case] core: SubdomainCore,
        #[case] global: Vec<f64>,
        #[case] restricted: Vec<f64>,
        #[case] weighted: Vec<f64>,
    ) {
        let mut local = vec![0.0; restricted.len()];
        core.restrict_weighted(&global, &mut local);
        assert_eq!(local, restricted);

        let mut out = vec![0.0; weighted.len()];
        core.weight(&local, &mut out);
        assert_eq!(out, weighted);
    }

    #[test]
    fn test_apply_weighted_with_identity_solver() {
        let core = SubdomainCore::uniform(vec![1, 2]);
        let solver = IdentityLocalSolver { n: 2 };
        let entry = SubdomainEntry::try_new(core, solver).expect("valid entry");

        let r = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; 2];
        let mut scratch = LocalSolveScratch::new(entry.scratch_size());

        entry
            .apply_weighted_local(&r, &mut out, &mut scratch, true)
            .expect("identity local solver should not fail");

        // Restricts DOFs [1, 2] → [20, 30] and solves with the identity.
        assert_eq!(out, vec![20.0, 30.0]);
    }
}
