//! Fixed-order serial application with caller-owned, owner-bound storage.
use super::SchwarzPreconditioner;
use crate::{LocalSolver, Operator, OperatorMut, SolveError};

/// Serial additive action borrowing one immutable preconditioner.
///
/// One array holds a transactional global result and two maximum-local scratch
/// vectors, including any augmentation reported by the local solver. Domains
/// run in stored order with `allow_inner_parallelism = false`. This executor
/// never enters Rayon, locks a scratch pool, or allocates during application.
/// Generic local solvers may still allocate, use threads, or allocate errors:
/// their trait does not promise otherwise. Owner/factor memory is excluded.
///
/// Output is unchanged on any returned error. Local errors may dirty scratch;
/// the next call resets accumulation and reuses storage. Local solvers must
/// obey their scratch contract and fully initialize any values they read, just
/// as with the existing pooled executor. The borrowed owner cannot be replaced;
/// callers must also keep any interior-mutable local numerical state fixed.
///
/// ```compile_fail
/// use schwarz_precond::{LocalSolver, SchwarzPreconditioner, SerialSchwarzWorkspace};
/// fn escape<S: LocalSolver>(owner: SchwarzPreconditioner<S>)
///     -> SerialSchwarzWorkspace<'static, S>
/// {
///     owner.try_serial_workspace(usize::MAX).unwrap()
/// }
/// ```
pub struct SerialSchwarzWorkspace<'owner, S: LocalSolver> {
    owner: &'owner SchwarzPreconditioner<S>,
    dimension: usize,
    max_scratch: usize,
    values: Vec<f64>,
}

fn checked_payload(dimension: usize, scratch: usize) -> Result<usize, SolveError> {
    scratch
        .checked_mul(2)
        .and_then(|n| n.checked_add(dimension))
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or(SolveError::SerialWorkspaceSizeOverflow)
}

fn layout<S: LocalSolver>(owner: &SchwarzPreconditioner<S>) -> Result<(usize, usize), SolveError> {
    let n = owner.nrows();
    let mut max_scratch = 0;
    for (subdomain, entry) in owner.subdomains().iter().enumerate() {
        let scratch = entry.scratch_size();
        if scratch < entry.global_indices().len()
            || entry.solver().n_local() != entry.global_indices().len()
            || entry.global_indices().iter().any(|&i| i as usize >= n)
        {
            return Err(SolveError::SerialWorkspaceInvalidDomain { subdomain });
        }
        max_scratch = max_scratch.max(scratch);
    }
    checked_payload(n, max_scratch)?;
    Ok((n, max_scratch))
}

impl<S: LocalSolver> SchwarzPreconditioner<S> {
    /// Checked requested bytes for serial action: `8 * (n_dofs + 2 * max_scratch)`.
    ///
    /// This is new outer workspace payload only. It excludes the owner, factor
    /// internals, any old pooled buffers, inline state and allocator overhead.
    pub fn serial_workspace_required_payload_bytes(&self) -> Result<usize, SolveError> {
        let (n, scratch) = layout(self)?;
        checked_payload(n, scratch)
    }

    /// Fallibly reserve a workspace permanently borrowing this preconditioner.
    ///
    /// Validate domains, representability and the requested-array budget before
    /// the single reservation. The limit is not a total-process or local-factor
    /// memory bound. Inspect actual retained capacity on the returned workspace;
    /// allocator rounding/excess capacity is outside the requested-byte limit.
    pub fn try_serial_workspace(
        &self,
        max_requested_payload_bytes: usize,
    ) -> Result<SerialSchwarzWorkspace<'_, S>, SolveError> {
        SerialSchwarzWorkspace::try_new_with(self, max_requested_payload_bytes, |count| {
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|_| SolveError::SerialWorkspaceAllocation)?;
            values.resize(count, 0.0);
            Ok(values)
        })
    }
}

impl<'owner, S: LocalSolver> SerialSchwarzWorkspace<'owner, S> {
    fn try_new_with(
        owner: &'owner SchwarzPreconditioner<S>,
        limit: usize,
        allocate: impl FnOnce(usize) -> Result<Vec<f64>, SolveError>,
    ) -> Result<Self, SolveError> {
        let (dimension, max_scratch) = layout(owner)?;
        let required = checked_payload(dimension, max_scratch)?;
        if required > limit {
            return Err(SolveError::SerialWorkspaceBudget { required, limit });
        }
        let values = allocate(required / std::mem::size_of::<f64>())?;
        Ok(Self {
            owner,
            dimension,
            max_scratch,
            values,
        })
    }

    /// Dimension of the bound square action.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Actual retained outer-array capacity; excludes all owner/local-solver memory.
    pub fn retained_payload_bytes(&self) -> usize {
        self.values.capacity() * std::mem::size_of::<f64>()
    }

    /// Apply in fixed domain order; only publish output after every local solve succeeds.
    ///
    /// Shape and current domain dimensions validate before touching scratch or
    /// output. A generic solver that now needs more than the bound scratch is rejected.
    /// Success follows the local-solver contract; it is not a finite-value,
    /// range-preservation, positivity or original-system certificate.
    pub fn apply(&mut self, rhs: &[f64], output: &mut [f64]) -> Result<(), SolveError> {
        if rhs.len() != self.dimension || output.len() != self.dimension {
            return Err(SolveError::SerialWorkspaceShape {
                rhs: rhs.len(),
                output: output.len(),
                expected: self.dimension,
            });
        }
        // Immutable core indices were checked once when binding. Only generic
        // local-solver dimensions can change through interior mutability; check
        // those before touching scratch without rescanning all covered DOFs.
        for (subdomain, entry) in self.owner.subdomains().iter().enumerate() {
            let len = entry.scratch_size();
            if entry.solver().n_local() != entry.global_indices().len()
                || len < entry.global_indices().len()
            {
                return Err(SolveError::SerialWorkspaceInvalidDomain { subdomain });
            }
            if len > self.max_scratch {
                return Err(SolveError::SerialWorkspaceChanged);
            }
        }
        let (result, local) = self.values.split_at_mut(self.dimension);
        let (r, z) = local.split_at_mut(self.max_scratch);
        result.fill(0.0);
        for (subdomain, entry) in self.owner.subdomains().iter().enumerate() {
            let len = entry.scratch_size();
            entry
                .apply_weighted_into_with_scratch(rhs, result, &mut r[..len], &mut z[..len], false)
                .map_err(|source| SolveError::LocalSolveFailed { subdomain, source })?;
        }
        output.copy_from_slice(result);
        Ok(())
    }
}

impl<S: LocalSolver> OperatorMut for SerialSchwarzWorkspace<'_, S> {
    fn nrows(&self) -> usize {
        self.dimension
    }
    fn ncols(&self) -> usize {
        self.dimension
    }
    fn apply(&mut self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        SerialSchwarzWorkspace::apply(self, x, y)
    }
    fn apply_adjoint(&mut self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        SerialSchwarzWorkspace::apply(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LocalSolveError, ReductionStrategy, SubdomainCore, SubdomainEntry};
    struct Identity(usize);
    impl LocalSolver for Identity {
        fn n_local(&self) -> usize {
            self.0
        }
        fn scratch_size(&self) -> usize {
            self.0
        }
        fn solve_local(
            &self,
            r: &mut [f64],
            z: &mut [f64],
            _: bool,
        ) -> Result<(), LocalSolveError> {
            z.copy_from_slice(r);
            Ok(())
        }
    }
    #[test]
    fn reservation_failure_and_unwind_leave_owner_usable() {
        let entry =
            SubdomainEntry::try_new(SubdomainCore::uniform(vec![0, 1]), Identity(2)).unwrap();
        let owner = SchwarzPreconditioner::new(vec![entry], ReductionStrategy::Auto);
        let failure = SerialSchwarzWorkspace::try_new_with(&owner, 48, |_| {
            Err(SolveError::SerialWorkspaceAllocation)
        });
        assert!(matches!(
            failure,
            Err(SolveError::SerialWorkspaceAllocation)
        ));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = SerialSchwarzWorkspace::try_new_with(&owner, 48, |_| {
                panic!("injected reservation unwind")
            });
        }))
        .is_err());
        let mut workspace = owner.try_serial_workspace(48).unwrap();
        let mut out = [0.; 2];
        workspace.apply(&[1., 2.], &mut out).unwrap();
        assert_eq!(out, [1., 2.]);
    }
    #[test]
    fn checked_sizes_reject_before_reservation() {
        assert!(checked_payload(usize::MAX, 0).is_err());
        assert!(checked_payload(0, usize::MAX).is_err());
        assert!(checked_payload(isize::MAX as usize / 8 + 1, 0).is_err());
        assert_eq!(checked_payload(0, 0).unwrap(), 0);
        let owner =
            SchwarzPreconditioner::<Identity>::with_n_dofs(vec![], 2, ReductionStrategy::Auto);
        let denied =
            SerialSchwarzWorkspace::try_new_with(&owner, 15, |_| panic!("must not reserve"));
        assert!(matches!(
            denied,
            Err(SolveError::SerialWorkspaceBudget {
                required: 16,
                limit: 15
            })
        ));
    }
}
