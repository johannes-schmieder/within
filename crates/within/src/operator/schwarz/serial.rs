//! Public serial action over the existing local numerical kernels.
use super::{DiagonalPreconditioner, Preconditioner, Variant};
use crate::block_elim::BlockElimSolver;
use schwarz_precond::{Operator, OperatorMut, SerialSchwarzWorkspace, SolveError};

/// Caller-owned serial action bound by borrowing one immutable preconditioner.
///
/// Additive actions visit domains in stored order and disable the concrete local
/// kernels' optional inner parallelism. One array holds the global transaction
/// result and two maximum-local scratch vectors. Diagonal actions need no array.
/// Outputs are unchanged on any returned error; scratch is reusable afterward.
///
/// This bounds the outer action workspace only. Construction remains unchanged
/// and may use Rayon. Local factors remain opaque; the pinned approximate-
/// Cholesky dependency can allocate permutation scratch for some factors, and
/// local errors may allocate text. This API alone is not complete terminal
/// memory admission, an allocation-free guarantee for every factor, or a
/// range/positivity certificate. Existing pooled APIs retain their behavior.
pub struct SerialPreconditionerWorkspace<'owner> {
    inner: SerialVariant<'owner>,
}

enum SerialVariant<'owner> {
    Additive(SerialSchwarzWorkspace<'owner, BlockElimSolver>),
    Diagonal(&'owner DiagonalPreconditioner),
}

impl Preconditioner {
    /// Requested outer serial array bytes, excluding this owner and local factors.
    pub fn serial_workspace_required_payload_bytes(&self) -> Result<usize, SolveError> {
        match &self.inner {
            Variant::Additive(p) => p.inner.serial_workspace_required_payload_bytes(),
            Variant::Diagonal(_) => Ok(0),
        }
    }

    /// Bind serial action scratch to this immutable numerical preconditioner.
    ///
    /// Check the requested-array limit before fallible reservation. Actual
    /// capacity, allocator overhead and opaque local internals are separate;
    /// this limit is not a complete retained-memory or construction-peak budget.
    pub fn try_serial_workspace(
        &self,
        max_requested_payload_bytes: usize,
    ) -> Result<SerialPreconditionerWorkspace<'_>, SolveError> {
        let inner = match &self.inner {
            Variant::Additive(p) => {
                SerialVariant::Additive(p.inner.try_serial_workspace(max_requested_payload_bytes)?)
            }
            Variant::Diagonal(p) => SerialVariant::Diagonal(p),
        };
        Ok(SerialPreconditionerWorkspace { inner })
    }
}

impl SerialPreconditionerWorkspace<'_> {
    /// Dimension of the bound square preconditioner.
    pub fn dimension(&self) -> usize {
        match &self.inner {
            SerialVariant::Additive(p) => p.dimension(),
            SerialVariant::Diagonal(p) => p.nrows(),
        }
    }

    /// Actual retained outer-array capacity; excludes owner and local-factor state.
    pub fn retained_payload_bytes(&self) -> usize {
        match &self.inner {
            SerialVariant::Additive(p) => p.retained_payload_bytes(),
            SerialVariant::Diagonal(_) => 0,
        }
    }

    /// Apply the existing numerical action serially, publishing output only on success.
    ///
    /// Static shape errors leave output and scratch untouched. Local numerical
    /// errors may dirty scratch but leave output untouched. A successful action
    /// still requires the caller's finite-value/original-operator checks.
    pub fn apply(&mut self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let n = self.dimension();
        if x.len() != n || y.len() != n {
            return Err(SolveError::SerialWorkspaceShape {
                rhs: x.len(),
                output: y.len(),
                expected: n,
            });
        }
        match &mut self.inner {
            SerialVariant::Additive(p) => p.apply(x, y),
            SerialVariant::Diagonal(p) => Operator::apply(*p, x, y),
        }
    }
}

impl OperatorMut for SerialPreconditionerWorkspace<'_> {
    fn nrows(&self) -> usize {
        self.dimension()
    }
    fn ncols(&self) -> usize {
        self.dimension()
    }
    fn apply(&mut self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        SerialPreconditionerWorkspace::apply(self, x, y)
    }
    fn apply_adjoint(&mut self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        SerialPreconditionerWorkspace::apply(self, x, y)
    }
}
