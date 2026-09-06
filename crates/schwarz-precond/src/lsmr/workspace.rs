//! Caller-owned serial modified-LSMR storage.
use super::{
    bidiag::ModifiedGolubKahan, bidiag::ModifiedGolubKahanBuffers, recurrence::SolutionState,
    run_recurrence, vec_norm, ConvergenceCriteria, EscalationHandler, LsmrResult, LsmrStopReason,
    RecurrenceRun,
};
use crate::{OperatorMut, SolveError};

/// Scalar outcome separate from the solution's storage lifetime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LsmrDiagnostics {
    /// Whether the native recurrence/audit reported convergence.
    pub converged: bool,
    /// Number of completed iterations.
    pub iterations: usize,
    /// Native final residual norm or recurrence estimate.
    pub residual_norm: f64,
    /// Native relative normal-equation diagnostic in the preconditioner metric.
    pub normal_eq_residual: f64,
    /// Native termination reason; not an independent caller certificate.
    pub stop_reason: LsmrStopReason,
}
impl LsmrDiagnostics {
    pub(super) fn into_owned(self, x: Vec<f64>) -> LsmrResult {
        LsmrResult {
            x,
            converged: self.converged,
            iterations: self.iterations,
            residual_norm: self.residual_norm,
            normal_eq_residual: self.normal_eq_residual,
            stop_reason: self.stop_reason,
        }
    }
}

/// Borrowed candidate; the workspace cannot be reused while this borrow is live.
#[derive(Debug)]
#[must_use]
pub struct LsmrWorkspaceResult<'workspace> {
    /// Coefficients in workspace storage; copy only when ownership is needed.
    pub x: &'workspace [f64],
    /// Native diagnostics, separate from any caller's original-operator certificate.
    pub diagnostics: LsmrDiagnostics,
}

/// Per-call optional behavior; reorthogonalization capacity belongs to the workspace.
#[derive(Default)]
pub struct MlsmrWorkspaceOptions<'a> {
    /// Residual-correction warm start, with tolerances relative to the original RHS.
    pub warm_start: Option<&'a [f64]>,
    /// Caller-created mutable escalation handler; creation is outside the solve.
    pub escalation: Option<&'a mut dyn EscalationHandler>,
}

/// Reusable modified-LSMR vectors and local reorthogonalization history.
///
/// The workspace is shape-bound, not operator-bound: every run initializes all
/// numerical state it reads, so equal-size operators/weights may change between
/// calls. Operators must remain fixed within a call. No prior operator image,
/// basis, recurrence or solution is reused implicitly. Errors may clobber scratch
/// but do not publish a candidate; the workspace can be used again after an error.
///
/// Storage is six coefficient vectors, three observation vectors (including
/// optional warm-start scratch), and two windows of `min(local_size, m, n)`
/// coefficient vectors. No full iteration trace or unbounded history is retained.
/// Construction is fallible; successful serial solves allocate no internal storage.
/// Operator actions and user escalation handlers control their own allocations.
pub struct MlsmrWorkspace {
    rows: usize,
    cols: usize,
    local_size: usize,
    bidiag: ModifiedGolubKahanBuffers,
    solution: SolutionState,
    warm_rhs: Vec<f64>,
}
impl MlsmrWorkspace {
    /// Checked requested vector payload before any allocation (excludes inline state).
    pub fn required_payload_bytes(
        rows: usize,
        cols: usize,
        local_size: Option<usize>,
    ) -> Result<usize, SolveError> {
        let cap = local_size.unwrap_or(0).min(rows.min(cols));
        let history = cap
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(2))
            .ok_or(SolveError::WorkspaceSizeOverflow)?;
        cols.checked_mul(6)
            .and_then(|n| rows.checked_mul(3).and_then(|m| n.checked_add(m)))
            .and_then(|n| n.checked_add(history))
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .ok_or(SolveError::WorkspaceSizeOverflow)
    }

    /// Allocate all recurrence, warm-start and reorthogonalization storage once.
    pub fn try_new(
        rows: usize,
        cols: usize,
        local_size: Option<usize>,
    ) -> Result<Self, SolveError> {
        Self::try_new_with(rows, cols, local_size, &mut zeros)
    }

    fn try_new_with<F>(
        rows: usize,
        cols: usize,
        local_size: Option<usize>,
        allocate: &mut F,
    ) -> Result<Self, SolveError>
    where
        F: FnMut(usize) -> Result<Vec<f64>, SolveError>,
    {
        Self::required_payload_bytes(rows, cols, local_size)?;
        let local_size = local_size.unwrap_or(0).min(rows.min(cols));
        Ok(Self {
            rows,
            cols,
            local_size,
            bidiag: ModifiedGolubKahanBuffers::try_new_with(rows, cols, local_size, allocate)?,
            solution: SolutionState::try_new_with(cols, allocate)?,
            warm_rhs: allocate(rows)?,
        })
    }
    /// Observation-space dimension.
    pub fn nrows(&self) -> usize {
        self.rows
    }
    /// Coefficient-space dimension.
    pub fn ncols(&self) -> usize {
        self.cols
    }
    /// Actual bounded local reorthogonalization window.
    pub fn local_size(&self) -> usize {
        self.local_size
    }
    /// Actual retained vector capacity in bytes, excluding inline state and allocator overhead.
    pub fn retained_payload_bytes(&self) -> Result<usize, SolveError> {
        self.bidiag
            .retained_elements()?
            .checked_add(self.solution.retained_elements()?)
            .and_then(|n| n.checked_add(self.warm_rhs.capacity()))
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .ok_or(SolveError::WorkspaceSizeOverflow)
    }
    pub(super) fn into_x(self) -> Vec<f64> {
        self.solution.into_x()
    }
}

pub(super) fn zeros(count: usize) -> Result<Vec<f64>, SolveError> {
    count
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(SolveError::WorkspaceSizeOverflow)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| SolveError::WorkspaceAllocation)?;
    values.resize(count, 0.0);
    Ok(values)
}

/// Solve modified LSMR with serial internal kernels and caller-owned mutable actions.
///
/// This function never enters a global Rayon pool. The actions themselves may
/// use caller-controlled parallelism. Numerical recurrence and local MGS order
/// match the allocating driver below its legacy parallel threshold. Different
/// execution configurations need numerical comparison, not bitwise equivalence.
///
/// Shape/options validate before scratch changes or actions execute. A result
/// borrows workspace coefficients; call again only after that borrow ends.
/// ```compile_fail
/// use schwarz_precond::{OperatorMut, MlsmrWorkspace, MlsmrWorkspaceOptions, mlsmr_with_workspace};
/// fn invalid_reuse<A: OperatorMut, M: OperatorMut>(a: &mut A, m: &mut M, b: &[f64], w: &mut MlsmrWorkspace) {
///     let first = mlsmr_with_workspace(a, b, m, 1e-8, 100, MlsmrWorkspaceOptions::default(), w).unwrap();
///     let second = mlsmr_with_workspace(a, b, m, 1e-8, 100, MlsmrWorkspaceOptions::default(), w).unwrap();
///     assert_eq!(first.x, second.x);
/// }
/// ```
/// Native convergence remains a candidate requiring the caller's certification.
pub fn mlsmr_with_workspace<'w, A: OperatorMut + ?Sized, M: OperatorMut + ?Sized>(
    operator: &mut A,
    b: &[f64],
    preconditioner: &mut M,
    tol: f64,
    maxiter: usize,
    options: MlsmrWorkspaceOptions<'_>,
    workspace: &'w mut MlsmrWorkspace,
) -> Result<LsmrWorkspaceResult<'w>, SolveError> {
    solve(
        operator,
        b,
        preconditioner,
        tol,
        maxiter,
        options,
        workspace,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn solve<'w, A: OperatorMut + ?Sized, M: OperatorMut + ?Sized>(
    operator: &mut A,
    b: &[f64],
    preconditioner: &mut M,
    tol: f64,
    maxiter: usize,
    options: MlsmrWorkspaceOptions<'_>,
    workspace: &'w mut MlsmrWorkspace,
    parallel: bool,
) -> Result<LsmrWorkspaceResult<'w>, SolveError> {
    if operator.nrows() != workspace.rows || operator.ncols() != workspace.cols {
        return Err(SolveError::WorkspaceMismatch);
    }
    validate_inputs(operator, b, preconditioner, tol, options.warm_start)?;
    let b_norm = vec_norm(b);
    let rhs = if let Some(x0) = options.warm_start {
        operator.apply(x0, &mut workspace.warm_rhs)?;
        for (ri, &bi) in workspace.warm_rhs.iter_mut().zip(b) {
            *ri = bi - *ri;
        }
        &workspace.warm_rhs[..]
    } else {
        b
    };
    let rhs_norm = vec_norm(rhs);
    if !rhs_norm.is_finite() {
        return Err(super::invalid_input(format!(
            "warm-start residual b - A·x0 has non-finite norm {rhs_norm}"
        )));
    }
    let diagnostics = if rhs_norm == 0.0 {
        let stop_reason = if let Some(x0) = options.warm_start {
            workspace.solution.x_mut().copy_from_slice(x0);
            LsmrStopReason::WarmStartExact
        } else {
            workspace.solution.x_mut().fill(0.0);
            LsmrStopReason::ZeroRhs
        };
        LsmrDiagnostics {
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
            normal_eq_residual: 0.0,
            stop_reason,
        }
    } else {
        let (bidiag, step1) = ModifiedGolubKahan::init(
            operator,
            preconditioner,
            rhs,
            &mut workspace.bidiag,
            parallel,
        )?;
        let diagnostics = run_recurrence(
            bidiag,
            &mut workspace.solution,
            RecurrenceRun {
                step1,
                rhs,
                rhs_norm,
                criteria: ConvergenceCriteria::new(b_norm, tol),
                maxiter,
                escalation: options.escalation,
                parallel,
            },
        )?;
        if let Some(x0) = options.warm_start {
            for (xi, &x0i) in workspace.solution.x_mut().iter_mut().zip(x0) {
                *xi += x0i;
            }
        }
        diagnostics
    };
    Ok(LsmrWorkspaceResult {
        x: workspace.solution.x(),
        diagnostics,
    })
}

pub(super) fn validate_inputs<A: OperatorMut + ?Sized, M: OperatorMut + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: &M,
    tol: f64,
    warm_start: Option<&[f64]>,
) -> Result<(), SolveError> {
    let n = operator.ncols();
    if b.len() != operator.nrows() {
        return Err(super::invalid_input(format!(
            "rhs length {} does not match operator row count {}",
            b.len(),
            operator.nrows()
        )));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(super::invalid_input(format!(
            "tolerance must be finite and nonnegative, got {tol}"
        )));
    }
    if let Some((index, value)) = b.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(super::invalid_input(format!(
            "rhs entry {index} must be finite, got {value}"
        )));
    }
    if preconditioner.nrows() != n || preconditioner.ncols() != n {
        return Err(super::invalid_input(format!(
            "preconditioner shape {}x{} must match operator column count {n}",
            preconditioner.nrows(),
            preconditioner.ncols()
        )));
    }
    if let Some(x0) = warm_start {
        if x0.len() != n {
            return Err(super::invalid_input(format!(
                "warm-start length {} does not match operator column count {n}",
                x0.len()
            )));
        }
        if let Some((index, value)) = x0.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(super::invalid_input(format!(
                "warm-start entry {index} must be finite, got {value}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_reservation_failure_returns_no_workspace_and_fresh_construction_recovers() {
        let mut reservations = 0;
        let complete = MlsmrWorkspace::try_new_with(10, 7, Some(8), &mut |n| {
            reservations += 1;
            zeros(n)
        })
        .unwrap();
        assert_eq!(reservations, 11);
        drop(complete);
        for fail_at in 0..reservations {
            let mut seen = 0;
            let result = MlsmrWorkspace::try_new_with(10, 7, Some(8), &mut |n| {
                let index = seen;
                seen += 1;
                if index == fail_at {
                    Err(SolveError::WorkspaceAllocation)
                } else {
                    zeros(n)
                }
            });
            assert!(matches!(result, Err(SolveError::WorkspaceAllocation)));
            assert_eq!(seen, fail_at + 1);
            let fresh = MlsmrWorkspace::try_new(10, 7, Some(8)).unwrap();
            assert_eq!(
                fresh.retained_payload_bytes().unwrap(),
                MlsmrWorkspace::required_payload_bytes(10, 7, Some(8)).unwrap()
            );
        }
    }
}
