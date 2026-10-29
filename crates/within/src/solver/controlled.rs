//! Caller-owned staging with the existing low-level warm-start and stop policy.

use super::solve_checkpoint;
use super::{CoefficientLayout, PrecondSlot, Run, SolveResult, Solver};
use crate::{Cancellation, EscalationPolicy, LsmrOptions, LsmrStopReason, SolveError};
use schwarz_precond::{
    lsmr_counted_cancellable, mlsmr_counted_cancellable, LsmrFailure, MlsmrOptions, Operator,
};
use std::time::Instant;

/// Arithmetic used to construct the returned observation residual.
///
/// This only selects the output calculation. It does not change the solver's
/// weighted residual audits, coefficients, stopping decision, or iteration count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResidualOutput {
    /// Reuse the measured weighted residual, dividing by square-root weights.
    ///
    /// This is the existing output calculation. An unweighted solve reuses its
    /// measured residual directly; zero-weight rows require model reconstruction.
    #[default]
    WeightedAudit,
    /// Reconstruct the original response minus the unweighted prepared model.
    ///
    /// Applies the prepared design to the total internal coefficients before
    /// coefficient back-transformation, then subtracts from the original response.
    /// This avoids undoing square-root weights in the output, which can round
    /// differently at very small positive weights. Whitened slopes and dropped
    /// directions remain those of the prepared model; this does not promise
    /// bitwise equality with raw slopes multiplied by the returned coefficients.
    /// The measured residual allocation is reused when available.
    PreparedModel,
}

/// Controls for one fixed-preconditioner attempt. No Krylov state is retained.
#[derive(Clone, Copy, Default)]
pub struct SolveControl<'a> {
    /// Initial *total* coefficients, in exactly the coordinates of [`SolveResult::x`].
    ///
    /// Term-major compact levels, original (unwhitened) slope units, and no
    /// observation weighting or locality permutation. The seed may carry a
    /// nonunique between-term gauge; it is retained in the returned total.
    /// Slots listed in [`SolveResult::unidentified`] must be zero: the solve
    /// basis cannot represent those directions, so nonzero values are rejected.
    pub initial_coefficients: Option<&'a [f64]>,
    /// Per-attempt progress/early-stop policy. A requested stop returns
    /// [`crate::LsmrStopReason::Escalated`] and the total coefficients without
    /// constructing or switching to another preconditioner.
    pub escalation: Option<&'a dyn EscalationPolicy>,
    /// Shared worker-safe cancellation query. Cancellation returns a distinct
    /// error with consumed iterations and discards all partial coefficients.
    pub cancellation: Cancellation<'a>,
    /// Calculation of [`SolveResult::demeaned`], restored to caller row order.
    pub residual_output: ResidualOutput,
