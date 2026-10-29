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
}

/// A completed controlled attempt, including its reason for stopping.
#[derive(Debug, Clone)]
#[must_use]
pub struct ControlledSolveResult {
    /// Total coefficients, demeaned response, convergence and work for this attempt.
    pub solution: SolveResult,
    /// A policy-requested stop is `Escalated`; it does not claim convergence.
    pub stop_reason: LsmrStopReason,
}

/// Failed controlled attempt, including work to debit from a shared budget.
pub type ControlledSolveError = LsmrFailure;

impl Solver<'_> {
    /// Solve one RHS with caller-controlled warm starts and early stopping.
    ///
    /// Requires an explicit fixed preconditioner (`Off`, `Diagonal`, `Additive`
    /// or a fixed prebuilt map); an adaptive solver is rejected. Callers can
    /// run RHSs in parallel and build a new solver between attempts, debiting
    /// [`SolveResult::iterations`] or [`ControlledSolveError::iterations`].
    /// Initialization spends zero iterations; a failed iteration spends one;
    /// residual audits and internal false-convergence restarts retain the count.
    ///
    /// Always pass the original `y`, including when correcting coefficients
    /// after an independent validation failure. The low-level solve forms the
    /// seed residual but retains the original response's convergence references
    /// (falling back to a correction reference when an original reference is
    /// zero or unusable, as in [`schwarz_precond::mlsmr`]).
    /// Returned `x` is the total seed plus correction, never the correction alone.
    /// With `Off`, a controlled warm/early-stop solve uses the equivalent
    /// identity-metric modified LSMR; a cold call without controls uses ordinary LSMR.
    pub fn solve_controlled<'o>(
        &self,
        y: &[f64],
        lsmr: impl Into<Option<&'o LsmrOptions>>,
        control: SolveControl<'_>,
    ) -> Result<ControlledSolveResult, ControlledSolveError> {
        let t_start = Instant::now();
        checkpoint(control.cancellation, 0)?;
        let default = LsmrOptions::default();
        let lsmr = lsmr.into().unwrap_or(&default);
        checkpoint(control.cancellation, 0)?;
        let map = match &self.slot {
            PrecondSlot::Static { map, .. } => map.as_ref(),
            PrecondSlot::Adaptive(_) => {
                return Err(invalid(
                    "controlled solving requires a fixed preconditioner".into(),
                ))
            }
        };
        let prepared = self.prepare(y);
        checkpoint(control.cancellation, 0)?;
        let rhs = prepared.map_err(|source| LsmrFailure {
            source,
            iterations: 0,
        })?;
        let seed = control
            .initial_coefficients
            .map(|x| self.prepare_seed(x))
            .transpose();
        checkpoint(control.cancellation, 0)?;
        let seed = seed?;
        let setup_secs = t_start.elapsed().as_secs_f64();
        let options = MlsmrOptions {
            warm_start: seed.as_deref(),
            escalation: control.escalation,
            local_size: lsmr.local_size,
        };
        let t_solve = Instant::now();
        let result = match map {
            Some(map) => mlsmr_counted_cancellable(
                &rhs.op,
                rhs.b(),
                map,
                lsmr.tol,
                lsmr.maxiter,
                options,
                control.cancellation,
            ),
            None if seed.is_none() && control.escalation.is_none() => lsmr_counted_cancellable(
                &rhs.op,
                rhs.b(),
                lsmr.tol,
                lsmr.maxiter,
                lsmr.local_size,
                control.cancellation,
            ),
            None => mlsmr_counted_cancellable(
                &rhs.op,
                rhs.b(),
                &Identity(self.n_dofs()),
                lsmr.tol,
                lsmr.maxiter,
                options,
                control.cancellation,
            ),
        }?;
        let stop_reason = result.stop_reason;
        let solution = self.finish_with_residual_output(
            rhs,
            Run {
                result,
                solve_secs: t_solve.elapsed().as_secs_f64(),
            },
            control.residual_output,
        );
        let result = ControlledSolveResult {
            stop_reason,
            solution: SolveResult {
                x: solution.x,
                unidentified: self.unidentified(),
                warnings: self.warnings().to_vec(),
                layout: CoefficientLayout::from_design(&self.prepared.design),
                demeaned: solution.demeaned,
                converged: solution.converged,
                iterations: solution.iterations,
                residual: solution.residual,
                time_total: t_start.elapsed().as_secs_f64(),
                time_setup: setup_secs,
                time_solve: solution.time_solve,
            },
        };
        checkpoint(control.cancellation, result.solution.iterations)?;
        Ok(result)
    }

    fn prepare_seed(&self, x: &[f64]) -> Result<Vec<f64>, ControlledSolveError> {
        if x.len() != self.n_dofs() {
            return Err(invalid(format!(
                "initial coefficient length {} does not match {}",
                x.len(),
                self.n_dofs()
            )));
        }
        if let Some(index) = x.iter().position(|x| !x.is_finite()) {
            return Err(invalid(format!(
                "initial coefficient {index} must be finite"
            )));
        }
        for position in self.prepared.unidentified() {
            let term = &self.prepared.design.terms[position.channel.term];
            let index = term.column_dofs(position.channel.column).start + position.level;
            if x[index] != 0.0 {
                return Err(invalid(format!(
                    "initial coefficient {index} is unidentified and must be zero"
                )));
            }
        }
        let mut seed = x.to_vec();
        self.prepared.forward_transform(&mut seed);
        if let Some(index) = seed.iter().position(|x| !x.is_finite()) {
            return Err(invalid(format!(
                "initial coefficient transform {index} is not finite"
            )));
        }
        Ok(seed)
    }
}

fn checkpoint(
    cancellation: Cancellation<'_>,
    iterations: usize,
) -> Result<(), ControlledSolveError> {
    solve_checkpoint(cancellation).map_err(|source| LsmrFailure { source, iterations })
}

fn invalid(message: String) -> ControlledSolveError {
    LsmrFailure {
        source: SolveError::InvalidInput {
            context: "Solver::solve_controlled",
            message,
        },
        iterations: 0,
    }
}

struct Identity(usize);
impl Operator for Identity {
    fn nrows(&self) -> usize {
        self.0
    }
    fn ncols(&self) -> usize {
        self.0
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y.copy_from_slice(x);
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}
