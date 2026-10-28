//! Restartable solves keep original RHS scales and coefficients in the solve basis.

use std::sync::Arc;
use std::time::Instant;

use schwarz_precond::{EscalationPolicy, LsmrStopReason, MlsmrContinuation, Operator};

use super::{CoefficientLayout, SolveResult, Solver};
use crate::operator::design::gather_apply;
use crate::operator::DesignOperator;
use crate::{LsmrOptions, PairReplacement, SolveError};

/// Result of one bounded stretch of Krylov work within a solve session.
#[derive(Clone, Copy, Debug)]
pub struct ActionProgress {
    /// Iterations consumed by this action.
    pub iterations: usize,
    /// Total iterations consumed since the original RHS was submitted.
    pub total_iterations: usize,
    /// Whether this action met the iterative stopping criteria.
    pub converged: bool,
    /// Normal-equation residual relative to the original RHS under this action's
    /// preconditioner. Values from different preconditioners are not comparable.
    pub residual: f64,
    /// Why this recurrence stopped.
    pub stop_reason: LsmrStopReason,
}

/// A single original RHS, coefficient iterate and total iteration budget.
/// Actions using the same preconditioner resume the unchanged recurrence. A
/// replacement or explicit correction discards the recurrence and recomputes
/// `b - A x`. Coefficients remain in the internal solve basis;
/// slope and observation-order recovery operate only on output copies.
pub struct SolveSession<'s, 'a> {
    solver: &'s Solver<'a>,
    /// Response in internal observation order, shared with an unseeded recurrence.
    y: Arc<[f64]>,
    /// `W^{1/2} y` when weighted; unit weights solve against `y` itself.
    b: Option<Arc<[f64]>>,
    x: Vec<f64>,
    options: LsmrOptions,
    progress: ActionProgress,
    started: Instant,
    time_setup: f64,
    time_solve: f64,
    failed: bool,
    ran: bool,
    continuation: Option<MlsmrContinuation>,
    replacement_identity: Option<std::sync::Arc<()>>,
}

impl<'a> Solver<'a> {
    /// Begin an optional restartable solve. Existing `solve` and `solve_batch`
    /// keep their established execution and serialization behavior.
    pub fn start_session(
        &self,
        y: &[f64],
        options: &LsmrOptions,
    ) -> Result<SolveSession<'_, 'a>, SolveError> {
        let started = Instant::now();
        if y.len() != self.n_obs() || y.iter().any(|value| !value.is_finite()) {
            return Err(invalid("response must have n_obs finite entries"));
        }
        if !options.tol.is_finite() || options.tol < 0.0 {
            return Err(invalid("tolerance must be finite and nonnegative"));
        }
        let y = self.design.permute_obs_in_shared(y);
        // Same products as `DesignOperator::weighted_rhs`, collected once.
        let b = self.sqrt_weights.as_deref().map(|sw| {
            y.iter()
                .zip(sw)
                .map(|(&yi, &swi)| swi * yi)
                .collect::<Arc<[f64]>>()
        });
        Ok(SolveSession {
            solver: self,
            y,
            b,
            x: vec![0.0; self.n_dofs()],
            options: options.clone(),
            progress: ActionProgress {
                iterations: 0,
                total_iterations: 0,
                converged: false,
                residual: 1.0,
                stop_reason: LsmrStopReason::MaxIterations,
            },
            started,
            time_setup: started.elapsed().as_secs_f64(),
            time_solve: 0.0,
            failed: false,
            ran: false,
            continuation: None,
            replacement_identity: None,
        })
    }
    /// Begin a categorical solve at finite, full coefficient coordinates in the
    /// existing term-major/original-level layout. The original response and
    /// weighted RHS remain the stopping references; the first action recomputes
    /// the exact correction residual. No Krylov iterations are consumed here.
    /// The caller must deduct any prefix iteration work from `options.maxiter`.
    /// Slope-bearing designs are rejected, so no solve-basis conversion or gauge
    /// projection is implicit in this interface.
    pub fn start_session_categorical_seeded(
        &self,
        y: &[f64],
        initial_coefficients: &[f64],
        options: &LsmrOptions,
    ) -> Result<SolveSession<'_, 'a>, SolveError> {
        if self.reparam.is_some()
            || self.design.terms.iter().any(|term| {
                term.columns.len() != 1 || term.columns.iter().any(|c| c.covariate().is_some())
            })
        {
            return Err(invalid(
                "categorical seed requires one constant column per term",
            ));
        }
        if initial_coefficients.len() != self.n_dofs()
            || initial_coefficients.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("categorical seed must have n_dofs finite entries"));
        }
        let mut session = self.start_session(y, options)?;
        session.x.copy_from_slice(initial_coefficients);
        session.ran = true;
        Ok(session)
    }
}

impl SolveSession<'_, '_> {
    /// Iterations still available from the original solve budget, including after
    /// an error. A partially executed failed step consumes one iteration.
    pub fn remaining_iterations(&self) -> usize {
        self.options
            .maxiter
            .saturating_sub(self.progress.total_iterations)
    }

    /// Continue with the incumbent (`None`) or a replacement built by this exact
    /// solver. The same backend retains every recurrence vector and rotation, so
    /// a rejected setup attempt can resume bit-for-bit. Changing backend discards
    /// the old recurrence and recomputes the residual. The action cap cannot
    /// enlarge the original budget.
    /// An operator failure leaves the previous iterate intact and makes the
    /// session terminal; it cannot launder unknown partial work into a new budget.
    pub fn run_action(
        &mut self,
        replacement: Option<&PairReplacement>,
        iteration_cap: usize,
        escalation: Option<&dyn EscalationPolicy>,
    ) -> Result<ActionProgress, SolveError> {
        if self.failed {
            return Err(invalid("session is terminal after an operator failure"));
        }
        if replacement.is_some_and(|p| {
            !self
                .solver
                .replacement_source
                .get()
                .is_some_and(|source| std::sync::Arc::ptr_eq(source, &p.source))
        }) {
            return Err(invalid(
                "replacement belongs to a different solver topology or weights",
            ));
        }
        let cap = iteration_cap.min(self.remaining_iterations());
        if cap == 0 {
            return Ok(ActionProgress {
                iterations: 0,
                ..self.progress
            });
        }
        let operator = DesignOperator::with_stable(
            &self.solver.design,
            self.solver.sqrt_weights.as_deref(),
            self.solver.stable_adjoint.as_ref(),
        );
        let identity = Identity(self.x.len());
        let preconditioner: &dyn Operator = match replacement {
            Some(p) => &p.preconditioner,
            None => self
                .solver
                .preconditioner
                .as_ref()
                .map_or(&identity as &dyn Operator, |p| p),
        };
        let changed = match (&self.replacement_identity, replacement) {
            (None, None) => false,
            (Some(previous), Some(current)) => !std::sync::Arc::ptr_eq(previous, &current.identity),
            _ => true,
        };
        if changed {
            self.continuation = None;
        }
        let started = Instant::now();
        let previously_started = self
            .continuation
            .as_ref()
            .map_or(0, MlsmrContinuation::iterations_started);
        let attempt = (|| {
            if self.continuation.is_none() {
                let b = Arc::clone(self.b.as_ref().unwrap_or(&self.y));
                self.continuation = Some(MlsmrContinuation::new_shared(
                    &operator,
                    b,
                    preconditioner,
                    self.options.tol,
                    self.ran.then_some(self.x.as_slice()),
                    self.options.local_size,
                )?);
            }
            self.continuation
                .as_mut()
                .expect("initialized continuation")
                .advance(&operator, preconditioner, cap, escalation)
        })();
        self.time_solve += started.elapsed().as_secs_f64();
        let consumed = self
            .continuation
            .as_ref()
            .map_or(0, MlsmrContinuation::iterations_started)
            .saturating_sub(previously_started);
        self.progress.iterations = consumed;
        self.progress.total_iterations += consumed;
        let result = match attempt {
            Ok(result) => result,
            Err(error) => {
                self.failed = true;
                return Err(error);
            }
        };
        self.replacement_identity = replacement.map(|p| p.identity.clone());
        self.x = result.x;
        self.ran = true;
        self.progress = ActionProgress {
            iterations: result.iterations,
            total_iterations: self.progress.total_iterations,
            converged: result.converged,
            residual: result.normal_eq_residual,
            stop_reason: result.stop_reason,
        };
        Ok(self.progress)
    }

    /// Discard the full recurrence for a residual correction at an equal or
    /// tighter tolerance. The original RHS, coefficient gauge and remaining
    /// iteration budget are preserved. No iterations run until `run_action`.
    pub fn restart_correction(&mut self, tolerance: f64) -> Result<(), SolveError> {
        if !tolerance.is_finite() || tolerance < 0.0 || tolerance > self.options.tol {
            return Err(invalid(
                "correction tolerance must be finite, nonnegative and no looser",
            ));
        }
        self.options.tol = tolerance;
        self.continuation = None;
        Ok(())
    }

    /// Recover coefficients and unweighted observation residuals from the current
    /// iterate. A failed session is explicitly marked unconverged.
    pub fn finish(mut self) -> SolveResult {
        self.continuation = None;
        self.b = None;
        let x = std::mem::take(&mut self.x);
        self.result(x)
    }

    /// Recover a reviewable copy of the current iterate without discarding any
    /// recurrence state. Allocates coefficient and observation output vectors.
    pub fn snapshot(&self) -> SolveResult {
        self.result(self.x.clone())
    }

    fn result(&self, mut x: Vec<f64>) -> SolveResult {
        let mut demeaned = vec![0.0; self.solver.n_obs()];
        gather_apply(&self.solver.design, &x, &mut demeaned, None);
        for (value, &response) in demeaned.iter_mut().zip(self.y.iter()) {
            *value = response - *value;
        }
        if let Some(reparam) = &self.solver.reparam {
            reparam.back_transform(&mut x);
        }
        SolveResult {
            x,
            unidentified: self.solver.unidentified(),
            warnings: self.solver.warnings.clone(),
            layout: CoefficientLayout::from_design(&self.solver.design),
            demeaned: self.solver.design.permute_obs_out(demeaned),
            converged: self.progress.converged && !self.failed,
            iterations: self.progress.total_iterations,
            residual: self.progress.residual,
            time_total: self.started.elapsed().as_secs_f64(),
            time_setup: self.time_setup,
            time_solve: self.time_solve,
        }
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

fn invalid(message: &str) -> SolveError {
    SolveError::InvalidInput {
        context: "within.solve_session",
        message: message.into(),
    }
}

#[cfg(test)]
mod tests;
