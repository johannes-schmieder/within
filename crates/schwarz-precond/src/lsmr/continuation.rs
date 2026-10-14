//! Explicit suspension keeps every recurrence vector, rotation and MGS window.

use super::bidiag::{Bidiagonalization, ModifiedGolubKahan, ModifiedGolubKahanState};
use super::recurrence::{
    ConvergenceCriteria, ConvergenceState, LsmrRecurrenceState, RotationStep, SolutionState, Stop,
};
use super::{
    invalid_input, validate_lsmr_inputs, vec_norm, EscalationPolicy, LsmrResult, LsmrStopReason,
    Progress,
};
use crate::{Operator, SolveError};

/// An owned modified-LSMR recurrence that can pause without a residual restart.
/// Repeated `advance` calls MUST receive the same mathematical operator and
/// fixed preconditioner. To change either, discard this object and construct a
/// new one with the latest coefficients as its warm start. Both references are
/// tied to the original RHS, including after an explicit residual correction.
pub struct MlsmrContinuation {
    bidiag: Option<ModifiedGolubKahanState>,
    convergence: ConvergenceState,
    recurrence: LsmrRecurrenceState,
    solution: SolutionState,
    prev_rot: RotationStep,
    rhs: Vec<f64>,
    offset: Option<Vec<f64>>,
    completed: usize,
    started: usize,
    terminal: Option<LsmrResult>,
}

impl MlsmrContinuation {
    /// Initialize a fresh recurrence, discarding all state from any earlier one.
    /// `warm_start` is in original coefficient coordinates and `b` is always the
    /// original RHS, never a caller-computed correction RHS.
    pub fn new<A: Operator + ?Sized, M: Operator + ?Sized>(
        operator: &A,
        b: &[f64],
        preconditioner: &M,
        tol: f64,
        warm_start: Option<&[f64]>,
        local_size: Option<usize>,
    ) -> Result<Self, SolveError> {
        validate_lsmr_inputs(operator, b, tol)?;
        let n = operator.ncols();
        if preconditioner.nrows() != n || preconditioner.ncols() != n {
            return Err(invalid_input(
                "preconditioner shape differs from operator columns".into(),
            ));
        }
        if warm_start.is_some_and(|x| x.len() != n || x.iter().any(|x| !x.is_finite())) {
            return Err(invalid_input(
                "warm start must contain ncols finite entries".into(),
            ));
        }
        let b_norm = vec_norm(b);
        let mut rhs = b.to_vec();
        if let Some(x) = warm_start {
            operator.apply(x, &mut rhs)?;
            for (r, b) in rhs.iter_mut().zip(b) {
                *r = b - *r;
            }
        }
        let rhs_norm = vec_norm(&rhs);
        if !rhs_norm.is_finite() {
            return Err(invalid_input(
                "warm-start residual has nonfinite norm".into(),
            ));
        }
        let reference = if warm_start.is_some() && rhs_norm > 0.0 {
            let (_, initial) = ModifiedGolubKahan::init(operator, preconditioner, b, 0)?;
            Some((initial.alpha * initial.beta).abs().max(f64::MIN_POSITIVE))
        } else {
            None
        };
        let (bidiag, step) =
            ModifiedGolubKahan::init(operator, preconditioner, &rhs, local_size.unwrap_or(0))?;
        let convergence = ConvergenceCriteria::new(b_norm, tol).start(step.alpha);
        let mut recurrence = LsmrRecurrenceState::init(step);
        if let Some(reference) = reference {
            recurrence.zeta0 = reference;
        }
        let solution = SolutionState::init(bidiag.v());
        let terminal = if rhs_norm == 0.0 || step.alpha == 0.0 {
            Some(LsmrResult {
                x: warm_start.map_or_else(|| vec![0.0; n], |x| x.to_vec()),
                converged: true,
                iterations: 0,
                residual_norm: rhs_norm,
                normal_eq_residual: 0.0,
                stop_reason: if rhs_norm > 0.0 {
                    LsmrStopReason::InitialNormalEquationResidualZero
                } else if warm_start.is_some() {
                    LsmrStopReason::WarmStartExact
                } else {
                    LsmrStopReason::ZeroRhs
                },
            })
        } else {
            None
        };
        Ok(Self {
            bidiag: Some(bidiag.suspend()),
            convergence,
            recurrence,
            solution,
            prev_rot: RotationStep::initial(),
            rhs,
            offset: warm_start.map(<[f64]>::to_vec),
            completed: 0,
            started: 0,
            terminal,
        })
    }

    /// Number of steps started, including a step that failed partway through.
    /// Initialization and terminal certification do not start additional steps.
    pub fn iterations_started(&self) -> usize {
        self.started
    }

    /// Advance at most `maxiter` additional steps. A cap or escalation pause
    /// retains all recurrence state exactly. A tolerance audit is terminal,
    /// because certification consumes the bidiagonalization scratch vectors.
    /// Escalation handlers are scoped to each call; use `None` or a stateless
    /// policy when stopping decisions must also match an uninterrupted call.
    pub fn advance<A: Operator + ?Sized, M: Operator + ?Sized>(
        &mut self,
        operator: &A,
        preconditioner: &M,
        maxiter: usize,
        escalation: Option<&dyn EscalationPolicy>,
    ) -> Result<LsmrResult, SolveError> {
        if let Some(result) = &self.terminal {
            let mut result = result.clone();
            result.iterations = 0;
            return Ok(result);
        }
        let state = self.bidiag.take().ok_or_else(|| {
            invalid_input("continuation is terminal after an operator failure".into())
        })?;
        let mut bidiag = ModifiedGolubKahan::resume(operator, preconditioner, state);
        let mut handler = escalation.map(|p| p.handler());
        let mut iterations = 0;
        let mut reason = LsmrStopReason::MaxIterations;
        for _ in 0..maxiter {
            self.started += 1;
            let step = bidiag.step()?;
            self.convergence.observe(step);
            let rotation = self.recurrence.step(step);
            self.solution.update(bidiag.v(), rotation, self.prev_rot);
            self.prev_rot = rotation;
            self.completed += 1;
            iterations += 1;
            let stop = match self.convergence.check(&self.recurrence) {
                Stop::Continue => None,
                Stop::ResidualTolerance => Some(LsmrStopReason::ResidualTolerance),
                Stop::NormalEquationTolerance => Some(LsmrStopReason::NormalEquationTolerance),
            };
            if let Some(stop) = stop {
                let cert = bidiag.certify(self.solution.x(), &self.rhs)?;
                let converged = self.convergence.certified(&cert, self.recurrence.zeta0);
                let result = LsmrResult {
                    x: self.coefficients(),
                    converged,
                    iterations,
                    residual_norm: cert.normr,
                    normal_eq_residual: cert.normar / self.recurrence.zeta0,
                    stop_reason: if converged {
                        stop
                    } else {
                        LsmrStopReason::FalseConvergence
                    },
                };
                self.terminal = Some(result.clone());
                return Ok(result);
            }
            if handler.as_deref_mut().is_some_and(|rule| {
                rule.should_escalate(Progress {
                    iteration: self.completed,
                    normal_eq_residual: self.recurrence.relative_normal_eq_residual(),
                })
            }) {
                reason = LsmrStopReason::Escalated;
                break;
            }
        }
        self.bidiag = Some(bidiag.suspend());
        Ok(LsmrResult {
            x: self.coefficients(),
            converged: false,
            iterations,
            residual_norm: self.recurrence.residual_estimate(),
            normal_eq_residual: self.recurrence.relative_normal_eq_residual(),
            stop_reason: reason,
        })
    }

    fn coefficients(&self) -> Vec<f64> {
        let mut x = self.solution.x().to_vec();
        if let Some(offset) = &self.offset {
            for (x, x0) in x.iter_mut().zip(offset) {
                *x += x0;
            }
        }
        x
    }
}
