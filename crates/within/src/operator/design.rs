use std::borrow::Cow;

use schwarz_precond::Operator;

use crate::domain::PreparedDesign;
use crate::ResidualOutput;

mod gather;
mod scatter;

#[cfg(test)]
mod tests;

use gather::{gather_apply, unscale};
use scatter::scatter_apply;

/// Minimum number of rows before scatter/gather loops are parallelized.
const PAR_THRESHOLD: usize = 10_000;

/// Design operator `D` or `W^{1/2} D`, whose normal equations `AᵀA = DᵀWD` recover the Gramian.
pub(crate) struct DesignOperator<'a> {
    prepared: &'a PreparedDesign<'a>,
}

impl<'a> DesignOperator<'a> {
    pub(crate) fn new(prepared: &'a PreparedDesign<'a>) -> Self {
        Self { prepared }
    }

    /// Return the measured residual or reconstruct `y − D x` in the prepared frame.
    pub(crate) fn demeaned(
        &self,
        x: &[f64],
        y: &[f64],
        measured: Option<Vec<f64>>,
        output: ResidualOutput,
    ) -> Vec<f64> {
        match (measured, self.prepared.sqrt_weights(), output) {
            (Some(residual), None, ResidualOutput::WeightedAudit) => residual,
            (Some(mut residual), Some(sw), ResidualOutput::WeightedAudit) if !sw.contains(&0.0) => {
                unscale(&mut residual, sw);
                residual
            }
            (measured, _, _) => {
                let mut demeaned = measured.unwrap_or_else(|| vec![0.0; y.len()]);
                gather_apply(self.prepared, x, &mut demeaned, None);
                for (d, &yi) in demeaned.iter_mut().zip(y) {
                    *d = yi - *d;
                }
                demeaned
            }
        }
    }

    /// Observation-space RHS `b = W^{1/2} y`; borrows unweighted, owns weighted.
    pub(crate) fn weighted_rhs<'y>(&self, y: &'y [f64]) -> Cow<'y, [f64]> {
        match self.prepared.sqrt_weights() {
            None => Cow::Borrowed(y),
            Some(sw) => Cow::Owned(y.iter().zip(sw).map(|(&yi, &swi)| swi * yi).collect()),
        }
    }
}

impl Operator for DesignOperator<'_> {
    fn nrows(&self) -> usize {
        self.prepared.design.n_obs
    }

    fn ncols(&self) -> usize {
        self.prepared.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.prepared.design.n_dofs);
        debug_assert_eq!(y.len(), self.prepared.design.n_obs);
        gather_apply(self.prepared, x, y, self.prepared.sqrt_weights());
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.prepared.design.n_obs);
        debug_assert_eq!(y.len(), self.prepared.design.n_dofs);
        match self.prepared.sqrt_weights() {
            Some(sw) => scatter_apply(self.prepared, y, &|i| sw[i] * x[i], None),
            None => scatter_apply(self.prepared, y, &|i| x[i], Some(x)),
        }
        Ok(())
    }
}
