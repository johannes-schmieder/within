use schwarz_precond::Operator;
use std::borrow::Cow;

use crate::domain::Design;

mod gather;
mod scatter;

#[cfg(test)]
mod tests;

pub(crate) use gather::gather_apply;
use scatter::scatter_apply;

/// Minimum number of rows before scatter/gather loops are parallelized.
const PAR_THRESHOLD: usize = 10_000;

/// Design operator `D` or `W^{1/2} D`, whose normal equations `AᵀA = DᵀWD` recover the Gramian.
pub(crate) struct DesignOperator<'a> {
    design: &'a Design<'a>,
    sqrt_weights: Option<&'a [f64]>,
}

impl<'a> DesignOperator<'a> {
    /// `sqrt_weights` must be pre-square-rooted and `design.n_obs` long.
    pub(crate) fn new(design: &'a Design<'a>, sqrt_weights: Option<&'a [f64]>) -> Self {
        if let Some(sw) = sqrt_weights {
            assert_eq!(
                sw.len(),
                design.n_obs,
                "sqrt-weights length {} does not match design.n_obs {}",
                sw.len(),
                design.n_obs
            );
        }
        Self {
            design,
            sqrt_weights,
        }
    }

    /// Observation-space RHS `b = W^{1/2} y`; borrows unweighted, owns weighted.
    pub(crate) fn weighted_rhs<'y>(&self, y: &'y [f64]) -> Cow<'y, [f64]> {
        match self.sqrt_weights {
            None => Cow::Borrowed(y),
            Some(sw) => Cow::Owned(y.iter().zip(sw).map(|(&yi, &swi)| swi * yi).collect()),
        }
    }
}

impl Operator for DesignOperator<'_> {
    fn nrows(&self) -> usize {
        self.design.n_obs
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.design.n_dofs);
        debug_assert_eq!(y.len(), self.design.n_obs);
        gather_apply(self.design, x, y, self.sqrt_weights);
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.design.n_obs);
        debug_assert_eq!(y.len(), self.design.n_dofs);
        y.fill(0.0);
        match self.sqrt_weights {
            Some(sw) => scatter_apply(self.design, y, &|i| sw[i] * x[i]),
            None => scatter_apply(self.design, y, &|i| x[i]),
        }
        Ok(())
    }
}
