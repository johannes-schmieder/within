use std::borrow::Cow;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use portable_atomic::AtomicF64;
use schwarz_precond::Operator;

use crate::domain::Design;

mod fused;
mod gather;
mod scatter;
mod stable;
pub(crate) use fused::FusedPlan;
pub(crate) use stable::StableAdjoint;

#[cfg(test)]
mod tests;

pub(crate) use gather::gather_apply;
use scatter::scatter_apply;

/// Minimum number of rows before scatter/gather loops are parallelized.
const PAR_THRESHOLD: usize = 10_000;

/// Fused-step state: the lane plan and its reusable lane buffers.
struct Fused<'a> {
    plan: FusedPlan,
    /// Ownership index for the plan's separate terms.
    owners: &'a StableAdjoint,
    buffers: Mutex<Vec<f64>>,
}

/// Design operator `D` or `W^{1/2} D`, whose normal equations `AᵀA = DᵀWD` recover the Gramian.
pub(crate) struct DesignOperator<'a> {
    design: &'a Design<'a>,
    sqrt_weights: Option<&'a [f64]>,
    /// Sized once to the largest term's block, so it allocates per operator, not per iteration.
    scatter_scratch: Vec<AtomicF64>,
    stable: Option<&'a StableAdjoint>,
    /// Fused kernel: one-pass steps and lane-ordered adjoints, without atomics.
    fused: Option<Fused<'a>>,
    /// Debug-only reentry sentinel: a concurrent `apply_adjoint` would race the scratch writes.
    #[cfg(debug_assertions)]
    adjoint_active: AtomicBool,
}

impl<'a> DesignOperator<'a> {
    /// The separate kernel without an ownership index; `sqrt_weights` must be
    /// pre-square-rooted and `design.n_obs` long.
    #[cfg(test)]
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
        Self::with_stable(design, sqrt_weights, None)
    }

    pub(crate) fn with_stable(
        design: &'a Design<'a>,
        sqrt_weights: Option<&'a [f64]>,
        stable: Option<&'a StableAdjoint>,
    ) -> Self {
        let max_block = design.terms.iter().map(|t| t.n_dofs()).max().unwrap_or(0);
        Self {
            design,
            sqrt_weights,
            scatter_scratch: if stable.is_some() {
                Vec::new()
            } else {
                (0..max_block).map(|_| AtomicF64::new(0.0)).collect()
            },
            stable,
            fused: None,
            #[cfg(debug_assertions)]
            adjoint_active: AtomicBool::new(false),
        }
    }

    /// The fused kernel: every adjoint follows the lane order of `plan`, and
    /// `owners` sums the plan's separate terms per coefficient in row order.
    pub(crate) fn fused(
        design: &'a Design<'a>,
        sqrt_weights: Option<&'a [f64]>,
        plan: FusedPlan,
        owners: &'a StableAdjoint,
    ) -> Self {
        if let Some(sw) = sqrt_weights {
            assert_eq!(sw.len(), design.n_obs, "sqrt-weights length mismatch");
        }
        Self {
            design,
            sqrt_weights,
            scatter_scratch: Vec::new(),
            stable: None,
            fused: Some(Fused {
                plan,
                owners,
                buffers: Mutex::new(Vec::new()),
            }),
            #[cfg(debug_assertions)]
            adjoint_active: AtomicBool::new(false),
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

/// RAII reentry guard; `Drop` clears the flag on every exit path including panics.
#[cfg(debug_assertions)]
struct ReentryGuard<'a>(&'a AtomicBool);

#[cfg(debug_assertions)]
impl<'a> ReentryGuard<'a> {
    fn acquire(active: &'a AtomicBool) -> Self {
        let already_in_flight = active.swap(true, Ordering::AcqRel);
        debug_assert!(
            !already_in_flight,
            "DesignOperator::apply_adjoint entered concurrently on one operator; \
             its shared scatter buffer is sound for only one in-flight call"
        );
        Self(active)
    }
}

#[cfg(debug_assertions)]
impl Drop for ReentryGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
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
        #[cfg(debug_assertions)]
        let _guard = ReentryGuard::acquire(&self.adjoint_active);
        debug_assert_eq!(x.len(), self.design.n_obs);
        debug_assert_eq!(y.len(), self.design.n_dofs);
        if let Some(fused) = &self.fused {
            let mut buffers = fused.buffers.lock().unwrap_or_else(PoisonError::into_inner);
            return fused::fused_adjoint(
                self.design,
                &fused.plan,
                &mut buffers,
                fused.owners,
                self.sqrt_weights,
                x,
                y,
            );
        }
        y.fill(0.0);
        if let Some(stable) = self.stable {
            stable.apply(self.design, self.sqrt_weights, x, y);
            return Ok(());
        }
        // No lock needed: `solve_batch` builds one operator per RHS, so calls are sequential.
        match self.sqrt_weights {
            Some(sw) => scatter_apply(self.design, &self.scatter_scratch, y, &|i| sw[i] * x[i]),
            None => scatter_apply(self.design, &self.scatter_scratch, y, &|i| x[i]),
        }
        Ok(())
    }

    fn apply_fused_update(
        &self,
        x: &[f64],
        c: f64,
        u: &mut [f64],
        y: &mut [f64],
    ) -> Option<Result<f64, schwarz_precond::SolveError>> {
        let fused = self.fused.as_ref()?;
        #[cfg(debug_assertions)]
        let _guard = ReentryGuard::acquire(&self.adjoint_active);
        let mut buffers = fused.buffers.lock().unwrap_or_else(PoisonError::into_inner);
        Some(fused::fused_update(
            self.design,
            &fused.plan,
            &mut buffers,
            fused.owners,
            self.sqrt_weights,
            x,
            c,
            u,
            y,
        ))
    }
}

// The guard's flag is private and debug-gated, so this test lives beside it.
#[cfg(all(test, debug_assertions))]
mod reentry_guard_tests {
    use std::sync::atomic::Ordering;

    use schwarz_precond::Operator;

    use super::DesignOperator;
    use crate::domain::Design;
    use crate::observation::ObservationFrame;

    fn one_factor_design() -> Design<'static> {
        let frame = ObservationFrame::new(
            vec![vec![0u32, 1, 0]].into_iter().map(Into::into).collect(),
            Vec::new(),
        )
        .expect("valid frame");
        Design::from_frame(frame).expect("valid design")
    }

    #[test]
    #[should_panic(expected = "concurrently")]
    fn apply_adjoint_detects_in_flight_reentry() {
        let design = one_factor_design();
        let op = DesignOperator::new(&design, None);
        // Simulate a sibling `apply_adjoint` already in flight on this operator.
        op.adjoint_active.store(true, Ordering::Release);
        op.apply_adjoint(
            &vec![0.0; op.design.n_obs],
            &mut vec![0.0; op.design.n_dofs],
        )
        .expect("unreachable: the guard panics before returning");
    }
}
