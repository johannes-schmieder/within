//! Schwarz domain decomposition preconditioner library.
//!
//! Provides a one-level additive Schwarz preconditioner generic over local
//! solvers, plus a Modified LSMR iterative solver for rectangular
//! least-squares problems. The preconditioner approximates `(AᵀA)^{-1}` by
//! solving overlapping subproblems and stitching them together with a
//! partition-of-unity weighting; each local solve runs in parallel.
//!
//! See `examples/` for runnable usage. Reference: Toselli & Widlund (2005).
//! *Domain Decomposition Methods — Algorithms and Theory*. Springer.
//!
//! [`lsmr`] solves `min ‖b − A x‖₂` for any [`Operator`], with no
//! preconditioner required:
//!
//! ```
//! use schwarz_precond::{lsmr, Operator, SolveError};
//!
//! struct Diag(Vec<f64>);
//! impl Operator for Diag {
//!     fn nrows(&self) -> usize {
//!         self.0.len()
//!     }
//!     fn ncols(&self) -> usize {
//!         self.0.len()
//!     }
//!     fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
//!         for i in 0..self.0.len() {
//!             y[i] = self.0[i] * x[i];
//!         }
//!         Ok(())
//!     }
//!     fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
//!         self.apply(x, y)
//!     }
//! }
//!
//! let a = Diag(vec![2.0, 3.0]);
//! let r = lsmr(&a, &[4.0, 9.0], 1e-10, 50, None).unwrap();
//! assert!(r.converged);
//! assert!((r.x[0] - 2.0).abs() < 1e-6);
//! assert!((r.x[1] - 3.0).abs() < 1e-6);
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

/// A linear operator with its adjoint; both applies are fallible for one uniform `?` path.
pub trait Operator: Send + Sync {
    /// Number of rows in the operator.
    fn nrows(&self) -> usize;
    /// Number of columns in the operator.
    fn ncols(&self) -> usize;
    /// Computes `y = A x`.
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), error::SolveError>;
    /// Computes `y = Aᵀ x`; symmetric operators should delegate to `apply`.
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), error::SolveError>;

    /// Optional fused modified-LSMR step: `u ← A x + c·u`, then `y = Aᵀ u`,
    /// returning `‖u‖²`. `None`, returned before touching `u` or `y`, selects
    /// the separate `apply`, update/norm and `apply_adjoint` passes.
    ///
    /// The updated `u` is used by later steps, so it must equal the separate
    /// update `(A x)_i + c·u_i` in each entry. Computing `‖u‖²` from
    /// [`lsmr_norm_chunk`] partials combined by [`lsmr_norm_reduce`] keeps the
    /// separate path's bits; `y` may differ from `apply_adjoint` by rounding.
    fn apply_fused_update(
        &self,
        x: &[f64],
        c: f64,
        u: &mut [f64],
        y: &mut [f64],
    ) -> Option<Result<f64, error::SolveError>> {
        let _ = (x, c, u, y);
        None
    }
}

// Compiles the README usage example as a doctest so it cannot drift from the public API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

/// Domain decomposition primitives: subdomain cores and partition weights.
pub mod domain;
/// Typed errors for build and runtime failures.
pub mod error;
mod local_solve;
mod lsmr;
mod schwarz;

pub use domain::{PartitionWeights, SubdomainCore};
pub use error::{BuildError, LocalSolveError, SolveError};
pub use local_solve::{LocalSolver, SubdomainEntry};
pub use lsmr::{
    lsmr, lsmr_norm_chunk, lsmr_norm_reduce, mlsmr, EscalationHandler, EscalationPolicy,
    LsmrResult, LsmrStopReason, MlsmrContinuation, MlsmrOptions, Progress, Staleness,
    StalenessError,
};
pub use schwarz::{ReductionStrategy, SchwarzPreconditioner};
