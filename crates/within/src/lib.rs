#![deny(missing_docs)]
//! Fixed-effects normal-equation solver. Solves `G x = D^T W y` (with
//! `G = D^T W D`) for a sparse categorical design `D` via modified LSMR with a
//! Schwarz preconditioner over factor-pair subdomains.
//!
//! ```
//! use ndarray::Array2;
//! use within::{solve, LsmrOptions};
//!
//! let categories = Array2::<u32>::zeros((10_000, 2));
//! let y = vec![0.0; 10_000];
//! let r = solve(categories.view(), &y, None, &LsmrOptions::default(), None).unwrap();
//! assert!(r.converged);
//! ```
//!
//! # Reproducibility
//!
//! On the same machine/build, coefficients, residuals and iteration counts are
//! bitwise-reproducible across Rayon worker counts with [`PreconditionerConfig::Off`],
//! [`PreconditionerConfig::Diagonal`], or additive Schwarz configured with
//! [`ReductionStrategy::ParallelReduction`]. The design operator and LSMR use
//! fixed arithmetic trees; Schwarz combines local outputs in subdomain order.
//! Batch RHS scheduling and concurrent preconditioner reuse do not change that order.
//!
//! [`ReductionStrategy::AtomicScatter`] retains scheduling-dependent floating-point
//! additions. [`ReductionStrategy::Auto`] may select that backend and therefore
//! does not guarantee bitwise reproducibility. Request `ParallelReduction`
//! explicitly for reproducible additive solves. Cross-machine/build bit identity
//! and wall-clock diagnostic identity are not guaranteed.

pub mod config;
pub mod error;
pub mod observation;

pub(crate) mod block_elim;
pub(crate) mod channel;
pub(crate) mod csr_block;
pub(crate) mod domain;
pub(crate) mod linalg;
pub(crate) mod operator;
pub(crate) mod solver;

pub use channel::{Channel, ChannelPair};
pub use config::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, LsmrOptions, PreconditionerConfig,
    ReductionStrategy, ScalingConfig, ScalingFailure, SchurMode,
};
pub use domain::{Design, Effect};
pub use error::{BuildError, BuildWarning, SolveError, WithinError};
pub use operator::schwarz::Preconditioner;
pub use solver::{
    solve, solve_batch, BatchSolveResult, CoefficientAddress, CoefficientLayout, IntoDesign,
    PreconditionerInput, SolveResult, Solver,
};
