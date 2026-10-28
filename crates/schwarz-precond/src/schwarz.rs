//! Additive Schwarz preconditioner `M⁻¹ = Σ Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ` as an
//! [`Operator`](crate::Operator).

mod buffers;
mod executor;
mod planning;
mod preconditioner;

pub use planning::ReductionStrategy;
pub use preconditioner::SchwarzPreconditioner;
