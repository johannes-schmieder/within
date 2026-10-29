//! Caller-controlled cooperative cancellation.

use crate::SolveError;

/// An optional, borrowed cancellation query, shared by all workers in one call.
///
/// The query must stay `true` after cancellation and must be safe to call from
/// any worker. Host APIs restricted to the main thread must be polled there;
/// workers can query an atomic flag set by that thread instead:
///
/// ```
/// use std::sync::atomic::{AtomicBool, Ordering};
/// use schwarz_precond::Cancellation;
/// let cancelled = AtomicBool::new(false);
/// let query = || cancelled.load(Ordering::Relaxed);
/// let cancellation = Cancellation::new(&query);
/// assert!(!cancellation.is_cancelled());
/// ```
///
/// Cancellation is cooperative: checks occur between major stages and solver
/// iterations, not inside observation loops or a factorization. An in-flight
/// stage and its parallel work finish before the call returns. No wall-clock
/// bound on cancellation latency is promised. The default never cancels.
#[derive(Clone, Copy, Default)]
pub struct Cancellation<'a> {
    query: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
}

impl<'a> Cancellation<'a> {
    /// Borrow a thread-safe, sticky cancellation query for this call.
    pub fn new(query: &'a (dyn Fn() -> bool + Send + Sync)) -> Self {
        Self { query: Some(query) }
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(self) -> bool {
        self.query.is_some_and(|query| query())
    }

    pub(crate) fn checkpoint(self) -> Result<(), SolveError> {
        if self.is_cancelled() {
            Err(SolveError::Cancelled)
        } else {
            Ok(())
        }
    }
}
