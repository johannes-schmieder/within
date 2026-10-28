//! Free lists that let the additive Schwarz apply reuse its buffers instead of allocating.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Free buffers of one length; holds as many as were ever in use at once.
pub(super) struct Pool {
    len: usize,
    free: Mutex<Vec<Vec<f64>>>,
}

impl Pool {
    pub(super) fn new(len: usize) -> Self {
        Self {
            len,
            free: Mutex::default(),
        }
    }

    /// Lends `f` a free buffer, or a new one, and keeps it for the next caller.
    pub(super) fn lend<R>(&self, f: impl FnOnce(&mut [f64]) -> R) -> R {
        let mut buffer = self
            .free_list()
            .pop()
            .unwrap_or_else(|| vec![0.0; self.len]);
        let result = f(&mut buffer);
        self.free_list().push(buffer);
        result
    }

    // A panic elsewhere cannot leave a list of whole buffers inconsistent.
    fn free_list(&self) -> MutexGuard<'_, Vec<Vec<f64>>> {
        self.free.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
