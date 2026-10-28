//! Free lists that let the additive Schwarz apply reuse its buffers instead of allocating.

use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::local_solve::LocalSolveScratch;

/// A buffer whose only shape is its length.
pub(super) trait Buffer: Send {
    fn with_len(len: usize) -> Self;
}

impl Buffer for Vec<f64> {
    fn with_len(len: usize) -> Self {
        vec![0.0; len]
    }
}

impl Buffer for LocalSolveScratch {
    fn with_len(len: usize) -> Self {
        Self::new(len)
    }
}

/// Free buffers of one length; holds as many as were ever in use at once.
pub(super) struct Pool<T> {
    len: usize,
    free: Mutex<Vec<T>>,
}

impl<T: Buffer> Pool<T> {
    pub(super) fn new(len: usize) -> Self {
        Self {
            len,
            free: Mutex::default(),
        }
    }

    /// Lends `f` a free buffer, or a new one, and keeps it for the next caller.
    pub(super) fn lend<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let mut buffer = self
            .free_list()
            .pop()
            .unwrap_or_else(|| T::with_len(self.len));
        let result = f(&mut buffer);
        self.free_list().push(buffer);
        result
    }

    // A panic elsewhere cannot leave a list of whole buffers inconsistent.
    fn free_list(&self) -> MutexGuard<'_, Vec<T>> {
        self.free.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
