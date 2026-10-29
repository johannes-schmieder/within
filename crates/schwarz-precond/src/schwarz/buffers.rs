//! Free lists that let the additive Schwarz apply reuse its buffers instead of allocating.

use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::SolveError;

/// Free buffers of one length; keeps at most `max_free` for reuse.
pub(super) struct Pool {
    len: usize,
    max_free: usize,
    free: Mutex<Vec<Vec<f64>>>,
}

impl Pool {
    pub(super) fn new(len: usize, max_free: usize) -> Self {
        Self {
            len,
            max_free,
            free: Mutex::default(),
        }
    }

    /// Lends `f` a free buffer, or a new one, and keeps it for the next caller if there is room.
    pub(super) fn lend<R>(
        &self,
        f: impl FnOnce(&mut [f64]) -> Result<R, SolveError>,
    ) -> Result<R, SolveError> {
        // A separate statement releases the lock before a miss allocates.
        let free = self.free_list().pop();
        let mut buffer = match free {
            Some(buffer) => buffer,
            None => {
                let mut buffer = Vec::new();
                buffer
                    .try_reserve_exact(self.len)
                    .map_err(|_| SolveError::Allocation {
                        context: "SchwarzPreconditioner::scratch",
                    })?;
                buffer.resize(self.len, 0.0);
                buffer
            }
        };
        let result = f(&mut buffer);
        let mut free = self.free_list();
        // Caching is optional: a failed descriptor reservation must not turn a
        // completed solve into an error or abort after its numeric work.
        if free.len() < self.max_free && free.try_reserve(1).is_ok() {
            free.push(buffer);
        }
        result
    }

    // A panic elsewhere cannot leave a list of whole buffers inconsistent.
    fn free_list(&self) -> MutexGuard<'_, Vec<Vec<f64>>> {
        self.free.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_at_most_max_free_buffers() {
        let pool = Pool::new(1, 2);
        pool.lend(|_| pool.lend(|_| pool.lend(|_| Ok(())))).unwrap();
        assert_eq!(pool.free_list().len(), 2);
    }

    #[test]
    fn impossible_numeric_capacity_returns_allocation_without_running_work() {
        let pool = Pool::new(usize::MAX, 1);
        let result = pool.lend::<()>(|_| panic!("reservation must precede work"));
        assert!(matches!(result, Err(SolveError::Allocation { .. })));
        assert!(pool.free_list().is_empty());
    }

    #[test]
    fn nested_reservation_failure_returns_the_outer_buffer_for_reuse() {
        let pool = Pool::new(4, 1);
        let impossible = Pool::new(usize::MAX, 1);
        let error = pool.lend(|buffer| {
            buffer.fill(7.0);
            impossible.lend::<()>(|_| unreachable!())
        });
        assert!(matches!(error, Err(SolveError::Allocation { .. })));
        assert_eq!(pool.free_list().len(), 1);
        pool.lend(|buffer| {
            // The caller owns rewriting its whole scratch, including after an error.
            assert_eq!(buffer, &[7.0; 4]);
            buffer.fill(0.0);
            Ok(())
        })
        .unwrap();
        assert_eq!(pool.free_list().len(), 1);
    }
}
