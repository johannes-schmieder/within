//! Whether local solves may use nested Rayon parallelism, from build-time work estimates and the
//! current Rayon width.

/// Ignored: every apply sums each DOF's subdomain outputs in subdomain order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ReductionStrategy {
    /// Ignored.
    #[default]
    Auto,
    /// Ignored.
    AtomicScatter,
    /// Ignored.
    ParallelReduction,
}

/// Build-time inner-parallelism work estimates.
#[derive(Debug, Clone, Copy)]
pub(super) struct AdditiveScheduler {
    pub(super) total_inner_parallel_work: usize,
    pub(super) max_inner_parallel_work: usize,
}

impl AdditiveScheduler {
    const MIN_INNER_PARALLEL_WORK: usize = 200_000;
    const OUTER_CAPACITY_TARGET: f64 = 0.75;

    fn outer_parallel_capacity(self) -> f64 {
        if self.max_inner_parallel_work == 0 {
            return 0.0;
        }
        self.total_inner_parallel_work as f64 / self.max_inner_parallel_work as f64
    }

    pub(super) fn allow_inner_parallelism(self, threads: usize) -> bool {
        if self.max_inner_parallel_work < Self::MIN_INNER_PARALLEL_WORK {
            return false;
        }

        self.outer_parallel_capacity() < (threads as f64 * Self::OUTER_CAPACITY_TARGET)
    }
}
