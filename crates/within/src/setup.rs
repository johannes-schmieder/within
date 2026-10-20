//! Checked live-payload accounting for the serial preconditioner build.
//!
//! [`Preconditioner::build_checked`](crate::Preconditioner::build_checked) reserves
//! every heap array through one [`Ledger`]. The ledger charges the requested
//! capacity before allocating and keeps the old capacity charged while a growing
//! array is reallocated. Exceeding the declared limit fails before the allocation
//! it would have needed. Local factors are built by `approx-chol`'s own checked
//! builder against the remaining budget, and their peak is folded in. Numerics
//! never depend on the limit: a build either returns what the unlimited build
//! returns or an error.

use crate::BuildError;

/// Declared ceiling on live payload during a checked preconditioner build.
///
/// Counts the requested heap capacity of every array the build allocates,
/// including the old and new buffers of a reallocation, `approx-chol`'s
/// construction payload, sampler scratch and the fixed-size reference-counted
/// blocks of the returned preconditioner. Excludes the caller's design and
/// weights, inline objects, stack, allocator headers and capacity the allocator
/// hands out beyond the request. This is not a process-memory quota.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupLimit {
    /// Inclusive ceiling in bytes.
    pub maximum_payload_bytes: usize,
}

impl SetupLimit {
    /// No ceiling; the accounting still runs and is reported.
    pub const UNLIMITED: Self = Self {
        maximum_payload_bytes: usize::MAX,
    };
}

/// Payload accounting of one successful checked build.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SetupReport {
    /// Peak live requested payload in bytes, including reallocation overlap.
    pub peak_payload_bytes: usize,
    /// Payload still charged when the preconditioner is returned: every retained
    /// array and reference-counted block it owns.
    pub retained_payload_bytes: usize,
}

/// Live requested payload against one declared limit.
pub(crate) struct Ledger {
    live: usize,
    peak: usize,
    limit: usize,
    /// Test-only fault injection: the reservation with this ordinal fails.
    #[cfg(test)]
    fail_at: Option<usize>,
    #[cfg(test)]
    reservations: usize,
}

impl Ledger {
    pub(crate) fn new(limit: SetupLimit) -> Self {
        Self {
            live: 0,
            peak: 0,
            limit: limit.maximum_payload_bytes,
            #[cfg(test)]
            fail_at: None,
            #[cfg(test)]
            reservations: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn failing_at(limit: SetupLimit, reservation: usize) -> Self {
        Self {
            fail_at: Some(reservation),
            ..Self::new(limit)
        }
    }

    #[cfg(test)]
    pub(crate) fn reservations(&self) -> usize {
        self.reservations
    }

    pub(crate) fn report(&self) -> SetupReport {
        SetupReport {
            peak_payload_bytes: self.peak,
            retained_payload_bytes: self.live,
        }
    }

    /// Bytes still admissible under the limit.
    pub(crate) fn remaining(&self) -> usize {
        self.limit - self.live
    }

    /// A size no allocation can satisfy is an allocation failure, as the standard
    /// library's capacity overflow is; only a representable excess is a budget refusal.
    pub(crate) fn charge(&mut self, context: &'static str, bytes: usize) -> Result<(), BuildError> {
        let required = self
            .live
            .checked_add(bytes)
            .filter(|&required| required <= isize::MAX as usize)
            .ok_or(BuildError::SetupAllocationFailed { context, bytes })?;
        if required > self.limit {
            return Err(BuildError::SetupBudgetExceeded {
                context,
                required,
                limit: self.limit,
            });
        }
        self.live = required;
        self.peak = self.peak.max(required);
        Ok(())
    }

    pub(crate) fn release(&mut self, bytes: usize) {
        debug_assert!(bytes <= self.live, "released more payload than charged");
        self.live -= bytes;
    }

    /// The refusal of a nested build that needed `required` bytes beyond the
    /// current live payload.
    pub(crate) fn refusal(&self, context: &'static str, required: usize) -> BuildError {
        BuildError::SetupBudgetExceeded {
            context,
            required: self.live.saturating_add(required),
            limit: self.limit,
        }
    }

    /// Fold in a nested checked build that started from the current live payload.
    pub(crate) fn absorb(&mut self, peak: usize, retained: usize) {
        debug_assert!(retained <= peak && self.live + peak <= self.limit);
        self.peak = self.peak.max(self.live + peak);
        self.live += retained;
    }

    pub(crate) fn bytes<T>(context: &'static str, capacity: usize) -> Result<usize, BuildError> {
        capacity
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(BuildError::SetupAllocationFailed {
                context,
                bytes: usize::MAX,
            })
    }

    #[cfg(test)]
    fn injected_failure(&mut self) -> bool {
        self.reservations += 1;
        self.fail_at == Some(self.reservations)
    }

    #[cfg(not(test))]
    #[inline(always)]
    fn injected_failure(&mut self) -> bool {
        false
    }

    /// Charge `bytes` for a reservation another crate performs, releasing the
    /// charge again if `reserve` fails.
    pub(crate) fn reserve_external<E>(
        &mut self,
        context: &'static str,
        bytes: usize,
        reserve: impl FnOnce() -> Result<(), E>,
    ) -> Result<(), BuildError> {
        self.charge(context, bytes)?;
        if self.injected_failure() || reserve().is_err() {
            self.release(bytes);
            return Err(BuildError::SetupAllocationFailed { context, bytes });
        }
        Ok(())
    }

    /// Grow `values` to hold at least `total` elements without another reallocation.
    ///
    /// Grows to `max(total, 2 * capacity)`, charging the new buffer while the old
    /// one is still live. A no-op when the capacity already suffices.
    pub(crate) fn ensure<T>(
        &mut self,
        context: &'static str,
        values: &mut Vec<T>,
        total: usize,
    ) -> Result<(), BuildError> {
        if total <= values.capacity() || core::mem::size_of::<T>() == 0 {
            return Ok(());
        }
        let old = values.capacity();
        let target = total.max(old.saturating_mul(2));
        let requested = Self::bytes::<T>(context, target)?;
        self.charge(context, requested)?;
        if self.injected_failure() || values.try_reserve_exact(target - values.len()).is_err() {
            self.release(requested);
            return Err(BuildError::SetupAllocationFailed {
                context,
                bytes: requested,
            });
        }
        // Charge any capacity the allocator handed out beyond the request.
        self.charge(
            context,
            Self::bytes::<T>(context, values.capacity() - target)?,
        )?;
        self.release(Self::bytes::<T>(context, old)?);
        Ok(())
    }

    /// An empty array with room for exactly `capacity` elements.
    pub(crate) fn with_capacity<T>(
        &mut self,
        context: &'static str,
        capacity: usize,
    ) -> Result<Vec<T>, BuildError> {
        let mut values = Vec::new();
        self.ensure(context, &mut values, capacity)?;
        Ok(values)
    }

    /// `len` copies of `value`.
    pub(crate) fn filled<T: Clone>(
        &mut self,
        context: &'static str,
        len: usize,
        value: T,
    ) -> Result<Vec<T>, BuildError> {
        let mut values = self.with_capacity(context, len)?;
        values.resize(len, value);
        Ok(values)
    }

    /// A copy of `source` with exactly its length as capacity.
    pub(crate) fn copied<T: Copy>(
        &mut self,
        context: &'static str,
        source: &[T],
    ) -> Result<Vec<T>, BuildError> {
        let mut values = self.with_capacity(context, source.len())?;
        values.extend_from_slice(source);
        Ok(values)
    }

    /// Push one element, growing through [`Self::ensure`] when full.
    #[inline]
    pub(crate) fn push<T>(
        &mut self,
        context: &'static str,
        values: &mut Vec<T>,
        value: T,
    ) -> Result<(), BuildError> {
        if values.len() == values.capacity() {
            let total = values
                .len()
                .checked_add(1)
                .ok_or(BuildError::SetupAllocationFailed {
                    context,
                    bytes: usize::MAX,
                })?;
            self.ensure(context, values, total)?;
        }
        values.push(value);
        Ok(())
    }

    /// Drop an array, releasing its whole capacity.
    pub(crate) fn free<T>(&mut self, values: Vec<T>) {
        let capacity = values.capacity();
        drop(values);
        self.forget::<T>(capacity);
    }

    /// Release the capacity of an array the caller has already dropped.
    pub(crate) fn forget<T>(&mut self, capacity: usize) {
        self.release(capacity * core::mem::size_of::<T>());
    }

    /// Charge one reference-counted block holding a `T`: two counters, then the value.
    pub(crate) fn charge_rc<T>(&mut self, context: &'static str) -> Result<(), BuildError> {
        self.charge(context, rc_block_bytes::<T>())
    }
}

/// Heap bytes of an `Arc<T>` allocation: strong and weak counters, then the value,
/// padded to the block's alignment.
pub(crate) const fn rc_block_bytes<T>() -> usize {
    let align = if core::mem::align_of::<T>() > core::mem::align_of::<usize>() {
        core::mem::align_of::<T>()
    } else {
        core::mem::align_of::<usize>()
    };
    let header = 2 * core::mem::size_of::<usize>();
    let start = header.div_ceil(align) * align;
    (start + core::mem::size_of::<T>()).div_ceil(align) * align
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_charges_overlap_and_limits_are_inclusive() {
        let mut ledger = Ledger::new(SetupLimit {
            maximum_payload_bytes: 8 * 12,
        });
        let mut values: Vec<u64> = ledger.with_capacity("test", 4).unwrap();
        assert_eq!(ledger.report().peak_payload_bytes, 32);
        values.extend([1, 2, 3, 4]);
        // Doubling to 8 holds 4 old plus 8 new elements: exactly the limit.
        ledger.push("test", &mut values, 5).unwrap();
        assert_eq!(ledger.report().peak_payload_bytes, 96);
        assert_eq!(ledger.report().retained_payload_bytes, 64);
        values.extend([6, 7, 8]);
        // Doubling to 16 would hold 8 + 16 elements.
        assert!(matches!(
            ledger.push("test", &mut values, 9),
            Err(BuildError::SetupBudgetExceeded {
                context: "test",
                required: 192,
                limit: 96,
            })
        ));
        ledger.free(values);
        assert_eq!(ledger.report().retained_payload_bytes, 0);
    }

    #[test]
    fn nested_builds_fold_in_from_the_current_payload() {
        let mut ledger = Ledger::new(SetupLimit {
            maximum_payload_bytes: 100,
        });
        ledger.charge("test", 40).unwrap();
        assert_eq!(ledger.remaining(), 60);
        ledger.absorb(60, 10);
        assert_eq!(
            ledger.report(),
            SetupReport {
                peak_payload_bytes: 100,
                retained_payload_bytes: 50,
            }
        );
    }

    #[test]
    fn unrepresentable_sizes_are_allocation_failures() {
        let mut ledger = Ledger::new(SetupLimit::UNLIMITED);
        assert!(matches!(
            ledger.with_capacity::<u64>("test", usize::MAX / 4),
            Err(BuildError::SetupAllocationFailed { .. })
        ));
        assert_eq!(ledger.report().retained_payload_bytes, 0);
    }

    #[test]
    fn injected_failures_release_their_charge() {
        let mut ledger = Ledger::failing_at(SetupLimit::UNLIMITED, 2);
        let first: Vec<u8> = ledger.with_capacity("test", 3).unwrap();
        assert!(matches!(
            ledger.with_capacity::<u8>("test", 5),
            Err(BuildError::SetupAllocationFailed {
                context: "test",
                bytes: 5,
            })
        ));
        assert_eq!(ledger.report().retained_payload_bytes, 3);
        ledger.free(first);
    }

    #[test]
    fn rc_blocks_match_the_standard_layout() {
        use std::sync::Arc;
        // A value at least as aligned as the counters starts right after them.
        assert_eq!(rc_block_bytes::<u64>(), 24);
        assert_eq!(rc_block_bytes::<[u8; 3]>(), 24);
        assert_eq!(
            rc_block_bytes::<Vec<u8>>(),
            16 + std::mem::size_of::<Vec<u8>>()
        );
        let _ = Arc::new(0u8);
    }
}
