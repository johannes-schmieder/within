//! Construction checkpoints use the same caller query as the iterative solver.

use crate::{BuildError, Cancellation};

pub(crate) fn checkpoint(cancellation: Cancellation<'_>) -> Result<(), BuildError> {
    if cancellation.is_cancelled() {
        Err(BuildError::Cancelled)
    } else {
        Ok(())
    }
}
