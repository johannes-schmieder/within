//! Scratch buffers for the additive Schwarz apply. [`WorkspacePool`] hands a [`Workspace`] to
//! each apply and takes it back afterwards, so the steady state allocates nothing.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use thread_local::ThreadLocal;

use crate::error::SolveError;
use crate::local_solve::LocalSolveScratch;

/// One apply's buffers: every subdomain's local output, plus per-worker local-solve scratch.
pub(super) struct Workspace {
    pub(super) local_outputs: Vec<f64>,
    scratch: Vec<LocalSolveScratch>,
}

impl Workspace {
    /// Runs `fill` on the local outputs with per-worker scratch, then reclaims the scratch.
    pub(super) fn fill_local_outputs(
        &mut self,
        max_scratch_size: usize,
        fill: impl FnOnce(&mut [f64], &WorkerBufferStack) -> Result<(), SolveError>,
    ) -> Result<(), SolveError> {
        let worker_scratch =
            WorkerBufferStack::new(std::mem::take(&mut self.scratch), max_scratch_size);
        let filled = fill(&mut self.local_outputs, &worker_scratch);
        // A pool-recovery failure must not mask a real `LocalSolveFailed`.
        match worker_scratch.into_pool() {
            Ok(recovered) => self.scratch = recovered,
            Err(into_err) => return filled.and(Err(into_err)),
        }
        filled
    }
}

/// Every slot of `local_outputs` is rewritten each apply, so a failed apply leaves nothing stale.
#[derive(Clone)]
pub(super) struct WorkspacePool {
    n_slots: usize,
    inner: Arc<Mutex<Vec<Workspace>>>,
}

impl WorkspacePool {
    const MAX_POOL_SIZE: usize = 4;

    pub(super) fn new(n_slots: usize) -> Self {
        Self {
            n_slots,
            inner: Arc::default(),
        }
    }

    pub(super) fn take(&self) -> Result<Workspace, SolveError> {
        let mut pool = self.inner.lock().map_err(|_| SolveError::Synchronization {
            context: "additive.workspace_pool.lock.pop",
        })?;
        Ok(pool.pop().unwrap_or_else(|| Workspace {
            local_outputs: vec![0.0; self.n_slots],
            scratch: Vec::new(),
        }))
    }

    /// Infallible: pool bookkeeping must never mask the caller's real result.
    pub(super) fn put(&self, workspace: Workspace) {
        // A poisoned lock means a worker panicked; drop it and let the next `take` re-allocate.
        if let Ok(mut pool) = self.inner.lock() {
            if pool.len() < Self::MAX_POOL_SIZE {
                pool.push(workspace);
            }
        }
    }
}

/// Thread-local scratch stack over a shared pool; retained buffers track re-entry depth.
pub(super) struct WorkerBufferStack {
    shared_pool: Mutex<Vec<LocalSolveScratch>>,
    worker_stacks: ThreadLocal<RefCell<Vec<LocalSolveScratch>>>,
    max_scratch_size: usize,
}

impl WorkerBufferStack {
    pub(super) fn new(pool: Vec<LocalSolveScratch>, max_scratch_size: usize) -> Self {
        Self {
            shared_pool: Mutex::new(pool),
            worker_stacks: ThreadLocal::with_capacity(rayon::current_num_threads().max(1)),
            max_scratch_size,
        }
    }

    pub(super) fn with_buffer<R>(&self, f: impl FnOnce(&mut LocalSolveScratch) -> R) -> R {
        let mut buffer = match self
            .worker_stacks
            .get_or(|| RefCell::new(Vec::new()))
            .borrow_mut()
            .pop()
        {
            Some(buffer) => buffer,
            None => self.take_or_alloc(),
        };
        let result = f(&mut buffer);
        self.worker_stacks
            .get_or(|| RefCell::new(Vec::new()))
            .borrow_mut()
            .push(buffer);
        result
    }

    fn take_or_alloc(&self) -> LocalSolveScratch {
        self.shared_pool
            .lock()
            .ok()
            .and_then(|mut pool| pool.pop())
            .unwrap_or_else(|| LocalSolveScratch::new(self.max_scratch_size))
    }

    /// Gather pool and worker stacks for the next apply.
    pub(super) fn into_pool(mut self) -> Result<Vec<LocalSolveScratch>, SolveError> {
        let mut pool = self
            .shared_pool
            .into_inner()
            .map_err(|_| SolveError::Synchronization {
                context: "additive.scratch.into_inner",
            })?;
        for worker_stack in self.worker_stacks.iter_mut() {
            pool.append(worker_stack.get_mut());
        }
        Ok(pool)
    }
}
