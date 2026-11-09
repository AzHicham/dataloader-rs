//! Pool threads attached to the interpreter for their whole life.
//!
//! On a thread Python does not know, each `Python::attach` creates a thread
//! state and the matching release destroys it: tens of microseconds on
//! free-threaded builds, paid by every batch and every item lane. The
//! loader's pool threads instead attach once when they start and run rayon's
//! loop detached, so attaching from a task only re-activates their thread
//! state.

use std::sync::{Mutex, Once, PoisonError, Weak};
use std::thread;

use pyo3::prelude::*;

/// Main function of the loader's pool threads.
///
/// A thread that starts while the interpreter is shutting down runs without a
/// thread state, as before. A pool dropped during shutdown makes its threads
/// re-attach on exit; CPython ends (≤ 3.13) or parks (3.14+) such a thread,
/// which is harmless since the process is exiting.
pub(crate) fn run_attached(thread: rayon::ThreadBuilder) {
    let mut thread = Some(thread);
    Python::try_attach(|py| py.detach(|| thread.take().map(rayon::ThreadBuilder::run)));
    if let Some(thread) = thread {
        thread.run();
    }
}

/// The pools of the loaders this process created, for [`keep_pools_alive`].
static POOLS: Mutex<Vec<Weak<rayon::ThreadPool>>> = Mutex::new(Vec::new());

/// Remember a loader's pool for [`keep_pools_alive`].
pub(crate) fn track(pool: Weak<rayon::ThreadPool>) {
    let mut pools = POOLS.lock().unwrap_or_else(PoisonError::into_inner);
    pools.retain(|p| p.strong_count() > 0);
    if !pools.iter().any(|p| p.ptr_eq(&pool)) {
        pools.push(pool);
    }
}

/// Run by `atexit`, before the interpreter starts shutting down: leak a
/// reference to every live pool, so that a loader freed during shutdown
/// (module teardown, a reference cycle collected last) does not wake its
/// threads. A woken thread re-attaches to the finalizing interpreter, which
/// CPython ≤ 3.13 answers with `pthread_exit`; that unwinds through Rust
/// frames, and the process aborts. The process is exiting: nothing is lost.
#[pyfunction]
#[pyo3(name = "_keep_pools_alive")]
pub(crate) fn keep_pools_alive() {
    let pools = std::mem::take(&mut *POOLS.lock().unwrap_or_else(PoisonError::into_inner));
    for pool in pools.iter().filter_map(Weak::upgrade) {
        std::mem::forget(pool);
    }
}

/// Attach the threads of rayon's global pool to the interpreter for their whole
/// life too: `PyAsyncDataset` calls `__getitem__` and submits its coroutines
/// from there. Called when the first such loader is built. Rayon waits for the global pool's threads to start, and a
/// starting thread may wait for every attached thread to pause (free-threaded
/// builds), so the caller waits with the interpreter released. A no-op if the
/// global pool already exists.
pub(crate) fn attach_global_pool(py: Python<'_>) {
    static ONCE: Once = Once::new();
    py.detach(|| {
        ONCE.call_once(|| {
            let _ = rayon::ThreadPoolBuilder::new()
                .thread_name(|i| format!("dataloader-map-{i}"))
                .spawn_handler(|thread| {
                    let mut builder = thread::Builder::new();
                    if let Some(name) = thread.name() {
                        builder = builder.name(name.to_owned());
                    }
                    builder.spawn(move || run_attached(thread))?;
                    Ok(())
                })
                .build_global();
        })
    });
}
