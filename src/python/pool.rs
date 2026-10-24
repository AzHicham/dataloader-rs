//! Pool threads attached to the interpreter for their whole life.
//!
//! On a thread Python does not know, each `Python::attach` creates a thread
//! state and the matching release destroys it: tens of microseconds on
//! free-threaded builds, paid by every batch and every item lane. The
//! loader's pool threads instead attach once when they start and run rayon's
//! loop detached, so attaching from a task only re-activates their thread
//! state.

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
