//! The tokio runtime of an async loader: its own (`runtime_threads`, shut down in the
//! background) or the caller's (`runtime(handle)`).

use std::sync::{Arc, OnceLock};
use std::thread;

use tokio::runtime::{Builder, Handle, Runtime};

/// The most worker threads a loader's own runtime may have, see
/// [`runtime_threads`](crate::DataLoaderBuilder::runtime_threads).
pub const MAX_RUNTIME_THREADS: usize = 1024;

/// Worker threads of the runtime a loader builds for itself unless told otherwise: the CPUs
/// available to the process (affinity and cgroup quota included), 1 if unknown, at most
/// [`MAX_RUNTIME_THREADS`].
fn default_runtime_threads() -> usize {
    thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(MAX_RUNTIME_THREADS)
}

/// A runtime the loader built, shut down without waiting when the last user drops it: a
/// plain `Runtime` drop panics inside an async context.
struct OwnedRuntime(Option<Runtime>);

impl Drop for OwnedRuntime {
    fn drop(&mut self) {
        if let Some(runtime) = self.0.take() {
            runtime.shutdown_background();
        }
    }
}

/// The runtime an async loader runs on: its own, or the caller's.
#[derive(Clone)]
pub(crate) struct LoaderRuntime {
    pub(super) handle: Handle,
    /// Keeps the loader's own runtime alive; `None` for a caller's runtime.
    _owned: Option<Arc<OwnedRuntime>>,
}

impl LoaderRuntime {
    fn own(threads: usize) -> Self {
        let runtime = Builder::new_multi_thread()
            .worker_threads(threads)
            .thread_name("dataloader-io")
            .enable_all()
            .build()
            .expect("failed to build the loader's tokio runtime");
        Self {
            handle: runtime.handle().clone(),
            _owned: Some(Arc::new(OwnedRuntime(Some(runtime)))),
        }
    }
}

/// Which runtime an async loader fetches on.
enum RuntimeSetting {
    /// Its own, with this many worker threads (`None`: the default).
    Own(Option<usize>),
    /// The caller's.
    Caller(Handle),
}

/// The runtime of an async loader, as set by its builder, built on first use.
pub(crate) struct LazyRuntime {
    setting: RuntimeSetting,
    runtime: OnceLock<LoaderRuntime>,
}

impl LazyRuntime {
    /// The loader's own runtime, one worker thread per available CPU.
    pub(crate) fn new() -> Self {
        Self {
            setting: RuntimeSetting::Own(None),
            runtime: OnceLock::new(),
        }
    }

    /// Run on the caller's runtime instead of the loader's own.
    ///
    /// # Panics
    /// If the number of threads of the loader's own runtime is set too.
    pub(crate) fn set_handle(&mut self, handle: Handle) {
        assert!(
            !matches!(self.setting, RuntimeSetting::Own(Some(_))),
            "runtime and runtime_threads are mutually exclusive: \
             runtime_threads sizes the loader's own runtime"
        );
        self.setting = RuntimeSetting::Caller(handle);
    }

    /// Worker threads of the loader's own runtime.
    ///
    /// # Panics
    /// If `threads` is 0 or above [`MAX_RUNTIME_THREADS`], or if the caller's runtime is set.
    pub(crate) fn set_threads(&mut self, threads: usize) {
        assert!(
            (1..=MAX_RUNTIME_THREADS).contains(&threads),
            "runtime_threads must be between 1 and {MAX_RUNTIME_THREADS}, got {threads}"
        );
        assert!(
            matches!(self.setting, RuntimeSetting::Own(_)),
            "runtime and runtime_threads are mutually exclusive: \
             runtime_threads sizes the loader's own runtime"
        );
        self.setting = RuntimeSetting::Own(Some(threads));
    }

    /// The runtime to fetch on, built on the first call.
    pub(crate) fn get(&self) -> &LoaderRuntime {
        self.runtime.get_or_init(|| match &self.setting {
            RuntimeSetting::Own(threads) => {
                LoaderRuntime::own(threads.unwrap_or_else(default_runtime_threads))
            }
            RuntimeSetting::Caller(handle) => LoaderRuntime {
                handle: handle.clone(),
                _owned: None,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::{LazyRuntime, MAX_RUNTIME_THREADS};

    fn workers(runtime: &LazyRuntime) -> usize {
        runtime.get().handle.metrics().num_workers()
    }

    #[test]
    fn default_runtime_threads_are_the_available_cpus() {
        let expected = thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(MAX_RUNTIME_THREADS);
        assert_eq!(workers(&LazyRuntime::new()), expected);
    }

    #[test]
    fn runtime_threads_sizes_the_runtime() {
        let mut runtime = LazyRuntime::new();
        runtime.set_threads(3);
        assert_eq!(workers(&runtime), 3);
    }
}
