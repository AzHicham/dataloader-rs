//! [`LoaderPool`]: the loader's rayon pool.

use std::sync::Arc;
#[cfg(feature = "python")]
use std::sync::Weak;

/// Main function of a pool thread: runs `ThreadBuilder::run`, possibly
/// inside some per-thread setup.
pub(crate) type ThreadMain = fn(rayon::ThreadBuilder);

/// Stack size of a pool thread: a main thread's usual 8 MiB rather than the
/// 2 MiB of a spawned Rust thread, so a dataset recurses as deep as it would
/// on the caller's thread (CPython 3.14+ sizes its recursion limit from the
/// real stack). Only the pages a thread touches take memory.
const THREAD_STACK_SIZE: usize = 8 << 20;

/// The loader's rayon pool, rebuilt after a `fork()`.
///
/// A forked child inherits the pool but none of its threads, so batch tasks
/// spawned on it would never run and the epoch would hang (e.g. Python
/// `multiprocessing` with the `fork` start method).
pub(super) struct LoaderPool {
    pool: Arc<rayon::ThreadPool>,
    threads: usize,
    thread_main: Option<ThreadMain>,
    pid: u32,
}

impl LoaderPool {
    pub(super) fn new(threads: usize, thread_main: Option<ThreadMain>) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("dataloader-{i}"))
            .stack_size(THREAD_STACK_SIZE)
            .spawn_handler(move |thread| {
                let mut builder = std::thread::Builder::new();
                if let Some(name) = thread.name() {
                    builder = builder.name(name.to_owned());
                }
                if let Some(size) = thread.stack_size() {
                    builder = builder.stack_size(size);
                }
                builder.spawn(move || match thread_main {
                    Some(main) => main(thread),
                    None => thread.run(),
                })?;
                Ok(())
            })
            .build()
            .expect("failed to build rayon thread pool");
        Self {
            pool: Arc::new(pool),
            threads,
            thread_main,
            pid: std::process::id(),
        }
    }

    #[cfg(test)]
    pub(super) fn threads(&self) -> usize {
        self.threads
    }

    /// The pool, rebuilt first if this process is a fork of the one that
    /// built it.
    pub(super) fn get(&mut self) -> &Arc<rayon::ThreadPool> {
        if self.pid != std::process::id() {
            // Leak the inherited pool: dropping it would signal threads that
            // do not exist here, through locks a parent thread may have held
            // at fork time.
            let fresh = Self::new(self.threads, self.thread_main);
            std::mem::forget(std::mem::replace(self, fresh));
        }
        &self.pool
    }

    /// The pool, without keeping it alive.
    #[cfg(feature = "python")]
    pub(super) fn handle(&self) -> Weak<rayon::ThreadPool> {
        Arc::downgrade(&self.pool)
    }
}
