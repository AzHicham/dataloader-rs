//! The async path: an [`AsyncDataset`] fetched on a tokio runtime, transformed and collated
//! on the loader's rayon pool, delivered to the consumer through a bounded channel.

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};
use std::thread;
#[cfg(feature = "python")]
use std::time::Instant;

#[cfg(feature = "python")]
use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::{Receiver, unbounded};
use futures_util::stream::{self, StreamExt, TryStreamExt};
use tokio::runtime::{Builder, Handle, Runtime};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};
use tokio::task::{JoinError, JoinHandle};

use crate::{
    collator::Collator,
    dataset::AsyncDataset,
    error::Result,
    loader::{
        iter::{BatchStream, forked},
        source::{AsyncSource, EpochSettings},
    },
    transform::{Transform, apply_all},
};

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
pub(super) struct LoaderRuntime {
    handle: Handle,
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
pub(super) struct LazyRuntime {
    setting: RuntimeSetting,
    runtime: OnceLock<LoaderRuntime>,
}

impl LazyRuntime {
    /// The loader's own runtime, one worker thread per available CPU.
    pub(super) fn new() -> Self {
        Self {
            setting: RuntimeSetting::Own(None),
            runtime: OnceLock::new(),
        }
    }

    /// Run on the caller's runtime instead of the loader's own.
    ///
    /// # Panics
    /// If the number of threads of the loader's own runtime is set too.
    pub(super) fn set_handle(&mut self, handle: Handle) {
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
    pub(super) fn set_threads(&mut self, threads: usize) {
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
    pub(super) fn get(&self) -> &LoaderRuntime {
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

/// A task handle that aborts the task when dropped, so that dropping the epoch drops every
/// future in flight.
struct AbortOnDrop<T>(JoinHandle<T>);

impl<T> Future for AbortOnDrop<T> {
    type Output = std::result::Result<T, JoinError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.0).poll(cx)
    }
}

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// A batch, or the panic that ended its task, with the permit of its place in the channel.
type Delivery<B> = (thread::Result<Result<B>>, OwnedSemaphorePermit);

/// One epoch of an async loader.
///
/// A driver task on the runtime walks the epoch's batches with
/// `stream::iter(batches).map(spawn batch task).buffered(num_workers)`: at most `num_workers`
/// batch tasks exist at once, each fetching at most `item_concurrency` items, transforming
/// them and collating on the rayon pool. Batches leave in epoch order through a channel holding at most
/// `prefetch_depth` of them (a semaphore permit travels with each). Each batch runs as its own
/// task, so its fetches progress while the driver waits for room in the channel.
///
/// Dropping the epoch aborts the driver, which drops (and so aborts) every batch task and
/// their fetches; it never waits for them.
pub struct AsyncEpoch<B> {
    rx: Receiver<Delivery<B>>,
    remaining: usize,
    _driver: AbortOnDrop<()>,
    /// Keeps the loader's runtime alive while the epoch runs.
    _runtime: LoaderRuntime,
    /// The process that started the epoch: a `fork()`ed child has no runtime threads.
    pid: u32,
}

impl<B: Send + 'static> AsyncEpoch<B> {
    pub(super) fn start<A, T, C>(
        source: &AsyncSource<A>,
        transform: &Arc<T>,
        collator: &Arc<C>,
        pool: &Arc<rayon::ThreadPool>,
        settings: EpochSettings,
        chunks: Vec<Vec<usize>>,
    ) -> Self
    where
        A: AsyncDataset,
        T: Transform<A::Item>,
        C: Collator<T::Output, Batch = B>,
    {
        let EpochSettings {
            num_workers,
            item_concurrency,
            prefetch_depth,
        } = settings;
        let remaining = chunks.len();
        let (tx, rx) = unbounded();
        let room = Arc::new(Semaphore::new(prefetch_depth));
        let dataset = Arc::clone(&source.dataset);
        let transform = Arc::clone(transform);
        let collator = Arc::clone(collator);
        let pool = Arc::clone(pool);
        let runtime = source.runtime.get();
        let handle = runtime.handle.clone();
        let driver = runtime.handle.spawn(async move {
            let mut batches = stream::iter(chunks)
                .map(|indices| {
                    let task = load_batch(
                        Arc::clone(&dataset),
                        Arc::clone(&transform),
                        Arc::clone(&collator),
                        Arc::clone(&pool),
                        indices,
                        item_concurrency,
                    );
                    AbortOnDrop(handle.spawn(task))
                })
                .buffered(num_workers.max(1));
            while let Some(joined) = batches.next().await {
                let result = match joined {
                    Ok(batch) => Ok(batch),
                    Err(error) => match error.try_into_panic() {
                        Ok(panic) => Err(panic),
                        // Cancelled: the runtime is shutting down.
                        Err(_) => return,
                    },
                };
                let Ok(permit) = Arc::clone(&room).acquire_owned().await else {
                    return;
                };
                // The consumer is gone: stop.
                if tx.send((result, permit)).is_err() {
                    return;
                }
            }
        });
        Self {
            rx,
            remaining,
            _driver: AbortOnDrop(driver),
            _runtime: runtime.clone(),
            pid: std::process::id(),
        }
    }

    /// Return a delivered batch, or re-raise the panic of its task, which ends the epoch.
    fn take(&mut self, (result, permit): Delivery<B>) -> Result<B> {
        drop(permit);
        self.remaining -= 1;
        result.unwrap_or_else(|panic| {
            self.remaining = 0;
            resume_unwind(panic)
        })
    }

    /// The driver ended before the epoch did: its runtime was shut down.
    fn stopped(&mut self) -> Result<B> {
        self.remaining = 0;
        Err("the async loader's runtime stopped before the end of the epoch".into())
    }
}

impl<B: Send + 'static> BatchStream<B> for AsyncEpoch<B> {
    fn len(&self) -> usize {
        self.remaining
    }

    fn next(&mut self) -> Option<Result<B>> {
        if self.remaining == 0 {
            return None;
        }
        if self.pid != std::process::id() {
            self.remaining = 0;
            return Some(Err(forked()));
        }
        match self.rx.recv() {
            Ok(delivery) => Some(self.take(delivery)),
            Err(_) => Some(self.stopped()),
        }
    }

    #[cfg(feature = "python")]
    fn next_until(&mut self, deadline: Option<Instant>) -> Option<Option<Result<B>>> {
        if self.remaining == 0 {
            return Some(None);
        }
        let delivery = match deadline {
            None => self.rx.try_recv().ok()?,
            Some(_) if self.pid != std::process::id() => {
                self.remaining = 0;
                return Some(Some(Err(forked())));
            }
            Some(deadline) => match self.rx.recv_deadline(deadline) {
                Ok(delivery) => delivery,
                Err(RecvTimeoutError::Timeout) => return None,
                Err(RecvTimeoutError::Disconnected) => return Some(Some(self.stopped())),
            },
        };
        Some(Some(self.take(delivery)))
    }
}

/// One batch: fetch its items (`get_batch`, or `get` with `item_concurrency` in flight),
/// transform and collate them on the rayon pool. With `get`, each item is transformed as soon as
/// it arrives, while the others are still in flight. The first failed item fails the batch and
/// drops the other fetches in flight.
async fn load_batch<A, T, C>(
    dataset: Arc<A>,
    transform: Arc<T>,
    collator: Arc<C>,
    pool: Arc<rayon::ThreadPool>,
    indices: Vec<usize>,
    item_concurrency: usize,
) -> Result<C::Batch>
where
    A: AsyncDataset,
    T: Transform<A::Item>,
    C: Collator<T::Output>,
{
    if item_concurrency == 1 {
        let items = dataset.get_batch(&indices).await?;
        return on_pool(&pool, move || {
            collator.collate(apply_all(&*transform, &indices, items)?)
        })
        .await;
    }
    let items: Vec<T::Output> = stream::iter(indices.iter().copied())
        .map(|index| {
            let (dataset, transform, pool) = (&dataset, &transform, &pool);
            async move {
                let item = dataset.get(index).await?;
                if transform.is_identity() {
                    return transform.apply(index, item);
                }
                let transform = Arc::clone(transform);
                on_pool(pool, move || transform.apply(index, item)).await
            }
        })
        .buffered(item_concurrency)
        .try_collect()
        .await?;
    on_pool(&pool, move || collator.collate(items)).await
}

/// Run `job` on the rayon pool and await its result. A panic in `job` is resumed here.
async fn on_pool<R, F>(pool: &rayon::ThreadPool, job: F) -> Result<R>
where
    R: Send + 'static,
    F: FnOnce() -> Result<R> + Send + 'static,
{
    let (tx, rx) = oneshot::channel();
    pool.spawn(move || {
        // A panic must not escape: rayon aborts the process on a panicking `spawn`.
        let _ = tx.send(catch_unwind(AssertUnwindSafe(job)));
    });
    match rx.await {
        Ok(Ok(result)) => result,
        Ok(Err(panic)) => resume_unwind(panic),
        Err(_) => Err("the loader's pool dropped a job before it ran".into()),
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::{LazyRuntime, MAX_RUNTIME_THREADS};
    use crate::{
        dataset::AsyncDataset,
        error::Result,
        loader::{AsyncSource, DataLoader},
        sampler::Sampler,
    };

    struct Echo;

    impl AsyncDataset for Echo {
        type Item = usize;

        async fn get(&self, index: usize) -> Result<usize> {
            Ok(index)
        }

        fn len(&self) -> usize {
            1
        }
    }

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

    fn pool_threads<S: Sampler, C, T>(loader: &DataLoader<AsyncSource<Echo>, S, C, T>) -> usize {
        loader
            .pool
            .as_ref()
            .expect("an async loader has a pool")
            .threads()
    }

    #[test]
    fn the_pool_has_a_thread_per_item_in_flight_up_to_the_cpus() {
        let cpus = thread::available_parallelism().map_or(1, |n| n.get());
        let plain = DataLoader::builder_async(Echo)
            .num_workers(2)
            .item_concurrency(32)
            .build();
        assert_eq!(
            pool_threads(&plain),
            2,
            "no transform: one thread per batch"
        );
        let transformed = DataLoader::builder_async(Echo)
            .num_workers(2)
            .item_concurrency(32)
            .transform(|_: usize, x: usize| -> Result<usize> { Ok(x) })
            .build();
        assert_eq!(pool_threads(&transformed), (2 * 32).min(cpus).max(2));
    }
}
