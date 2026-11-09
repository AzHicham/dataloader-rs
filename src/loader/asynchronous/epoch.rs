//! [`AsyncEpoch`]: an [`AsyncDataset`] fetched on a tokio runtime, transformed and collated
//! on the loader's rayon pool, delivered to the consumer through a bounded channel.

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread;
#[cfg(feature = "python")]
use std::time::Instant;

#[cfg(feature = "python")]
use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::{Receiver, unbounded};
use futures_util::stream::{self, StreamExt, TryStreamExt};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};
use tokio::task::{JoinError, JoinHandle};

use super::runtime::LoaderRuntime;
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
    pub(crate) fn start<A, T, C>(
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
