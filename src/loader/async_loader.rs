//! Loader for [`AsyncDataset`]s: many samples in flight on few threads.

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::pin::pin;
use std::sync::Arc;

use crossbeam_channel::{Receiver, bounded};
use futures::{Stream, StreamExt, channel::oneshot, stream};

use crate::{
    collator::{Collator, VecCollator},
    dataset::AsyncDataset,
    error::Result,
    sampler::{BatchSampler, Sampler, SequentialSampler},
};

/// DataLoader for an [`AsyncDataset`].
///
/// Up to `concurrency` batches are fetched at once, each with all its
/// samples in flight concurrently (see [`AsyncDataset::get_batch`]), and
/// batches are yielded in sampler order. With
/// [`max_concurrency`](AsyncDataLoaderBuilder::max_concurrency), samples are
/// instead fetched one `get` each, at most `n` at a time across batches. Collation runs inline, or on a
/// dedicated rayon pool with [`collate_threads`](AsyncDataLoaderBuilder::collate_threads)
/// so CPU work never blocks the executor.
///
/// Consume it either as a [`Stream`] from your own runtime
/// ([`stream`](Self::stream)) or as a blocking iterator
/// ([`iter`](Self::iter)), which drives the stream on a background thread:
/// with `futures::executor` by default, or inside a tokio runtime with the
/// `tokio` feature (for datasets built on reqwest, the AWS SDK, …).
pub struct AsyncDataLoader<D, S: Sampler, C> {
    dataset: Arc<D>,
    batch_sampler: BatchSampler<S>,
    collator: Arc<C>,
    concurrency: usize,
    max_concurrency: Option<usize>,
    collate_pool: Option<Arc<rayon::ThreadPool>>,
    #[cfg(feature = "tokio")]
    tokio: TokioDriver,
}

impl<D: AsyncDataset> AsyncDataLoader<D, SequentialSampler, VecCollator> {
    /// Create an [`AsyncDataLoaderBuilder`] with [`SequentialSampler`] and
    /// [`VecCollator`] as defaults.
    pub fn builder(dataset: D) -> AsyncDataLoaderBuilder<D, SequentialSampler, VecCollator> {
        AsyncDataLoaderBuilder {
            dataset,
            sampler: SequentialSampler,
            collator: VecCollator,
            batch_size: 1,
            drop_last: false,
            concurrency: 4,
            max_concurrency: None,
            collate_threads: 0,
            #[cfg(feature = "tokio")]
            tokio_handle: None,
        }
    }
}

impl<D, S, C> AsyncDataLoader<D, S, C>
where
    D: AsyncDataset,
    S: Sampler,
    C: Collator<D::Item>,
{
    /// One epoch as a [`Stream`] of batches, in sampler order.
    ///
    /// Runtime-agnostic: poll it from any executor. Advances the sampler.
    pub fn stream(&mut self) -> impl Stream<Item = Result<C::Batch>> + Send + 'static {
        let chunks = self.batch_sampler.batch_indices(self.dataset.len());
        self.stream_of(chunks)
    }

    fn stream_of(
        &self,
        chunks: Vec<Vec<usize>>,
    ) -> impl Stream<Item = Result<C::Batch>> + Send + 'static {
        let dataset = Arc::clone(&self.dataset);
        let collator = Arc::clone(&self.collator);
        let pool = self.collate_pool.clone();

        let Some(max_concurrency) = self.max_concurrency else {
            return stream::iter(chunks)
                .map(move |indices| {
                    let dataset = Arc::clone(&dataset);
                    let collator = Arc::clone(&collator);
                    let pool = pool.clone();
                    async move {
                        let items = dataset.get_batch(&indices).await?;
                        collate(collator, pool, items).await
                    }
                })
                // `buffered` keeps sampler order and caps batches in flight.
                .buffered(self.concurrency)
                .left_stream();
        };

        // Sample-level limit: one `get` per sample, at most `max_concurrency`
        // in flight across batch boundaries, regrouped in order. The batch
        // sampler only yields full batches plus a possibly shorter last one,
        // so `chunks(batch_size)` restores the original batches.
        let batch_size = self.batch_sampler.batch_size();
        stream::iter(chunks.into_iter().flatten())
            .map(move |index| {
                let dataset = Arc::clone(&dataset);
                async move { dataset.get(index).await }
            })
            .buffered(max_concurrency)
            .chunks(batch_size)
            .map(move |samples| {
                let collator = Arc::clone(&collator);
                let pool = pool.clone();
                async move {
                    let items = samples.into_iter().collect::<Result<Vec<_>>>()?;
                    collate(collator, pool, items).await
                }
            })
            .buffered(self.concurrency)
            .right_stream()
    }

    /// One epoch as a blocking iterator, for synchronous training loops.
    ///
    /// The stream is driven on a background thread, which keeps requests
    /// progressing while the caller works; up to `concurrency` finished
    /// batches wait for the caller. The thread uses `futures::executor`, or
    /// with the `tokio` feature the loader's tokio runtime, so datasets built
    /// on tokio-based clients work. Without that feature, datasets whose
    /// futures need a specific runtime must use [`stream`](Self::stream)
    /// from that runtime instead.
    pub fn iter(&mut self) -> AsyncDataLoaderIter<C::Batch> {
        let chunks = self.batch_sampler.batch_indices(self.dataset.len());
        let remaining = chunks.len();
        let stream = self.stream_of(chunks);
        let (tx, rx) = bounded(self.concurrency);
        #[cfg(feature = "tokio")]
        let tokio = self.tokio.handle.clone();

        std::thread::Builder::new()
            .name("dataloader-async".into())
            .spawn(move || {
                let panic_tx = tx.clone();
                let driven = catch_unwind(AssertUnwindSafe(|| {
                    let drive = async move {
                        let mut stream = pin!(stream);
                        while let Some(batch) = stream.next().await {
                            // The receiver is gone: the iterator was dropped.
                            if tx.send(Ok(batch)).is_err() {
                                return;
                            }
                        }
                    };
                    #[cfg(feature = "tokio")]
                    tokio.block_on(drive);
                    #[cfg(not(feature = "tokio"))]
                    futures::executor::block_on(drive);
                }));
                if let Err(payload) = driven {
                    let _ = panic_tx.send(Err(payload));
                }
            })
            .expect("failed to spawn dataloader-async thread");

        AsyncDataLoaderIter { rx, remaining }
    }

    /// Reference to the underlying dataset.
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// Total number of batches for one epoch with current batch settings.
    pub fn batch_len(&self) -> usize {
        let n = self.dataset.len();
        let bs = self.batch_sampler.batch_size();
        if self.batch_sampler.drop_last() {
            n / bs
        } else {
            n.div_ceil(bs)
        }
    }
}

/// Collate inline, or on `pool` so CPU-heavy collation never blocks the
/// executor. A collator panic is re-raised where the stream is polled.
async fn collate<Item, C>(
    collator: Arc<C>,
    pool: Option<Arc<rayon::ThreadPool>>,
    items: Vec<Item>,
) -> Result<C::Batch>
where
    Item: Send + 'static,
    C: Collator<Item>,
{
    let Some(pool) = pool else {
        return collator.collate(items);
    };
    let (tx, rx) = oneshot::channel();
    pool.spawn(move || {
        let _ = tx.send(catch_unwind(AssertUnwindSafe(|| collator.collate(items))));
    });
    match rx.await {
        Ok(Ok(batch)) => batch,
        Ok(Err(payload)) => resume_unwind(payload),
        Err(oneshot::Canceled) => Err("collate task was dropped".into()),
    }
}

/// The tokio runtime that drives [`AsyncDataLoader::iter`] under the
/// `tokio` feature: one supplied by the caller, or one the loader owns for
/// its lifetime (so connection pools survive across epochs).
#[cfg(feature = "tokio")]
struct TokioDriver {
    handle: tokio::runtime::Handle,
    owned: Option<tokio::runtime::Runtime>,
}

#[cfg(feature = "tokio")]
impl TokioDriver {
    fn new(handle: Option<tokio::runtime::Handle>) -> Self {
        if let Some(handle) = handle {
            return Self {
                handle,
                owned: None,
            };
        }
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("dataloader-tokio")
            .enable_all()
            .build()
            .expect("failed to build tokio runtime");
        Self {
            handle: runtime.handle().clone(),
            owned: Some(runtime),
        }
    }
}

#[cfg(feature = "tokio")]
impl Drop for TokioDriver {
    fn drop(&mut self) {
        // Dropping a runtime blocks, which panics inside async code; the
        // loader may well be dropped there.
        if let Some(runtime) = self.owned.take() {
            runtime.shutdown_background();
        }
    }
}

/// Blocking iterator over one epoch of an [`AsyncDataLoader`].
///
/// Dropping it stops the background driver after its current step.
pub struct AsyncDataLoaderIter<B> {
    rx: Receiver<std::thread::Result<Result<B>>>,
    remaining: usize,
}

impl<B> Iterator for AsyncDataLoaderIter<B> {
    type Item = Result<B>;

    fn next(&mut self) -> Option<Result<B>> {
        if self.remaining == 0 {
            return None;
        }
        match self.rx.recv() {
            Ok(Ok(batch)) => {
                self.remaining -= 1;
                Some(batch)
            }
            // The dataset or collator panicked: re-raise here.
            Ok(Err(payload)) => {
                self.remaining = 0;
                resume_unwind(payload)
            }
            Err(_) => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<B> ExactSizeIterator for AsyncDataLoaderIter<B> {}

/// Builder for [`AsyncDataLoader`].
pub struct AsyncDataLoaderBuilder<D, S, C> {
    dataset: D,
    sampler: S,
    collator: C,
    batch_size: usize,
    drop_last: bool,
    concurrency: usize,
    max_concurrency: Option<usize>,
    collate_threads: usize,
    #[cfg(feature = "tokio")]
    tokio_handle: Option<tokio::runtime::Handle>,
}

impl<D, S, C> AsyncDataLoaderBuilder<D, S, C> {
    /// Number of items per batch. Default: `1`.
    pub fn batch_size(mut self, n: usize) -> Self {
        assert!(n > 0, "batch_size must be > 0");
        self.batch_size = n;
        self
    }

    /// Whether to drop a non-full final batch. Default: `false`.
    pub fn drop_last(mut self, v: bool) -> Self {
        self.drop_last = v;
        self
    }

    /// Number of batches fetched at once (each with all its samples in
    /// flight), and of finished batches buffered for [`AsyncDataLoader::iter`].
    /// Default: `4`.
    pub fn concurrency(mut self, n: usize) -> Self {
        assert!(n > 0, "concurrency must be > 0");
        self.concurrency = n;
        self
    }

    /// Fetch at most `n` samples at a time, across batch boundaries.
    ///
    /// Samples are then fetched with one [`AsyncDataset::get`] each, in
    /// sampler order, and regrouped into batches; [`AsyncDataset::get_batch`]
    /// is not used. For datasets that fetch a batch in one request, limit
    /// [`concurrency`](Self::concurrency) instead. Default: no limit.
    pub fn max_concurrency(mut self, n: usize) -> Self {
        assert!(n > 0, "max_concurrency must be > 0");
        self.max_concurrency = Some(n);
        self
    }

    /// Drive [`AsyncDataLoader::iter`] on this tokio runtime instead of one
    /// owned by the loader — e.g. the runtime your clients were created on.
    /// Use a multi-thread runtime: a current-thread runtime's timers and I/O
    /// only advance while that runtime itself is blocked on.
    #[cfg(feature = "tokio")]
    pub fn tokio_handle(mut self, handle: tokio::runtime::Handle) -> Self {
        self.tokio_handle = Some(handle);
        self
    }

    /// Collate on a dedicated rayon pool of `n` threads instead of inline in
    /// the future, so CPU-heavy collation does not block the executor.
    /// Default: `0` (inline).
    pub fn collate_threads(mut self, n: usize) -> Self {
        self.collate_threads = n;
        self
    }

    /// Replace the sampler and preserve all other settings.
    pub fn sampler<S2: Sampler>(self, sampler: S2) -> AsyncDataLoaderBuilder<D, S2, C> {
        AsyncDataLoaderBuilder {
            dataset: self.dataset,
            sampler,
            collator: self.collator,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            concurrency: self.concurrency,
            max_concurrency: self.max_concurrency,
            collate_threads: self.collate_threads,
            #[cfg(feature = "tokio")]
            tokio_handle: self.tokio_handle,
        }
    }

    /// Replace the collator and preserve all other settings.
    pub fn collator<C2>(self, collator: C2) -> AsyncDataLoaderBuilder<D, S, C2> {
        AsyncDataLoaderBuilder {
            dataset: self.dataset,
            sampler: self.sampler,
            collator,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            concurrency: self.concurrency,
            max_concurrency: self.max_concurrency,
            collate_threads: self.collate_threads,
            #[cfg(feature = "tokio")]
            tokio_handle: self.tokio_handle,
        }
    }

    /// Finalize configuration and construct an [`AsyncDataLoader`].
    pub fn build(self) -> AsyncDataLoader<D, S, C>
    where
        D: AsyncDataset,
        S: Sampler,
        C: Collator<D::Item>,
    {
        let collate_pool = (self.collate_threads > 0).then(|| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(self.collate_threads)
                .thread_name(|i| format!("dataloader-collate-{i}"))
                .build()
                .expect("failed to build rayon thread pool");
            Arc::new(pool)
        });
        AsyncDataLoader {
            dataset: Arc::new(self.dataset),
            batch_sampler: BatchSampler::new(self.sampler, self.batch_size, self.drop_last),
            collator: Arc::new(self.collator),
            concurrency: self.concurrency,
            max_concurrency: self.max_concurrency,
            collate_pool,
            #[cfg(feature = "tokio")]
            tokio: TokioDriver::new(self.tokio_handle),
        }
    }
}
