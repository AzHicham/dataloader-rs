use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::python::collator::{PyBatch, PyCollator};
use crate::python::dataloader::PyDataloader;
use crate::python::dataset::{Fetcher, PyDataset};
use crate::python::into_py_err;

type CorePyDataloaderIter = crate::loader::DataLoaderIter<PyDataset, PyCollator>;

pub(crate) enum PyIterInner {
    /// Sequential (num_workers=0): call Python directly in `__next__` using the
    /// already-held `py` token — no thread, no channel, no extra GIL acquire.
    Direct {
        chunks: std::vec::IntoIter<Vec<usize>>,
        remaining: usize,
        /// `__getitems__` or cached `__getitem__`, resolved once per epoch.
        fetcher: Fetcher,
        /// Collator cloned at `__iter__` time so `__next__` skips borrowing `_owner`.
        collator: PyCollator,
    },
    /// Parallel (num_workers>0): threaded prefetch with crossbeam channel.
    Threaded(Option<CorePyDataloaderIter>),
    /// Async dataset: up to `window` batches awaited concurrently on the
    /// loader's event loop thread, returned in sampler order.
    Async {
        chunks: std::vec::IntoIter<Vec<usize>>,
        remaining: usize,
        /// `concurrent.futures.Future`s of submitted batches, in order.
        pending: VecDeque<Py<PyAny>>,
        window: usize,
        /// `dataloader_rs._async.LoopThread`.
        runner: Py<PyAny>,
        /// `dataloader_rs._async.fetch_batch`.
        fetch_batch: Py<PyAny>,
        /// `asyncio.Semaphore` capping dataset calls in flight, if any.
        limit: Option<Py<PyAny>>,
        /// Bound async `__getitems__` (batched) or `__getitem__`.
        method: Py<PyAny>,
        batched: bool,
        collator: PyCollator,
    },
}

#[pyclass(name = "PyDataloaderIter", module = "dataloader_rs", unsendable)]
pub struct PyDataloaderIter {
    /// Keeps the `PyDataloader` alive while the iterator exists.
    pub(crate) _owner: Py<PyDataloader>,
    pub(crate) inner: PyIterInner,
}

#[pymethods]
impl PyDataloaderIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &mut self.inner {
            PyIterInner::Direct {
                chunks,
                remaining,
                fetcher,
                collator,
            } => {
                let Some(chunk) = chunks.next() else {
                    return Ok(None);
                };
                *remaining -= 1;

                // We already hold `py` — call Python directly with no extra GIL acquire.
                let items = fetcher.fetch(py, &chunk)?;

                let batch = collator.collate_with_py(py, items).map_err(into_py_err)?;

                let out = match batch {
                    PyBatch::Ready(obj) => obj,
                    PyBatch::Items(items) => PyList::new(py, items)?.unbind().into_any(),
                };
                Ok(Some(out))
            }

            PyIterInner::Async {
                chunks,
                remaining,
                pending,
                window,
                runner,
                fetch_batch,
                limit,
                method,
                batched,
                collator,
            } => {
                let submit = |py: Python<'_>, indices: Vec<usize>| -> PyResult<Py<PyAny>> {
                    let coro =
                        fetch_batch.call1(py, (&*method, indices, *batched, limit.as_ref()))?;
                    runner.call_method1(py, intern!(py, "submit"), (coro,))
                };
                // Keep the window full: submit batches before waiting.
                while pending.len() < *window
                    && let Some(indices) = chunks.next()
                {
                    pending.push_back(submit(py, indices)?);
                }
                // The batch being awaited counts towards the window; the next
                // call refills it.
                let Some(future) = pending.pop_front() else {
                    return Ok(None);
                };
                *remaining -= 1;

                // `Future.result()` waits on a lock, releasing the GIL, and
                // re-raises the dataset's exception with its own type.
                let samples = future.call_method0(py, intern!(py, "result"))?;
                let items = samples
                    .bind(py)
                    .try_iter()?
                    .map(|sample| sample.map(Bound::unbind))
                    .collect::<PyResult<Vec<_>>>()?;
                let batch = collator.collate_with_py(py, items).map_err(into_py_err)?;
                let out = match batch {
                    PyBatch::Ready(obj) => obj,
                    PyBatch::Items(items) => PyList::new(py, items)?.unbind().into_any(),
                };
                Ok(Some(out))
            }

            PyIterInner::Threaded(inner_opt) => {
                let Some(inner) = inner_opt.take() else {
                    return Ok(None);
                };
                // Release Python thread state while blocked on the channel.
                // A worker panic is re-raised here; join the workers before
                // re-attaching, so none is left waiting on our GIL.
                let next_item = py.detach(move || {
                    let mut inner = inner;
                    let next_item = catch_unwind(AssertUnwindSafe(|| inner.next()));
                    match next_item {
                        Ok(next_item) => Ok((next_item, inner)),
                        Err(payload) => {
                            drop(inner);
                            Err(payload)
                        }
                    }
                });
                let (next_item, inner) = match next_item {
                    Ok(next_item) => next_item,
                    Err(payload) => resume_unwind(payload),
                };
                match next_item {
                    Some(Ok(batch)) => {
                        let out = match batch {
                            PyBatch::Ready(obj) => obj,
                            PyBatch::Items(items) => PyList::new(py, items)?.unbind().into_any(),
                        };
                        *inner_opt = Some(inner);
                        Ok(Some(out))
                    }
                    // A failed batch does not end the epoch: the next call
                    // returns the following batch, as with num_workers=0.
                    Some(Err(e)) => {
                        *inner_opt = Some(inner);
                        Err(into_py_err(e))
                    }
                    None => {
                        py.detach(|| drop(inner));
                        Ok(None)
                    }
                }
            }
        }
    }

    fn __len__(&self) -> usize {
        match &self.inner {
            PyIterInner::Direct { remaining, .. } | PyIterInner::Async { remaining, .. } => {
                *remaining
            }
            PyIterInner::Threaded(inner) => inner.as_ref().map_or(0, |it| it.len()),
        }
    }
}

impl Drop for PyDataloaderIter {
    /// Joining the workers must happen with the thread state detached: a
    /// worker blocked in `Python::attach` (inside `__getitem__` or
    /// `collate_fn`) can only finish once this thread lets go of the GIL.
    fn drop(&mut self) {
        match &mut self.inner {
            PyIterInner::Threaded(inner) => {
                if let Some(inner) = inner.take() {
                    Python::attach(|py| py.detach(|| drop(inner)));
                }
            }
            // Batches fetched ahead are no longer wanted.
            PyIterInner::Async { pending, .. } => Python::attach(|py| {
                preserving_exception(py, || {
                    for future in pending.drain(..) {
                        let _ = future.call_method0(py, intern!(py, "cancel"));
                    }
                });
            }),
            PyIterInner::Direct { .. } => {}
        }
    }
}

/// Run `f`, which calls into Python, from a destructor.
///
/// A destructor can run while an exception is propagating (e.g. the
/// temporary in `next(iter(loader))` is freed as `next` raises). Calling
/// Python with that exception still set is invalid and turns it into a
/// `SystemError`, so set it aside and restore it afterwards.
pub(crate) fn preserving_exception(py: Python<'_>, f: impl FnOnce()) {
    let pending = PyErr::take(py);
    f();
    if let Some(err) = pending {
        err.restore(py);
    }
}
