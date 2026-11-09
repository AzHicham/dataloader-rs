//! `PyAsyncDataset`: a Python dataset with `async def __getitem__`, run on an asyncio event
//! loop thread owned by the loader.

use std::ffi::CStr;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyString, PyTuple};
use tokio::sync::oneshot;

use crate::{dataset::AsyncDataset, error::Result};

/// Base class of the datasets whose `__getitem__` (or `__getitems__`) is `async def`.
#[pyclass(
    frozen,
    subclass,
    name = "PyAsyncDatasetBase",
    module = "dataloader_rs"
)]
pub struct PyAsyncDatasetBase;

#[pymethods]
impl PyAsyncDatasetBase {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self
    }
}

/// Starts an event loop on a daemon thread and returns `submit(coroutine)`, which runs the
/// coroutine on it, and `stop()`.
///
/// `stop()` first awaits `aclose()` on the loop, if the dataset defines it, for at most
/// `ACLOSE_TIMEOUT` seconds; an error or a timeout is logged, then the loop stops anyway. The
/// stopped loop cancels the coroutines still pending and waits for them before it closes.
/// `stop()` runs once; `submit` refuses coroutines from then on, so that none is scheduled
/// after the loop has cancelled what was pending.
const EVENT_LOOP: &CStr = cr#"
import asyncio, concurrent.futures, logging, threading

# How long the loader waits for PyAsyncDataset.aclose() before it stops the loop without it.
ACLOSE_TIMEOUT = 5.0

logger = logging.getLogger("dataloader_rs")

async def close(aclose):
    try:
        task = asyncio.ensure_future(aclose())
        done, _ = await asyncio.wait({task}, timeout=ACLOSE_TIMEOUT)
        if not done:
            task.cancel()
            logger.error(
                f"PyAsyncDataset.aclose() did not finish in {ACLOSE_TIMEOUT} s; "
                "the event loop stops without it"
            )
        elif not task.cancelled() and task.exception() is not None:
            error = task.exception()
            logger.error(f"PyAsyncDataset.aclose() failed: {error!r}", exc_info=error)
    except Exception as error:
        logger.error(f"PyAsyncDataset.aclose() failed: {error!r}", exc_info=error)

def start(aclose):
    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
            # Let the coroutines still pending end before the loop closes: run what was
            # queued before the stop, cancel what is left and wait for it, for at most
            # ACLOSE_TIMEOUT seconds. A coroutine left unstarted is never awaited.
            loop.run_until_complete(asyncio.sleep(0))
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()
            if tasks:
                loop.run_until_complete(asyncio.wait(tasks, timeout=ACLOSE_TIMEOUT))
        finally:
            loop.close()

    thread = threading.Thread(target=run, name="dataloader-asyncio", daemon=True)
    thread.start()
    lock = threading.Lock()
    stopped = False

    def stop():
        nonlocal stopped
        with lock:
            if stopped:
                return
            stopped = True
        if not thread.is_alive():
            return
        if aclose is None:
            loop.call_soon_threadsafe(loop.stop)
            return
        future = asyncio.run_coroutine_threadsafe(close(aclose), loop)
        future.add_done_callback(lambda _: loop.call_soon_threadsafe(loop.stop))
        if threading.current_thread() is not thread:
            concurrent.futures.wait([future], timeout=ACLOSE_TIMEOUT + 1)

    def submit(coroutine):
        try:
            with lock:
                if stopped:
                    raise RuntimeError("the loader's event loop is stopped")
                return asyncio.run_coroutine_threadsafe(coroutine, loop)
        except BaseException:
            # Not scheduled: close it so that it is not left unawaited.
            if asyncio.iscoroutine(coroutine):
                coroutine.close()
            raise

    return submit, stop
"#;

/// The loader's event loop: stopped, after `aclose()`, when the loader is dropped
/// ([`PyAsyncItems::close`]), or else when the last user drops it. Stopping twice does nothing.
struct EventLoop {
    submit: Py<PyAny>,
    stop: Py<PyAny>,
}

impl EventLoop {
    fn stop(&self) {
        // Not attached while the interpreter finalizes: the daemon thread dies with it.
        Python::try_attach(|py| {
            let _ = self.stop.call0(py);
        });
    }
}

impl Drop for EventLoop {
    fn drop(&mut self) {
        self.stop();
    }
}

/// A `PyAsyncDataset` as an [`AsyncDataset`]: each `get` creates the coroutine and submits it
/// to the event loop on rayon's global pool, then awaits its result without the interpreter.
pub(crate) struct PyAsyncItems {
    shared: Arc<Shared>,
    batched: bool,
}

/// What a submission needs, shared with the pool threads that make it.
struct Shared {
    dataset: Py<PyAny>,
    event_loop: EventLoop,
}

impl PyAsyncItems {
    /// `dataset` must have passed the checks of `PyDataloader`; `aclose`: whether it defines
    /// `async def aclose()`.
    pub(crate) fn new(
        py: Python<'_>,
        dataset: &Bound<'_, PyAny>,
        batched: bool,
        aclose: bool,
    ) -> PyResult<Self> {
        let module = PyModule::from_code(py, EVENT_LOOP, c"event_loop.py", c"_dataloader_loop")?;
        let aclose = match aclose {
            true => Some(dataset.getattr(intern!(py, "aclose"))?),
            false => None,
        };
        let (submit, stop) = module
            .getattr("start")?
            .call1((aclose,))?
            .extract::<(Py<PyAny>, Py<PyAny>)>()?;
        let shared = Shared {
            dataset: dataset.clone().unbind(),
            event_loop: EventLoop { submit, stop },
        };
        Ok(Self {
            shared: Arc::new(shared),
            batched,
        })
    }

    /// Await `aclose()` and stop the event loop; coroutines submitted later are refused.
    pub(crate) fn close(&self) {
        self.shared.event_loop.stop();
    }

    /// Run `dataset.<method>(arg)` on the event loop.
    ///
    /// The call is made on rayon's global pool, whose threads stay attached to the interpreter
    /// (`pool::attach_global_pool`): attaching the runtime's thread instead would create and
    /// destroy a thread state per item, several times the cost of the rest on free-threaded
    /// builds.
    async fn submit<A>(&self, method: &'static str, arg: A) -> PyResult<Py<PyAny>>
    where
        A: for<'py> IntoPyObject<'py> + Send + 'static,
    {
        let shared = Arc::clone(&self.shared);
        let (tx, rx) = oneshot::channel();
        rayon::spawn(move || {
            // A panic must not escape: rayon aborts the process on a panicking `spawn`.
            let submitted = catch_unwind(AssertUnwindSafe(|| {
                Python::attach(|py| shared.submit(py, method, arg))
            }));
            // Released before the result can end the epoch.
            drop(shared);
            let _ = tx.send(submitted);
        });
        let pending = match rx.await {
            Ok(Ok(pending)) => pending?,
            Ok(Err(panic)) => resume_unwind(panic),
            Err(_) => return Err(PyRuntimeError::new_err("the coroutine was not submitted")),
        };
        pending.await
    }
}

impl Shared {
    /// Call `dataset.<method>(arg)` and run the coroutine it returns on the event loop.
    fn submit<'py, A>(&self, py: Python<'py>, method: &str, arg: A) -> PyResult<Pending>
    where
        A: IntoPyObject<'py>,
    {
        let method = PyString::intern(py, method);
        let coroutine = self.dataset.bind(py).call_method1(method, (arg,))?;
        let future = self.event_loop.submit.bind(py).call1((coroutine,))?;
        let (tx, rx) = oneshot::channel();
        let tx = Mutex::new(Some(tx));
        let done = PyCFunction::new_closure(
            py,
            None,
            None,
            move |args: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| -> PyResult<()> {
                let result = args.get_item(0)?.call_method0("result").map(Bound::unbind);
                if let Some(tx) = tx.lock().unwrap_or_else(PoisonError::into_inner).take() {
                    let _ = tx.send(result);
                }
                Ok(())
            },
        )?;
        future.call_method1(intern!(py, "add_done_callback"), (done,))?;
        Ok(Pending {
            rx,
            future: Some(future.unbind()),
        })
    }
}

/// A coroutine running on the event loop; dropping it cancels the coroutine.
struct Pending {
    rx: oneshot::Receiver<PyResult<Py<PyAny>>>,
    /// The `concurrent.futures.Future`, until it is done.
    future: Option<Py<PyAny>>,
}

impl Future for Pending {
    type Output = PyResult<Py<PyAny>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let result = match Pin::new(&mut self.rx).poll(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Ok(result)) => result,
            Poll::Ready(Err(_)) => Err(PyRuntimeError::new_err(
                "the event loop stopped before the coroutine finished",
            )),
        };
        self.future = None;
        Poll::Ready(result)
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(future) = self.future.take() {
            Python::try_attach(|py| {
                let _ = future.call_method0(py, intern!(py, "cancel"));
            });
        }
    }
}

impl AsyncDataset for PyAsyncItems {
    type Item = Py<PyAny>;

    async fn get(&self, index: usize) -> Result<Py<PyAny>> {
        Ok(self.submit("__getitem__", index).await?)
    }

    async fn get_batch(&self, indices: &[usize]) -> Result<Vec<Py<PyAny>>> {
        if !self.batched {
            let mut items = Vec::with_capacity(indices.len());
            for &index in indices {
                items.push(self.get(index).await?);
            }
            return Ok(items);
        }
        let samples = self.submit("__getitems__", indices.to_vec()).await?;
        let samples = Python::attach(|py| {
            samples
                .bind(py)
                .try_iter()?
                .map(|sample| sample.map(Bound::unbind))
                .collect::<PyResult<Vec<_>>>()
        })?;
        if samples.len() != indices.len() {
            return Err(PyValueError::new_err(format!(
                "__getitems__ returned {} samples for {} indices",
                samples.len(),
                indices.len()
            ))
            .into());
        }
        Ok(samples)
    }

    fn len(&self) -> usize {
        Python::attach(|py| {
            self.shared
                .dataset
                .bind(py)
                .len()
                .unwrap_or_else(|e| panic!("PyAsyncDataset.__len__ failed: {e}"))
        })
    }
}
