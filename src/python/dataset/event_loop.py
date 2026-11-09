"""The event loop of a loader over a ``PyAsyncDataset``: ``start(aclose)`` runs one on a daemon
thread and returns ``submit(coroutine)`` and ``stop()``. Compiled into the Rust module."""

import asyncio
import concurrent.futures
import logging
import threading

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
