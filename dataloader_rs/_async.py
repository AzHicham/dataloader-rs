"""asyncio support for async datasets (internal).

A dataset whose ``__getitems__`` or ``__getitem__`` is ``async def`` is
fetched on one persistent event loop running on a daemon thread, so many
samples and batches are awaited concurrently without a thread per request.
The loop lives as long as the loader: clients that bind to a loop on first
use (aiohttp sessions, asyncio locks) keep working across epochs.
"""

import asyncio
import threading


class LoopThread:
    """An asyncio event loop running forever on a daemon thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self.loop.run_forever, name="dataloader-asyncio", daemon=True
        )
        self._thread.start()

    def submit(self, coro):
        """Schedule *coro* on the loop; returns a concurrent.futures.Future."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def close(self) -> None:
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)


async def fetch_batch(fetch, indices, batched):
    """Fetch one batch: one ``await fetch(indices)``, or every item concurrently."""
    if batched:
        samples = list(await fetch(indices))
        if len(samples) != len(indices):
            raise ValueError(
                f"__getitems__ returned {len(samples)} samples for {len(indices)} indices"
            )
        return samples
    return list(await asyncio.gather(*(fetch(index) for index in indices)))
