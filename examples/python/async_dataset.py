"""A dataset written with ``async def``: each item is an HTTP request made with aiohttp, and
the loader keeps many requests in flight on one event loop instead of one thread each. The
example starts a small local server that answers after 20 ms, so it runs offline. Needs
aiohttp.

Run it with ``python examples/python/async_dataset.py``.
"""

import time

import aiohttp
from local_server import serve

from dataloader_rs import PyAsyncDataset, PyDataloader


def slow_answer(path: str, headers) -> tuple[int, bytes]:
    time.sleep(0.02)  # every answer takes 20 ms, like a remote server
    return 200, f"content of {path}".encode()


class Pages(PyAsyncDataset):
    """Item ``i`` is the body of ``urls[i]``."""

    def __init__(self, urls: list[str]) -> None:
        super().__init__()
        self.urls = urls
        self.session: aiohttp.ClientSession | None = None

    def __len__(self) -> int:
        return len(self.urls)

    async def __getitem__(self, index: int) -> bytes:
        # The loader runs this coroutine on an asyncio event loop of its own, on a background
        # thread. A client session belongs to the loop that created it, so it is created
        # here, on the loader's loop, the first time it is needed.
        if self.session is None:
            self.session = aiohttp.ClientSession()
        async with self.session.get(self.urls[index]) as response:
            response.raise_for_status()
            return await response.read()

    async def aclose(self) -> None:
        # Optional. When the loader is dropped, it awaits aclose() once on its loop, then
        # stops the loop: the place to close the session.
        if self.session is not None:
            await self.session.close()


def main() -> None:
    with serve(slow_answer) as base:
        urls = [f"{base}/page/{i}" for i in range(64)]
        loader = PyDataloader(
            Pages(urls),
            batch_size=16,
            num_workers=2,  # 2 batches are fetched at the same time,
            item_concurrency=8,  # with 8 items of each in flight: 16 requests at once
            transform=lambda index, body: body.decode(),  # runs on the loader's threads
        )
        start = time.perf_counter()
        batches = list(loader)
        elapsed = time.perf_counter() - start
        print(batches[0][:3], "...")
        print(
            f"{sum(map(len, batches))} pages of 20 ms each in {elapsed * 1000:.0f} ms "
            f"with 16 in flight (one at a time: {len(urls) * 20} ms)"
        )
        del loader  # awaits Pages.aclose() on the loader's loop


if __name__ == "__main__":
    main()
