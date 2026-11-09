#!/usr/bin/env python3
"""
Tiles over HTTP: torch.utils.data.DataLoader against dataloader_rs, sync and
async datasets.

A mock tile server, aiohttp.web in several processes sharing the port
(SO_REUSEPORT, keep-alive), answers every GET /tile/<i> with a 64 KiB body
after 10 ms. An epoch is 2048 tiles in batches of 64, like the msintuit-ec feature pass. Each
measurement builds a fresh loader (one per slide in the product), so worker
start-up counts, then reads one epoch:

  epoch        loader creation + one epoch, seconds              (median)
  first        loader creation + first batch, seconds            (median)
  cpu          CPU seconds of the process and its workers         (median)

Loaders (requests in flight in brackets):
  torch_w8          torch, 8 worker processes, requests.Session each    [8]
  torch_w16         the same with 16 workers                            [16]
  torch_w32         the same with 32 workers                            [32]
  torch_w64         the same with 64 workers                            [64]
  dlrs_sync_w4_c16  PyDataset, a requests.Session per thread, 64 threads [64]
  dlrs_async_w2_c32 PyAsyncDataset, one aiohttp.ClientSession on the
                    loader's event loop, closed in aclose()              [64]
  dlrs_http_w2_c32  HttpDataset (Rust, no interpreter while fetching)   [64]
  dlrs_http_w8_c1   HttpDataset at torch_w8's concurrency              [8]

requests is the client of the product (prime_dx); every client keeps its
connections alive. aiohttp (and its C extensions) has free-threaded wheels
for 3.14t; the GIL stays off.

Rounds run every loader once, in a rotated order, after one warm-up epoch
each. The server runs on CPUs given by --server-cpus, the loaders on
--loader-cpus.

Run:
  python bench/compare_async.py [--rounds 5] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import resource
import socket
import statistics
import subprocess
import sys
import threading
import time

ITEMS = 2048
BATCH = 64
BODY = 64 << 10
LATENCY = 0.010
HOST = "127.0.0.1"


# ── server ────────────────────────────────────────────────────────────────────


def serve(port: int) -> None:
    """One process of the mock tile server; keep-alive is aiohttp's default."""
    from aiohttp import web

    body = bytes(BODY)

    async def tile(request: web.Request) -> web.Response:
        await asyncio.sleep(LATENCY)
        return web.Response(body=body, content_type="application/octet-stream")

    app = web.Application()
    app.router.add_get("/tile/{index}", tile)
    web.run_app(
        app, host=HOST, port=port, reuse_port=True, backlog=1024, access_log=None, print=None
    )


def start_servers(processes: int, cpus: set[int]) -> tuple[int, list[subprocess.Popen]]:
    with socket.socket() as s:
        s.bind((HOST, 0))
        port = s.getsockname()[1]
    procs = []
    for _ in range(processes):
        proc = subprocess.Popen(
            [sys.executable, __file__, "--serve", str(port)],
            preexec_fn=lambda: os.sched_setaffinity(0, cpus),
        )
        procs.append(proc)
    deadline = time.monotonic() + 10
    while True:
        try:
            socket.create_connection((HOST, port), timeout=1).close()
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
    return port, procs


# ── clients ───────────────────────────────────────────────────────────────────


class KeepAliveGet:
    """GET /tile/<i> through one requests.Session (keep-alive) per thread and process."""

    def __init__(self, port: int):
        self.port = port
        self.local = threading.local()

    def __call__(self, index: int) -> bytes:
        import requests

        session = getattr(self.local, "session", None)
        if session is None or self.local.pid != os.getpid():
            session = self.local.session = requests.Session()
            self.local.pid = os.getpid()
        response = session.get(f"http://{HOST}:{self.port}/tile/{index}")
        if response.status_code != 200:
            raise OSError(f"status {response.status_code}")
        return response.content

    def __getstate__(self):
        return {"port": self.port}

    def __setstate__(self, state):
        self.port = state["port"]
        self.local = threading.local()


def torch_loader(port: int, workers: int):
    import torch.utils.data as tud

    class Tiles(tud.Dataset):
        def __init__(self):
            self.get = KeepAliveGet(port)

        def __len__(self):
            return ITEMS

        def __getitem__(self, index):
            return self.get(index)

    return tud.DataLoader(
        Tiles(),
        batch_size=BATCH,
        num_workers=workers,
        collate_fn=list,
        multiprocessing_context="fork",
    )


def dlrs_sync_loader(port: int, workers: int, concurrency: int):
    from dataloader_rs import PyDataloader, PyDataset

    class Tiles(PyDataset):
        def __init__(self):
            super().__init__()
            self.get = KeepAliveGet(port)

        def __len__(self):
            return ITEMS

        def __getitem__(self, index):
            return self.get(index)

    return PyDataloader(
        Tiles(), batch_size=BATCH, num_workers=workers, item_concurrency=concurrency
    )


def dlrs_async_loader(port: int, workers: int, concurrency: int):
    import aiohttp

    from dataloader_rs import PyAsyncDataset, PyDataloader

    class Tiles(PyAsyncDataset):
        """GET /tile/<i> through one aiohttp.ClientSession, created on the loader's
        event loop by the first item and closed there by aclose() when the loader goes."""

        def __init__(self):
            super().__init__()
            self.session = None

        def __len__(self):
            return ITEMS

        async def __getitem__(self, index):
            if self.session is None:  # no await before the assignment: one session
                connector = aiohttp.TCPConnector(limit=workers * concurrency)
                self.session = aiohttp.ClientSession(connector=connector)
            async with self.session.get(f"http://{HOST}:{port}/tile/{index}") as response:
                if response.status != 200:
                    raise OSError(f"status {response.status}")
                return await response.read()

        async def aclose(self):
            if self.session is not None:
                await self.session.close()

    return PyDataloader(
        Tiles(), batch_size=BATCH, num_workers=workers, item_concurrency=concurrency
    )


def dlrs_http_loader(port: int, workers: int, concurrency: int):
    from dataloader_rs import HttpDataset, PyDataloader

    urls = [f"http://{HOST}:{port}/tile/{i}" for i in range(ITEMS)]
    return PyDataloader(
        HttpDataset(urls), batch_size=BATCH, num_workers=workers, item_concurrency=concurrency
    )


LOADERS = {
    "torch_w8": lambda port: torch_loader(port, 8),
    "torch_w16": lambda port: torch_loader(port, 16),
    "torch_w32": lambda port: torch_loader(port, 32),
    "torch_w64": lambda port: torch_loader(port, 64),
    "dlrs_sync_w4_c16": lambda port: dlrs_sync_loader(port, 4, 16),
    "dlrs_async_w2_c32": lambda port: dlrs_async_loader(port, 2, 32),
    "dlrs_http_w2_c32": lambda port: dlrs_http_loader(port, 2, 32),
    "dlrs_http_w8_c1": lambda port: dlrs_http_loader(port, 8, 1),
}


# ── measurement ───────────────────────────────────────────────────────────────


def cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def measure(make, port: int) -> dict[str, float]:
    gc.collect()
    cpu = cpu_seconds()
    start = time.perf_counter()
    loader = make(port)
    it = iter(loader)
    n = len(next(it))
    first = time.perf_counter() - start
    for batch in it:
        n += len(batch)
    epoch = time.perf_counter() - start
    assert n == ITEMS, n
    del it
    del loader  # an async loader awaits aclose() here
    gc.collect()  # torch joins its workers, so that their CPU time is counted
    return {"epoch": epoch, "first": first, "cpu": cpu_seconds() - cpu}


def cpus(spec: str) -> set[int]:
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--serve", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--servers", type=int, default=4, help="server processes")
    parser.add_argument("--server-cpus", default="12-19")
    parser.add_argument("--loader-cpus", default="0-11")
    parser.add_argument("--only", nargs="*", choices=sorted(LOADERS), help="loaders to run")
    parser.add_argument("--json", help="write every measurement here")
    args = parser.parse_args()
    if args.serve:
        serve(args.serve)
        return

    port, servers = start_servers(args.servers, cpus(args.server_cpus))
    os.sched_setaffinity(0, cpus(args.loader_cpus))
    names = args.only or list(LOADERS)
    gil = "GIL" if getattr(sys, "_is_gil_enabled", lambda: True)() else "free-threaded"
    print(
        f"Python {sys.version.split()[0]} ({gil}), {ITEMS} tiles of {BODY >> 10} KiB, "
        f"{LATENCY * 1000:.0f} ms each, batch {BATCH}, {args.rounds} rounds",
        flush=True,
    )
    results: dict[str, list[dict[str, float]]] = {name: [] for name in names}
    try:
        for name in names:
            measure(LOADERS[name], port)  # warm-up
        for r in range(args.rounds):
            order = names[r % len(names) :] + names[: r % len(names)]
            for name in order:
                results[name].append(measure(LOADERS[name], port))
            print(f"round {r + 1}/{args.rounds} done", flush=True)
    finally:
        for proc in servers:
            proc.terminate()

    print(f"\n{'loader':<20} {'epoch s (min-max)':>22} {'items/s':>9} {'first s':>8} {'cpu s':>7}")
    for name in names:
        runs = results[name]
        epochs = [m["epoch"] for m in runs]
        epoch = statistics.median(epochs)
        print(
            f"{name:<20} {epoch:>8.3f} ({min(epochs):.3f}-{max(epochs):.3f})"
            f" {ITEMS / epoch:>9.0f} {statistics.median(m['first'] for m in runs):>8.3f}"
            f" {statistics.median(m['cpu'] for m in runs):>7.2f}"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"python": sys.version, "gil": gil, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
