#!/usr/bin/env python3
"""
Fetch images over HTTP, decode, transform and collate them: dataloader_rs's
Source.from_urls + ImageCollator against torch.utils.data.DataLoader with
requests, Pillow and torchvision transforms.

A local HTTP/1.1 server (a separate process, keep-alive, threaded) serves 64
lossless WebP images of 224x224 from memory. Every batch is 64 of them,
fetched by URL, decoded to RGB, then ToTensor + Normalize (ImageNet) into a
float32 64x3x224x224 batch. The outputs of both loaders are compared once,
then throughput is measured over whole epochs:

  rs-urls     PyDataloader(Source.from_urls(urls), collate_fn=ImageCollator(...)):
              request, decode, transforms and collation run in Rust with the
              GIL released; connections are reused
  torch       torch DataLoader, Dataset doing requests.Session().get(url) (one
              session per worker process, connections reused), then
              Image.open(...).convert("RGB") + ToTensor + Normalize, default
              collation, persistent worker processes

Reported: items per second (median of --repeats epochs, after one warm-up
epoch), per loader and num_workers.

Needs numpy, Pillow, requests, torch and torchvision. Pin the loader and the
server to different cores; a free-threaded server keeps up best:
  taskset -c 12-19 python bench/sources_http.py --workers 0 2 4 8 \\
      --server-python .venv/bin/python --server-cores 0-7
"""

from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import subprocess
import sys
import sysconfig
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SIDE = 224
BATCH = 64
DISTINCT = 64
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def scene(seed: int):
    """Gradients, a checkerboard and noise: not trivially compressible."""
    import numpy as np

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:SIDE, 0:SIDE].astype(np.float64)
    img = np.stack([255 * x / SIDE, 255 * y / SIDE, 127.5 + 127.5 * np.sin((x + seed) / 9)], -1)
    img[(x // 8 + y // 8) % 2 == 0] *= 0.6
    img += rng.normal(0, 12, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def webp_images(n: int) -> list[bytes]:
    from PIL import Image

    out = []
    for seed in range(n):
        buf = io.BytesIO()
        Image.fromarray(scene(seed), "RGB").save(buf, format="WEBP", lossless=True)
        out.append(buf.getvalue())
    return out


# ── Server (its own process) ─────────────────────────────────────────────────


def serve(images_path: str) -> None:
    """Serve `/<i>` with image `i % DISTINCT` of the file of length-prefixed images; print the
    port, then run until killed."""
    with open(images_path, "rb") as f:
        blob = f.read()
    images, pos = [], 0
    while pos < len(blob):
        size = int.from_bytes(blob[pos : pos + 8], "little")
        images.append(blob[pos + 8 : pos + 8 + size])
        pos += 8 + size

    class Handle(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            body = images[int(self.path.rsplit("/", 1)[-1]) % len(images)]
            self.send_response(200)
            self.send_header("Content-Type", "image/webp")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    class Server(ThreadingHTTPServer):
        request_queue_size = 256
        daemon_threads = True

    server = Server(("127.0.0.1", 0), Handle)
    print(server.server_address[1], flush=True)
    server.serve_forever()


def start_server(images: list[bytes], python: str, cores: str | None, scratch: str):
    path = os.path.join(scratch, "sources_http_images.bin")
    with open(path, "wb") as f:
        for image in images:
            f.write(len(image).to_bytes(8, "little"))
            f.write(image)
    command = [python, "-X", "gil=0", __file__, "--serve", path]
    if cores:
        command = ["taskset", "-c", cores, *command]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    port = int(process.stdout.readline())
    return process, f"http://127.0.0.1:{port}"


# ── Loaders ──────────────────────────────────────────────────────────────────


class TorchUrls:
    """A torch Dataset: requests + Pillow + torchvision, one session per process."""

    def __init__(self, urls: list[str]):
        from torchvision import transforms

        self.urls = urls
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        )
        self.session = None
        self.pid = None

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, index: int):
        import requests
        from PIL import Image

        if self.pid != os.getpid():
            self.session, self.pid = requests.Session(), os.getpid()
        response = self.session.get(self.urls[index], timeout=30)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return self.transform(img)


def make_loader(kind: str, urls: list[str], workers: int):
    if kind == "rs-urls":
        from dataloader_rs import ImageCollator, Normalize, PyDataloader, Source, ToFloat

        collate = ImageCollator([ToFloat(), Normalize(MEAN, STD)], input_size=(SIDE, SIDE))
        return PyDataloader(
            Source.from_urls(urls), batch_size=BATCH, num_workers=workers, collate_fn=collate
        )
    from torch.utils.data import DataLoader

    return DataLoader(
        TorchUrls(urls), batch_size=BATCH, num_workers=workers, persistent_workers=workers > 0
    )


def epoch_rate(loader, length: int) -> float:
    start = time.perf_counter()
    n = 0
    for batch in loader:
        n += len(batch)
    assert n == length, (n, length)
    return length / (time.perf_counter() - start)


def check_outputs(urls: list[str]) -> float:
    import numpy as np

    ours = np.array(next(iter(make_loader("rs-urls", urls[:BATCH], 0))))
    theirs = next(iter(make_loader("torch", urls[:BATCH], 0))).numpy()
    return float(np.abs(ours - theirs).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--batches", type=int, default=8, help="batches per epoch")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--loaders", nargs="+", default=["rs-urls", "torch"])
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--server-cores", help="taskset -c list for the server process")
    parser.add_argument("--scratch", default=".", help="directory for the server's image file")
    parser.add_argument("--json", help="write the results here")
    parser.add_argument("--serve", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        serve(args.serve)
        return

    import torch

    torch.set_num_threads(1)
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    build = f"{sys.version.split()[0]}{'' if gil else 't'}"
    print(f"Python {build} ({'GIL' if gil else 'free-threaded'}), {sysconfig.get_platform()}")
    images = webp_images(DISTINCT)
    size = sum(map(len, images)) / len(images) / 1024
    print(f"{DISTINCT} lossless WebP of {SIDE}x{SIDE}, {size:.0f} KiB each on average, over HTTP")
    server, base = start_server(images, args.server_python, args.server_cores, args.scratch)
    try:
        length = BATCH * args.batches
        urls = [f"{base}/{i}" for i in range(length)]
        print(f"max |rs - torchvision| on one batch: {check_outputs(urls):.3g}")
        results = []
        print(f"\n{'loader':<8} {'workers':>7} {'items/s':>9}")
        for kind in args.loaders:
            for workers in args.workers:
                loader = make_loader(kind, urls, workers)
                epoch_rate(loader, length)  # warm-up: workers, connections, buffer pool
                rate = statistics.median(epoch_rate(loader, length) for _ in range(args.repeats))
                results.append(
                    {"python": build, "loader": kind, "workers": workers, "items_per_s": rate}
                )
                print(f"{kind:<8} {workers:>7} {rate:>9.0f}", flush=True)
                del loader
    finally:
        server.kill()
        server.wait()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
