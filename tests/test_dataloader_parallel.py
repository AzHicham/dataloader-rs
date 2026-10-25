"""Parallel DataLoader tests (num_workers > 0).

Tests verify:
  - item coverage (same items as sequential path)
  - drop_last semantics in the parallel path
  - loader reuse across epochs
  - safe early-drop / iterator cancellation
  - custom collate_fn applied correctly in parallel mode
  - large prefetch_depth does not corrupt results
  - a forked child can still iterate a loader the parent already used
  - the interpreter exits cleanly with loaders still alive
"""

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time

import pytest

from dataloader_rs import PyDataloader as DataLoader
from dataloader_rs import PyDataset
from tests.py_dataloader_test_utils import (
    ListDataset,
    all_items,
)

# ── Coverage / correctness ────────────────────────────────────────────────────


def test_par_all_items_covered_2w():
    """N=20, bs=4, num_workers=2 → all 20 items present, none duplicated."""
    n = 20
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=2)
    assert all_items(loader) == list(range(n))


def test_par_all_items_covered_4w():
    """N=20, bs=4, num_workers=4 → all 20 items present, none duplicated."""
    n = 20
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=4)
    assert all_items(loader) == list(range(n))


def test_par_order_matches_sequential():
    """Sequential sampler (default) → parallel output order matches sequential.

    When ordering is deterministic (no shuffle), the parallel path must
    deliver batches in the same order as the sequential path.
    """
    n = 16
    bs = 4
    seq = DataLoader(ListDataset(range(n)), batch_size=bs, num_workers=0)
    par = DataLoader(ListDataset(range(n)), batch_size=bs, num_workers=4)

    seq_batches = list(seq)
    par_batches = list(par)
    assert seq_batches == par_batches, "parallel order must match sequential for sequential sampler"


def test_par_more_workers_than_batches():
    """N=8, bs=2 → 4 batches; num_workers=20 (most workers sit idle).

    edge case: num_workers > number of batches — must still yield all items.
    """
    n = 8
    loader = DataLoader(ListDataset(range(n)), batch_size=2, num_workers=20)
    assert all_items(loader) == list(range(n))


def test_par_empty_dataset():
    """N=0, num_workers=4 → 0 batches, no hang."""
    # edge case: empty dataset with parallel workers
    loader = DataLoader(ListDataset([]), batch_size=4, num_workers=4)
    assert list(loader) == []


def test_par_single_batch():
    """N=4, bs=4, num_workers=4 → exactly 1 batch with all 4 items."""
    loader = DataLoader(ListDataset(range(4)), batch_size=4, num_workers=4)
    batches = list(loader)
    assert len(batches) == 1
    assert sorted(batches[0]) == [0, 1, 2, 3]


def test_par_drop_last():
    """N=11, bs=4, num_workers=4, drop_last=True → floor(11/4)=2 full batches."""
    loader = DataLoader(ListDataset(range(11)), batch_size=4, num_workers=4, drop_last=True)
    batches = list(loader)
    assert len(batches) == 2
    for b in batches:
        assert len(b) == 4, "every batch must be full when drop_last=True"


def test_par_multi_epoch_consistent():
    """A parallel loader used over 5 epochs must cover the same items each time."""
    n = 16
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=2)
    first = all_items(loader)
    for _ in range(4):
        assert all_items(loader) == first, "items must be identical across epochs"


# ── Cancellation / early drop ─────────────────────────────────────────────────


def test_par_early_drop_no_hang():
    """Consume 1 batch then discard the iterator; the next full epoch must work.

    If there is a deadlock in the worker shutdown path this test hangs — a
    clear signal of the bug.
    """
    n = 20
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=2)

    # Consume just one batch.
    it = iter(loader)
    next(it)
    del it  # drop iterator with workers potentially still running

    # A new full epoch must complete correctly.
    assert all_items(loader) == list(range(n))


def test_par_early_drop_zero_consumed():
    """Discard the iterator immediately without consuming anything.

    edge case: zero batches consumed before drop.
    """
    n = 20
    loader = DataLoader(ListDataset(range(n)), batch_size=4, num_workers=4)

    it = iter(loader)
    del it  # dropped before any next() calls

    assert all_items(loader) == list(range(n))


# ── Collate function ──────────────────────────────────────────────────────────


def test_par_with_collate_fn():
    """Custom collate_fn must be applied correctly in the parallel path."""

    def collate_fn(items):
        return sum(items)  # reduce the batch to a single integer

    n = 12
    bs = 4
    loader = DataLoader(ListDataset(range(n)), batch_size=bs, num_workers=2, collate_fn=collate_fn)
    sums = list(loader)
    # batch 0: 0+1+2+3=6, batch 1: 4+5+6+7=22, batch 2: 8+9+10+11=38
    assert sums == [6, 22, 38]


# ── Prefetch depth ────────────────────────────────────────────────────────────


def test_par_large_prefetch():
    """prefetch_depth=100 >> number of batches; results must still be correct."""
    # edge case: extremely large prefetch_depth
    n = 8
    loader = DataLoader(ListDataset(range(n)), batch_size=2, num_workers=2, prefetch_depth=100)
    assert all_items(loader) == list(range(n))


# ── fork ──────────────────────────────────────────────────────────────────────


def _count_batches_in_child(loader, queue):
    queue.put(len(list(loader)))


@pytest.mark.skipif(sys.platform != "linux", reason="fork start method is Linux-only here")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_par_forked_child_can_iterate():
    """The child inherits the worker pool without its threads; it must rebuild it."""
    loader = DataLoader(ListDataset(range(20)), batch_size=4, num_workers=2)
    assert len(list(loader)) == 5  # parent's pool threads now exist

    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    child = ctx.Process(target=_count_batches_in_child, args=(loader, queue))
    child.start()
    child.join(30)
    if child.is_alive():
        child.kill()
        pytest.fail("forked child hung iterating the loader")
    assert child.exitcode == 0
    assert queue.get(timeout=5) == 5


def _iterate_inherited(it, queue):
    try:
        queue.put(("finished", len(list(it))))
    except RuntimeError as e:
        queue.put(("RuntimeError", str(e)))


@pytest.mark.skipif(sys.platform != "linux", reason="fork start method is Linux-only here")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(("num_workers", "items_in_parallel"), [(2, 1), (0, 2)])
def test_par_iterator_from_before_fork_raises_in_child(num_workers, items_in_parallel):
    """An epoch started in the parent cannot run in a forked child: it raises instead of hanging."""
    loader = DataLoader(
        ListDataset(range(64)),
        batch_size=4,
        num_workers=num_workers,
        items_in_parallel=items_in_parallel,
    )
    it = iter(loader)
    next(it)

    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    child = ctx.Process(target=_iterate_inherited, args=(it, queue))
    child.start()
    child.join(30)
    if child.is_alive():
        child.kill()
        pytest.fail("forked child hung on an iterator from its parent")
    kind, detail = queue.get(timeout=5)
    assert kind == "RuntimeError" and "fork()" in detail


# ── interpreter shutdown ──────────────────────────────────────────────────────

_DS = """
from dataloader_rs import PyDataloader, PyDataset

class Ds(PyDataset):
    def __len__(self):
        return 64

    def __getitem__(self, index):
        return index
"""

_EXIT_SCRIPTS = {
    # Pool threads start while the interpreter shuts down: the process's
    # first attach from Rust happens during finalization.
    "fresh_loader": _DS + "loader = PyDataloader(Ds(), batch_size=4, num_workers=4)\nprint('ok')\n",
    # Pools whose threads hold a thread state, one epoch left unfinished.
    "used_loaders": _DS
    + """
loaders = [PyDataloader(Ds(), batch_size=4, num_workers=2, items_in_parallel=k) for k in (1, 3)]
for loader in loaders:
    assert sum(len(batch) for batch in loader) == 64
it = iter(loaders[0])
next(it)
print("ok")
""",
}


@pytest.mark.parametrize("script", sorted(_EXIT_SCRIPTS))
def test_par_interpreter_exits_cleanly_with_live_loaders(script):
    """Pool threads attach to the interpreter; shutdown must neither hang, abort nor panic."""
    result = subprocess.run(
        [sys.executable, "-c", _EXIT_SCRIPTS[script]], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    assert "Fatal Python error" not in result.stderr
    assert "panicked" not in result.stderr


# ── signals ───────────────────────────────────────────────────────────────────

_SLOW_BATCH = """
from dataloader_rs import PyDataloader, PyDataset
import time

class Slow(PyDataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        time.sleep(30)
        return index

it = iter(PyDataloader(Slow(), num_workers=1))
print("waiting", flush=True)
try:
    next(it)
except KeyboardInterrupt:
    print("interrupted", flush=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_par_ctrl_c_interrupts_a_wait_for_the_next_batch():
    """SIGINT while next() waits on a slow batch raises KeyboardInterrupt within moments."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _SLOW_BATCH],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "waiting"
        time.sleep(0.3)
        start = time.monotonic()
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=10)
    finally:
        proc.kill()
    assert out.strip() == "interrupted", err
    assert time.monotonic() - start < 2


# ── thread stacks ─────────────────────────────────────────────────────────────


class _DeepJson(PyDataset):
    """Parses JSON nested 20 000 levels deep: within a main thread's stack on CPython 3.14+."""

    def __len__(self):
        return 1

    def __getitem__(self, index):
        try:
            json.loads("[" * 20_000 + "]" * 20_000)
        except RecursionError:
            return "RecursionError"
        return "ok"


def test_par_workers_recurse_as_deep_as_the_main_thread():
    """A worker's stack is a main thread's: deep recursion succeeds, or fails, the same way."""
    expected = _DeepJson()[0]
    assert list(DataLoader(_DeepJson(), num_workers=1)) == [[expected]]


_ENDLESS_SAMPLER = """
import itertools
from dataloader_rs import PyDataloader, PyDataset

class Ds(PyDataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return index

loader = PyDataloader(Ds(), sampler=itertools.repeat(0))
print("reading", flush=True)
try:
    iter(loader)
except KeyboardInterrupt:
    print("interrupted", flush=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_par_ctrl_c_stops_an_endless_sampler():
    """iter() reads the whole epoch; Ctrl-C stops a sampler that never ends."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _ENDLESS_SAMPLER],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "reading"
        time.sleep(0.2)
        start = time.monotonic()
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=10)
    finally:
        proc.kill()
    assert out.strip() == "interrupted", err
    assert time.monotonic() - start < 2


_POOLS_AT_EXIT = """
import gc, os, time
import dataloader_rs.dataloader_rs as ext
from dataloader_rs import PyDataloader, PyDataset

class Ds(PyDataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return index

def threads():
    return len(os.listdir("/proc/self/task"))

loader = PyDataloader(Ds(), batch_size=4, num_workers=3)
assert sum(len(batch) for batch in loader) == 8
before = threads()
ext._keep_pools_alive()  # what atexit runs
del loader
gc.collect()
time.sleep(0.3)
print(before, threads())
"""


@pytest.mark.skipif(not os.path.isdir("/proc/self/task"), reason="counts threads in /proc")
def test_par_exit_hook_keeps_pools_parked():
    """After the exit hook, freeing a loader leaves its threads parked instead of waking them."""
    result = subprocess.run(
        [sys.executable, "-c", _POOLS_AT_EXIT], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    before, after = map(int, result.stdout.split())
    assert after == before
