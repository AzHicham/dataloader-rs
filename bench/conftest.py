"""pytest configuration for benchmark tests.

Adds the bench/ directory to sys.path so test files can import
common directly (matching the style of the standalone bench scripts).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from real_world import PINNED_ALLOCATOR  # noqa: E402


def pytest_report_header(config):
    pinned = all(os.environ.get(k) == v for k, v in PINNED_ALLOCATOR.items())
    if pinned:
        return "allocator: pinned (BENCH.md, section 8)"
    return "allocator: defaults; set real_world.PINNED_ALLOCATOR in the environment to pin it"
