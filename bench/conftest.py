"""pytest configuration for benchmark tests.

Adds the bench/ directory to sys.path so test files can import
common directly (matching the style of the standalone bench scripts).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
