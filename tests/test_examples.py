"""The Python examples of ``examples/python`` run to the end, so they do not rot.

Each example runs in its own interpreter and must exit with 0 and print the expected text. An
example whose optional dependency (NumPy, aiohttp, torch, torchvision) is missing is skipped.
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples" / "python"


@pytest.mark.parametrize(
    ("name", "needs", "expected"),
    [
        ("basic", [], "10 batches per epoch"),
        ("collate_fn", ["numpy"], "features (4, 2)"),
        ("async_dataset", ["aiohttp"], "64 pages of 20 ms each"),
        ("http_dataset", [], "item 5 took 3 requests"),
        ("files_and_objects", [], "objects: ['dogs #3', 'dogs #4', 'dogs #5']"),
        ("tiles_by_coordinates", ["numpy"], "(2, 3, 224, 224) float32 at [(224, 224), (448, 224)]"),
        ("rust_transforms", ["numpy", "torch"], "(4, 3, 224, 224)"),
        ("mixed_rust_torch", ["numpy", "torch"], "(4, 3, 224, 224)"),
        ("torch_transforms", ["numpy", "torch", "torchvision"], "(4, 3, 224, 224)"),
    ],
)
def test_example_runs(name, needs, expected):
    for module in needs:
        pytest.importorskip(module)
    run = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", str(EXAMPLES / f"{name}.py")],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert expected in run.stdout, run.stdout
