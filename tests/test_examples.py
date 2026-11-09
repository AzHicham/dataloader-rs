"""The Python examples of ``examples/python`` run to the end, so they do not rot."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples" / "python"


@pytest.mark.parametrize(
    ("name", "needs"),
    [
        ("rust_transforms", ["torch"]),
        ("mixed_rust_torch", ["torch"]),
        ("torch_transforms", ["torch", "torchvision"]),
    ],
)
def test_example_runs(name, needs):
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
    assert "(4, 3, 224, 224)" in run.stdout
