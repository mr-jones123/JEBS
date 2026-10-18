"""The analysis levers must reproduce their receipts exactly.

Each tool writes a receipt that later work reads as evidence. If a rerun could drift,
the receipt would be a snapshot of one moment rather than a check a reviewer can
repeat, so determinism is a property worth asserting rather than assuming.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LEVERS = (
    ("tools/chance_floor.py", ".audit/chance-floor.json"),
    ("tools/doclaynet_floor.py", ".audit/doclaynet-floor.json"),
)


def run(relative: str) -> None:
    result = subprocess.run([sys.executable, relative], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, f"{relative} failed:\n{result.stdout}\n{result.stderr}"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("tool,receipt", LEVERS)
def test_lever_receipt_is_reproducible(tool, receipt):
    path = ROOT / receipt
    if not path.is_file():
        pytest.skip(f"{receipt} has not been generated in this tree")
    run(tool)
    first = digest(path)
    run(tool)
    assert digest(path) == first, f"{tool} produced a different receipt on rerun"
