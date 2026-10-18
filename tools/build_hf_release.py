#!/usr/bin/env python3
"""Assemble the public Hugging Face release of JEBS v3 into release/JEBS-4B.

The output holds the sealed v3 artifact byte for byte, the runtime source needed by
`python -m mdtrain predict`, the locked environment, and the hand-written card files.
Every build starts from an empty directory, re-verifies the artifact seal, refuses
private paths or keys, and writes SHA256SUMS over the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdtrain.storage import verify_seal  # noqa: E402

ARTIFACT = ROOT / ".audit/v3-paid-run/decision-v3/runs/v3-finalized"
FREEZE_RECEIPT = ROOT / ".audit/v3-paid-run/decision-v3/runs/v3-finalized-freeze-receipt.json"
CARD = ROOT / "tools/hf_release"
LICENSE = ROOT / ".venv/lib/python3.12/site-packages/transformers-5.19.0.dist-info/licenses/LICENSE"

# destination -> source; directories copy recursively without bytecode caches.
LAYOUT = {
    "model": ARTIFACT,
    "src/mdtrain": ROOT / "src/mdtrain",
    "src/mdprep": ROOT / "src/mdprep",
    "pyproject.toml": ROOT / "pyproject.toml",
    "uv.lock": ROOT / "uv.lock",
    "LICENSE": LICENSE,
    "README.md": CARD / "README.md",
    "NOTICE": CARD / "NOTICE",
    "examples/requests.jsonl": CARD / "examples/requests.jsonl",
    "architecture.svg": CARD / "architecture.svg",
}
FORBIDDEN = ("/Users/", "rpa_", ".runpod-key", "runpod.io")
TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".toml", ".lock", ".txt", ".jinja", ""}


def _copy(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output: Path) -> dict:
    expected = json.loads(FREEZE_RECEIPT.read_text())["artifact_id"]
    if verify_seal(ARTIFACT) != expected:
        raise SystemExit("source artifact does not match its freeze receipt")
    missing = [str(source) for source in LAYOUT.values() if not source.exists()]
    if missing:
        raise SystemExit("missing release input(s):\n  " + "\n  ".join(missing))
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for destination, source in LAYOUT.items():
        _copy(source, output / destination)

    if verify_seal(output / "model") != expected:
        raise SystemExit("copied artifact seal differs from the source artifact")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    # model/ is the sealed artifact; its text holds only pod paths written at training time.
    leaks = [f"{path.relative_to(output)}: {needle}" for path in files
             if path.suffix in TEXT_SUFFIXES and "model" not in path.relative_to(output).parts[:1]
             for needle in FORBIDDEN if needle in path.read_text(errors="ignore")]
    if leaks:
        raise SystemExit("private path or key in release:\n  " + "\n  ".join(leaks))
    sums = [f"{_sha256(path)}  {path.relative_to(output)}" for path in files]
    (output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return {"output": str(output), "artifact_id": expected, "files": len(files) + 1,
            "bytes": sum(path.stat().st_size for path in files)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release/JEBS-4B")
    print(json.dumps(build(parser.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
