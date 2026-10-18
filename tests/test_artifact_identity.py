"""Guard current processor identity while preserving accepted-artifact history."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mdprep.native import NativeCompiler

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "data" / "processor-contract.json"
ARTIFACT = ROOT / ".audit" / "finalized" / "finalized-model"
IDENTITY_BOUND_SOURCES = (
    ("record_schema_sha256", "src/mdprep/records.py"),
    ("compiler_sha256", "src/mdprep/native.py"),
)

pytestmark = pytest.mark.skipif(
    not CONTRACT.is_file(),
    reason="the current processor contract is not present in this tree",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict:
    document = json.loads(path.read_text())
    return document.get("identity", document)


def identity_faults(root: Path, specification: Path) -> list[str]:
    """Identity-bound sources whose current bytes disagree with a processor spec."""
    frozen = identity(specification)
    faults = []
    for field, relative in IDENTITY_BOUND_SOURCES:
        current = sha256(root / relative)
        if current != frozen[field]:
            faults.append(
                f"{relative} changed after processor qualification: {field} is "
                f"{current}, but the current contract records {frozen[field]}."
            )
    return faults


@pytest.mark.parametrize("field,relative", IDENTITY_BOUND_SOURCES)
def test_identity_bound_source_matches_current_contract(field, relative):
    assert not identity_faults(ROOT, CONTRACT), (
        f"{relative} no longer matches the qualified {field}"
    )


def test_current_compiler_matches_current_contract():
    contract = json.loads(CONTRACT.read_text())
    compiler = NativeCompiler.from_pretrained(ROOT)
    assert compiler.processor_id == contract["processor_id"]
    assert compiler.identity["record_schema_sha256"] == contract["record_schema_sha256"]


def test_accepted_artifact_still_loads_when_identity_matches():
    if not (ARTIFACT / "processor-spec.json").is_file():
        pytest.skip("the accepted artifact is not present in this tree")
    contract_id = json.loads(CONTRACT.read_text())["processor_id"]
    artifact_id = json.loads((ARTIFACT / "processor-spec.json").read_text())["processor_id"]
    if artifact_id != contract_id:
        pytest.skip(
            "accepted-artifact load is inapplicable because processor ids differ: "
            f"accepted artifact {artifact_id}, current contract {contract_id}"
        )
    compiler = NativeCompiler.from_artifact(ARTIFACT, media_root=ROOT / "data")
    assert compiler.processor_id == artifact_id


def test_guard_detects_a_single_character_change(tmp_path):
    """Negative control: one changed identity digest must make the guard fire."""
    spec = json.loads(CONTRACT.read_text())
    field = "record_schema_sha256"
    spec[field] = spec[field][:-1] + ("0" if spec[field][-1] != "0" else "1")
    path = tmp_path / "processor-contract.json"
    path.write_text(json.dumps(spec))

    faults = identity_faults(ROOT, path)

    assert len(faults) == 1
    assert "records.py" in faults[0]
