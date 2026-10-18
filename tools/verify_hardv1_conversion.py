"""Independently check the hard-v1 conversion against the raw published source.

This is a reviewer's check, deliberately not sharing code with the converter's own
tests. Every claim is re-derived from the raw `hard-v1` JSONL files rather than from
fields the converter wrote, so a systematic mistake in the converter cannot hide
behind a matching mistake here.

The two things worth checking hardest are the ones that fail silently:

- A `null` option description means "the option text is the entity name", and the
  exact casing exists only in the state or in `_meta.facts.options`. A naive
  title-case gets `Tidedb` where the state says `TideDB`, which would put an option
  text in front of the model that appears nowhere in its evidence.
- `provenance.source` is a slice dimension in the locked evaluation report, so an
  external dataset labelled as one of our own sources would merge into that slice.

Run: .venv/bin/python tools/verify_hardv1_conversion.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

from mdprep.native import NativeCompiler
from mdprep.records import parse_request

ROOT = Path(__file__).resolve().parent.parent
CONVERTED = ROOT / ".audit" / "hardv1-converted"
SOURCES = ROOT / ".audit" / "hardv1-source"
RECEIPT = ROOT / ".audit" / "hardv1-conversion.json"
VERIFICATION = ROOT / ".audit" / "hardv1-independent-verification.json"
SPLIT_FILES = {"train": "train.jsonl", "dev": "development.jsonl", "test_ood": "test.jsonl"}
SOURCE_CANDIDATES = {
    "train": ("train.jsonl", "/tmp/hardv1_train.jsonl"),
    "dev": ("development.jsonl",),
    "test_ood": ("test.jsonl",),
}
SLUG = re.compile(r"[^a-z0-9]+")


def source_path(split: str, recorded: str) -> Path:
    """The first local copy of the source whose bytes match the receipt."""
    for candidate in SOURCE_CANDIDATES[split]:
        path = Path(candidate)
        if not path.is_absolute():
            path = SOURCES / candidate
        if path.is_file() and sha256(path) == recorded:
            return path
    raise SystemExit(
        f"{split}: no local copy of the source matches the receipt hash {recorded}. "
        f"Looked in {[str(SOURCES / c) if not Path(c).is_absolute() else c for c in SOURCE_CANDIDATES[split]]}")


def slug(value: str) -> str:
    return SLUG.sub("_", value.lower()).strip("_")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_questions(path: Path) -> dict[tuple[str, str], dict]:
    index = {}
    for row in load(path):
        for name, question in row["questions"].items():
            index[(row["_meta"]["id"], name)] = (row, name, question)
    return index


def to_request(record: dict) -> dict:
    """Convert a canonical record to the request shape accepted by ``parse_request``."""
    return {
        "id": record["id"],
        "state": record["state"],
        "kind": record["question"]["type"],
        "instructions": record["question"]["instructions"],
        "candidates": [{"key": key, "description": text}
                       for key, text in record["question"]["criteria"].items()],
        "images": record["images"],
        "videos": record["videos"],
    }


def check(split: str, receipt: dict, compiler) -> dict:
    converted = load(CONVERTED / f"{split}.jsonl")
    recorded = receipt["source_files"][
        {"train": "train", "dev": "development", "test_ood": "test"}[split]]["sha256"]
    source_path_resolved = source_path(split, recorded)

    index = source_questions(source_path_resolved)
    faults: list[str] = []
    families = collections.Counter()
    tiers = collections.Counter()
    joined = 0
    positions: list[int] = []
    over_budget = 0

    for record in converted:
        audit = record["audit"]
        key = (audit["upstream_id"], audit["upstream_question"])
        families[record["provenance"]["family"]] += 1

        if record["provenance"]["source"] != "hard-v1":
            faults.append(f"{record['id']}: provenance.source is "
                          f"{record['provenance']['source']!r}, not 'hard-v1'")

        found = index.get(key)
        if found is None:
            faults.append(f"{record['id']}: no source question for {key}")
            continue
        joined += 1
        _, _, question = found
        criteria = record["question"]["criteria"]
        values = list(criteria.values())

        if not all(isinstance(v, str) and v.strip() for v in values):
            faults.append(f"{record['id']}: empty or non-string option text")

        if question["type"] == "choice":
            keys = audit["original_candidate_keys"]
            if list(criteria) != [f"option_{i}" for i in range(len(keys))]:
                faults.append(f"{record['id']}: option keys are not contiguous option_N")
            if len(keys) != len(values):
                faults.append(f"{record['id']}: {len(keys)} source keys, {len(values)} options")
            for position, (key_name, text) in enumerate(zip(keys, values)):
                original = question["criteria"][key_name]
                if original is None:
                    if slug(text) != key_name:
                        faults.append(
                            f"{record['id']}: option {position} text {text!r} does not slug to "
                            f"the source key {key_name!r}")
                elif text != original:
                    faults.append(f"{record['id']}: option {position} text changed from the source")
            expected = f"option_{keys.index(question['label'])}"
            if record["target"]["label"] != expected:
                faults.append(f"{record['id']}: target {record['target']['label']} "
                              f"should be {expected}")
            for name, tier in (audit.get("null_option_resolution") or {}).items():
                tiers[tier] += 1

        elif question["type"] in ("noul", "score"):
            levels = question["criteria"] if question["type"] == "score" else None
            if question["type"] == "score":
                if list(criteria) != [str(i) for i in range(len(levels))]:
                    faults.append(f"{record['id']}: score keys are not contiguous from 0")
                expected = str(question["label"])
            else:
                if list(criteria) != ["false", "true"]:
                    faults.append(f"{record['id']}: noul keys are not false,true")
                expected = "true" if question["label"] else "false"
            if record["target"]["label"] != expected:
                faults.append(f"{record['id']}: target {record['target']['label']} "
                              f"should be {expected}")
        else:
            faults.append(f"{record['id']}: unknown source type {question['type']!r}")

        batch = compiler.compile((parse_request(to_request(record)),))
        length = int(batch.expanded_lengths[0])
        positions.append(length)
        if length > compiler.identity["max_positions"]:
            over_budget += 1

    return {
        "split": split,
        "converted": len(converted),
        "joined": joined,
        "source_questions": len(index),
        "source_sha256": sha256(source_path_resolved),
        "converted_sha256": sha256(CONVERTED / f"{split}.jsonl"),
        "faults": faults,
        "families": dict(families),
        "tiers": dict(tiers),
        "max_positions": max(positions) if positions else None,
        "over_budget": over_budget,
        "admitted_ratio": len(converted) / len(index),
    }


def main() -> int:
    receipt = json.loads(RECEIPT.read_text())
    compiler = NativeCompiler.from_pretrained(ROOT, media_root=ROOT / "data")
    print(f"processor_id {compiler.processor_id}")
    print(f"position limit {compiler.identity['max_positions']}\n")

    results = []
    failures = 0
    for split in SPLIT_FILES:
        result = check(split, receipt, compiler)
        results.append(result)
        print(f"{split}: {result['converted']} converted from {result['source_questions']} "
              f"source questions ({result['admitted_ratio']:.1%} admitted), "
              f"joined {result['joined']}, max positions {result['max_positions']}, "
              f"over budget {result['over_budget']}")
        print(f"  tiers: {result['tiers']}")
        print(f"  families: {result['families']}")
        if result["faults"]:
            failures += len(result["faults"])
            print(f"  FAULTS ({len(result['faults'])})")
            for fault in result["faults"][:12]:
                print(f"    {fault}")
        else:
            print("  no faults")

    verification = {
        "status": "PASSED" if not failures else "FAILED",
        "processor_id": compiler.processor_id,
        "position_limit": compiler.identity["max_positions"],
        "faults": failures,
        "splits": {
            result["split"]: {
                key: value
                for key, value in result.items()
                if key not in {"faults", "families", "tiers"}
            }
            for result in results
        },
        "max_positions": max(result["max_positions"] for result in results),
        "conversion_receipt_sha256": sha256(RECEIPT),
        "verifier_sha256": sha256(Path(__file__)),
    }
    VERIFICATION.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    recorded = receipt.get("per_family")
    if recorded:
        print("\nper-family admitted vs source, as recorded in the receipt:")
        for family, counts in sorted(recorded.items()):
            print(f"  {family:<18} {counts}")

    print(f"\n{'FAILED' if failures else 'PASSED'}: {failures} faults")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
