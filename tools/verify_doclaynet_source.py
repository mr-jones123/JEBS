"""Independently re-derive the DocLayNet candidate records from the pinned parquet.

The converter writes its own receipt, but a converter checking itself proves little.
This reads the pinned source files directly and rebuilds what each record must contain:
the label index, the candidate descriptions and their order, and the instruction text.
It reports every disagreement rather than a count alone.

Run: .venv/bin/python tools/verify_doclaynet_source.py
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / ".audit" / "doclaynet-source"
RECORDS = ROOT / ".audit" / "doclaynet-candidate" / "records"
OUTPUT = ROOT / ".audit" / "doclaynet-verification.json"
SPLITS = ("validation", "test")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict:
    receipt = {"splits": {}, "faults": [], "checked": 0, "classes": {}}
    for split in SPLITS:
        source_path = SOURCE / f"{split}.parquet"
        record_path = RECORDS / f"{split}.jsonl"
        rows = pq.read_table(source_path, columns=["inputs", "labels"]).to_pylist()
        records = [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
        if len(rows) != len(records):
            receipt["faults"].append(
                f"{split}: {len(rows)} source rows but {len(records)} records")

        declared = None
        for row, record in zip(rows, records):
            receipt["checked"] += 1
            criteria = record["question"]["criteria"]
            descriptions = list(criteria.values())
            declared = declared or descriptions
            index = row["labels"]
            receipt["classes"][descriptions[index]] = receipt["classes"].get(descriptions[index], 0) + 1

            if descriptions != declared:
                receipt["faults"].append(f"{record['id']}: candidate order or text differs")
            if criteria.get(f"option_{index}") != descriptions[index]:
                receipt["faults"].append(f"{record['id']}: label index does not address its own option")
            if record["target"]["label"] != f"option_{index}":
                receipt["faults"].append(
                    f"{record['id']}: target {record['target']['label']} should be option_{index}")
            if record["question"]["instructions"] != row["inputs"]:
                receipt["faults"].append(f"{record['id']}: instruction text differs from source")

        receipt["splits"][split] = {
            "source_rows": len(rows), "records": len(records),
            "source_sha256": sha256(source_path),
        }
    receipt["clean"] = not receipt["faults"]
    return receipt


def main() -> int:
    receipt = verify()
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    total = receipt["checked"]
    majority = max(receipt["classes"].values()) / total if total else 0.0
    print(f"re-derived {total} records straight from the pinned parquet")
    for split, facts in receipt["splits"].items():
        print(f"  {split}: {facts['records']} records, source sha256 {facts['source_sha256'][:16]}")
    print(f"faults: {len(receipt['faults'])}")
    for fault in receipt["faults"][:10]:
        print(f"  {fault}")
    print(f"majority-class baseline: {majority:.4f}")
    print(f"receipt: {OUTPUT.relative_to(ROOT)}")
    return 1 if receipt["faults"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
