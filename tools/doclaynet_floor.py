"""Floors for the external DocLayNet evaluation set.

Every question offers all 11 region classes, so the uniform chance floor is 1/11, or
9.09%. That number is misleading here. The set is heavily skewed: `Text` alone is
about half of it and two classes never appear at all, so a model that always answers
`Text` scores far above chance without reading the page.

Any accuracy reported on this set has to be read against the majority-class rate, not
against the uniform floor. This records both, plus per-class counts, so the comparison
is available rather than reconstructed later.

Run: .venv/bin/python tools/doclaynet_floor.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / ".audit" / "doclaynet-candidate" / "records"
OUTPUT = ROOT / ".audit" / "doclaynet-floor.json"


def load() -> list[dict]:
    rows = []
    for path in sorted(RECORDS.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["_split"] = path.stem
                rows.append(row)
    return rows


def analyse() -> dict:
    rows = load()
    if not rows:
        raise SystemExit(f"no candidate records under {RECORDS}")
    counts: collections.Counter[str] = collections.Counter()
    per_split: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    options: set[int] = set()
    for row in rows:
        criteria = row["question"]["criteria"]
        options.add(len(criteria))
        index = int(row["target"]["label"].removeprefix("option_"))
        description = list(criteria.values())[index]
        counts[description] += 1
        per_split[row["_split"]][description] += 1

    if len(options) != 1:
        raise SystemExit(f"questions do not share one candidate count: {sorted(options)}")
    candidates = options.pop()
    total = len(rows)
    majority, majority_count = counts.most_common(1)[0]

    return {
        "records": total,
        "per_split": {name: dict(sorted(c.items())) for name, c in sorted(per_split.items())},
        "candidates_per_question": candidates,
        "class_counts": dict(counts.most_common()),
        "distinct_classes_present": len(counts),
        "classes_absent": [name for name in _declared() if name not in counts],
        "uniform_chance": 1.0 / candidates,
        "majority_class": majority,
        "majority_baseline": majority_count / total,
        "reading": (
            "Compare accuracy against majority_baseline, not uniform_chance. A "
            "constant answer beats the uniform floor here."),
    }


def _declared() -> list[str]:
    """The dataset's declared class order, read from any question's options."""
    rows = load()
    return list(rows[0]["question"]["criteria"].values()) if rows else []


def main() -> None:
    receipt = analyse()
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    print(f"{receipt['records']} records, {receipt['candidates_per_question']} candidates each")
    print(f"uniform chance     {receipt['uniform_chance']:.4f}")
    print(f"majority baseline  {receipt['majority_baseline']:.4f} "
          f"by always answering {receipt['majority_class']!r}")
    print(f"classes present    {receipt['distinct_classes_present']}, "
          f"absent {receipt['classes_absent']}")
    print("\nclass counts")
    for name, count in receipt["class_counts"].items():
        print(f"  {name:<18} {count:>4}  {count / receipt['records']:6.1%}")
    print(f"\nreceipt: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
