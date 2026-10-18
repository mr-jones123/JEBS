"""Compare the trained run against the pointer-only baseline, per slice.

The headline question was how much of the locked accuracy came from JEBS training
rather than the frozen backbone. A single accuracy gap answers it weakly, because
slices differ in how much of their score is available from chance alone. This pairs
each slice's gap with the chance floor for that slice, so a 5 point gap on a slice
whose floor is near its accuracy reads differently from the same gap on an easy one.

Both reports come from the same locked splits and the same harness, so the slices are
directly comparable.

Run: .venv/bin/python tools/compare_baseline.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINED = ROOT / ".audit" / "final-evaluation.json"
BASELINE = ROOT / ".audit" / "pointer-only" / "pointer-only-evaluation.json"
OUTPUT = ROOT / ".audit" / "baseline-comparison.json"
DIMENSIONS = ("modality", "ood_axis", "primitive", "family", "candidate_count", "origin", "source")


def slice_keys(row: dict) -> dict[str, str]:
    provenance = row["provenance"]
    return {
        "candidate_count": str(len(row["question"]["criteria"])),
        "family": provenance.get("family"),
        "modality": row["modality"],
        "source": provenance.get("source"),
        "origin": provenance.get("origin"),
        "ood_axis": provenance.get("ood_axis"),
        "primitive": row["question"]["type"],
    }


def floors(split: str) -> dict[str, dict[str, float]]:
    rows = [json.loads(line) for line
            in (ROOT / "data" / "prepared" / f"{split}.jsonl").read_text().splitlines() if line.strip()]
    grouped: dict[str, dict[str, list[float]]] = {d: defaultdict(list) for d in DIMENSIONS}
    for row in rows:
        keys = slice_keys(row)
        chance = 1.0 / len(row["question"]["criteria"])
        for dimension in DIMENSIONS:
            if keys[dimension] is not None:
                grouped[dimension][keys[dimension]].append(chance)
    return {d: {name: statistics.mean(values) for name, values in slices.items()}
            for d, slices in grouped.items()}


def compare() -> dict:
    trained, baseline = json.loads(TRAINED.read_text()), json.loads(BASELINE.read_text())
    receipt: dict = {
        "trained_artifact": trained["artifact_id"],
        "baseline_artifact": baseline["artifact_id"],
        "trained_temperature": trained["temperature"],
        "baseline_temperature": baseline["temperature"],
        "splits": {},
    }
    for split in trained["splits"]:
        if split not in baseline["splits"]:
            continue
        chance = floors(split)
        t_split, b_split = trained["splits"][split], baseline["splits"][split]
        # Subtracting metrics is only meaningful on the same questions. Without these
        # guards a mismatched report would silently produce a plausible comparison.
        if t_split["view_identity"] != b_split["view_identity"]:
            raise SystemExit(f"{split}: view identities differ, so the reports are not "
                             f"comparable ({t_split['view_identity'][:12]} vs "
                             f"{b_split['view_identity'][:12]})")
        t_metrics = t_split["metrics"]["all"]
        b_metrics = b_split["metrics"]["all"]
        if t_metrics["count"] != b_metrics["count"]:
            raise SystemExit(f"{split}: counts differ, {t_metrics['count']} vs {b_metrics['count']}")
        for dimension in DIMENSIONS:
            t_slices = t_metrics["slices"].get(dimension) or {}
            b_slices = b_metrics["slices"].get(dimension) or {}
            for name, entry in t_slices.items():
                if name in b_slices and b_slices[name]["count"] != entry["count"]:
                    raise SystemExit(f"{split}/{dimension}/{name}: slice counts differ, "
                                     f"{entry['count']} vs {b_slices[name]['count']}")
        entry: dict = {
            "view_identity": t_split["view_identity"],
            "count": t_metrics["count"],
            "trained": {k: t_metrics[k] for k in ("accuracy", "nll", "brier", "ece")},
            "baseline": {k: b_metrics[k] for k in ("accuracy", "nll", "brier", "ece")},
            "accuracy_gap": t_metrics["accuracy"] - b_metrics["accuracy"],
            "dimensions": {},
        }
        for dimension in DIMENSIONS:
            t_slices = t_metrics["slices"].get(dimension) or {}
            b_slices = b_metrics["slices"].get(dimension) or {}
            shared = {}
            for name in t_slices:
                if name not in b_slices or name not in chance.get(dimension, {}):
                    continue
                floor = chance[dimension][name]
                shared[name] = {
                    "count": t_slices[name]["count"],
                    "chance": floor,
                    "trained": t_slices[name]["accuracy"],
                    "baseline": b_slices[name]["accuracy"],
                    "trained_lift": t_slices[name]["accuracy"] / floor,
                    "baseline_lift": b_slices[name]["accuracy"] / floor,
                    "gap": t_slices[name]["accuracy"] - b_slices[name]["accuracy"],
                    "baseline_minus_chance": b_slices[name]["accuracy"] - floor,
                }
            entry["dimensions"][dimension] = shared
        receipt["splits"][split] = entry
    return receipt


def main() -> None:
    if not BASELINE.is_file():
        raise SystemExit(f"baseline report not found at {BASELINE}; pull it first")
    receipt = compare()
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    for split, entry in receipt["splits"].items():
        print(f"\n=== {split}  n={entry['count']} ===")
        print("  %-10s trained %.4f   baseline %.4f   gap %+.4f"
              % ("accuracy", entry["trained"]["accuracy"], entry["baseline"]["accuracy"],
                 entry["accuracy_gap"]))
        print("  %-10s trained %.4f   baseline %.4f"
              % ("ece", entry["trained"]["ece"], entry["baseline"]["ece"]))
        print("  temperature: trained %.3f   baseline %.3f"
              % (receipt["trained_temperature"], receipt["baseline_temperature"]))
        for dimension in ("modality", "ood_axis", "primitive"):
            slices = entry["dimensions"].get(dimension) or {}
            if not slices:
                continue
            print(f"\n  {dimension}")
            print("    %-32s %5s %7s %8s %8s %8s" % ("slice", "n", "chance", "trained", "base", "gap"))
            for name, s in sorted(slices.items(), key=lambda kv: kv[1]["gap"]):
                print("    %-32s %5d %7.4f %8.4f %8.4f %+8.4f"
                      % (name[:32], s["count"], s["chance"], s["trained"], s["baseline"], s["gap"]))
    print(f"\nreceipt: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
