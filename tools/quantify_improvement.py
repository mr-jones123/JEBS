"""Quantify how much the trained model improved over the frozen-backbone baseline.

Both locked reports score the same questions with the same harness and carry a
percentile group bootstrap over `group_id` with the same seed and resample count, so
the two intervals are comparable even though per-question predictions were not
preserved.

What can and cannot be claimed:

- The intervals are unpaired. The two models answered identical questions, so a paired
  test (McNemar, or a paired group bootstrap) would give a narrower interval than the
  disjointness this checks. Reporting the unpaired comparison is therefore
  conservative, and the real difference is at least as decisive as it looks.
- No per-slice interval exists in either report, so per-slice figures are gaps only,
  with the slice's own chance floor beside them.

Run: .venv/bin/python tools/quantify_improvement.py
     .venv/bin/python tools/quantify_improvement.py --trained NEW.json --baseline OLD.json \
         --trained-locked NEW-locked.tar.gz --baseline-locked OLD-locked.tar.gz --output OUT.json

Reports from different qualification cycles carry different view identities even on the
same questions, because the identity includes the qualification commitments. In that
case the comparison is accepted only when both locked archives contain byte-identical
copies of the split's prepared file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINED = ROOT / ".audit" / "final-evaluation.json"
BASELINE = ROOT / ".audit" / "pointer-only" / "pointer-only-evaluation.json"
OUTPUT = ROOT / ".audit" / "improvement.json"
DIMENSIONS = ("modality", "ood_axis", "primitive", "family", "candidate_count", "origin")


def chance_floors(split: str) -> dict[str, dict[str, float]]:
    rows = [json.loads(line) for line
            in (ROOT / "data" / "prepared" / f"{split}.jsonl").read_text().splitlines() if line.strip()]
    grouped: dict[str, dict[str, list[float]]] = {d: defaultdict(list) for d in DIMENSIONS}
    for row in rows:
        provenance = row["provenance"]
        keys = {
            "candidate_count": str(len(row["question"]["criteria"])),
            "family": provenance.get("family"),
            "modality": row["modality"],
            "origin": provenance.get("origin"),
            "ood_axis": provenance.get("ood_axis"),
            "primitive": row["question"]["type"],
        }
        floor = 1.0 / len(row["question"]["criteria"])
        for dimension in DIMENSIONS:
            if keys[dimension] is not None:
                grouped[dimension][keys[dimension]].append(floor)
    return {d: {name: statistics.mean(v) for name, v in slices.items()}
            for d, slices in grouped.items()}


def locked_split_hash(archive: Path, split: str) -> str:
    with tarfile.open(archive, "r:gz") as bundle:
        return hashlib.sha256(bundle.extractfile(f"data/prepared/{split}.jsonl").read()).hexdigest()


def same_questions(split: str, trained_split: dict, baseline_split: dict,
                   trained_locked: Path | None, baseline_locked: Path | None) -> str:
    if trained_split["view_identity"] == baseline_split["view_identity"]:
        return "identical view identity"
    if trained_locked is None or baseline_locked is None:
        raise SystemExit(f"{split}: view identities differ and no locked archives were given "
                         "to prove the questions are identical")
    left, right = locked_split_hash(trained_locked, split), locked_split_hash(baseline_locked, split)
    if left != right:
        raise SystemExit(f"{split}: locked prepared files differ ({left[:12]} vs {right[:12]})")
    return f"byte-identical locked {split}.jsonl {left}"


def quantify(trained_path: Path = TRAINED, baseline_path: Path = BASELINE,
             trained_locked: Path | None = None, baseline_locked: Path | None = None) -> dict:
    trained = json.loads(trained_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    receipt: dict = {
        "trained_artifact": trained["artifact_id"],
        "baseline_artifact": baseline["artifact_id"],
        "method": "unpaired comparison of two percentile group bootstrap intervals, "
                  "same questions, same group key, same seed, 1000 resamples",
        "splits": {},
    }
    for split in trained["splits"]:
        if split not in baseline["splits"]:
            continue
        t_split, b_split = trained["splits"][split], baseline["splits"][split]
        question_identity = same_questions(split, t_split, b_split, trained_locked, baseline_locked)
        t_all, b_all = t_split["metrics"]["all"], b_split["metrics"]["all"]
        t_ci = t_all["uncertainty"]["intervals"]["accuracy"]
        b_ci = b_all["uncertainty"]["intervals"]["accuracy"]
        floors = chance_floors(split)

        gap = t_all["accuracy"] - b_all["accuracy"]
        t_error, b_error = 1 - t_all["accuracy"], 1 - b_all["accuracy"]
        entry: dict = {
            "question_identity": question_identity,
            "count": t_all["count"],
            "accuracy": {
                "trained": t_all["accuracy"], "baseline": b_all["accuracy"], "gap": gap,
                "trained_interval": t_ci, "baseline_interval": b_ci,
                "intervals_disjoint": t_ci["lower"] > b_ci["upper"],
                "closest_edge_margin": t_ci["lower"] - b_ci["upper"],
                "relative_improvement": t_all["accuracy"] / b_all["accuracy"] - 1,
            },
            "error": {
                "trained": t_error, "baseline": b_error,
                "reduction_fraction": (b_error - t_error) / b_error,
                "baseline_errors_remaining": t_error / b_error,
            },
            "chance": {
                "floor": t_all["count"] and statistics.mean(
                    [1.0 / len(r["question"]["criteria"])
                     for r in (json.loads(l) for l in
                               (ROOT / "data" / "prepared" / f"{split}.jsonl").read_text().splitlines() if l.strip())]),
                "trained_lift": None, "baseline_lift": None,
            },
            "other_metrics": {},
            "dimensions": {},
        }
        floor = entry["chance"]["floor"]
        entry["chance"]["trained_lift"] = t_all["accuracy"] / floor
        entry["chance"]["baseline_lift"] = b_all["accuracy"] / floor

        for metric in ("nll", "brier", "ece"):
            entry["other_metrics"][metric] = {
                "trained": t_all[metric], "baseline": b_all[metric],
                "gap": t_all[metric] - b_all[metric],
                "relative_change": t_all[metric] / b_all[metric] - 1,
            }

        for dimension in DIMENSIONS:
            t_slices = t_all["slices"].get(dimension) or {}
            b_slices = b_all["slices"].get(dimension) or {}
            shared = {}
            for name, t_entry in t_slices.items():
                if name not in b_slices or name not in floors.get(dimension, {}):
                    continue
                slice_floor = floors[dimension][name]
                b_acc = b_slices[name]["accuracy"]
                t_acc = t_entry["accuracy"]
                b_err, t_err = 1 - b_acc, 1 - t_acc
                shared[name] = {
                    "count": t_entry["count"], "chance": slice_floor,
                    "trained": t_acc, "baseline": b_acc,
                    "gap": t_acc - b_acc,
                    "baseline_lift": b_acc / slice_floor,
                    "trained_lift": t_acc / slice_floor,
                    "baseline_error": b_err, "trained_error": t_err,
                    "error_reduction": (b_err - t_err) / b_err if b_err else None,
                }
            entry["dimensions"][dimension] = shared
        receipt["splits"][split] = entry
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", type=Path, default=TRAINED)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--trained-locked", type=Path)
    parser.add_argument("--baseline-locked", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    receipt = quantify(args.trained, args.baseline, args.trained_locked, args.baseline_locked)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    for split, entry in receipt["splits"].items():
        a, e, c = entry["accuracy"], entry["error"], entry["chance"]
        print(f"\n=== {split}  n={entry['count']} ===")
        print(f"  accuracy      {a['baseline']:.4f} -> {a['trained']:.4f}   gap {a['gap']:+.4f}")
        print(f"  intervals     baseline [{a['baseline_interval']['lower']:.4f}, "
              f"{a['baseline_interval']['upper']:.4f}]  trained "
              f"[{a['trained_interval']['lower']:.4f}, {a['trained_interval']['upper']:.4f}]")
        print(f"  disjoint      {a['intervals_disjoint']}  closest edge margin "
              f"{a['closest_edge_margin']:+.4f}")
        print(f"  relative      {a['relative_improvement']:+.1%} accuracy")
        print(f"  error         {e['baseline']:.4f} -> {e['trained']:.4f}  "
              f"{e['reduction_fraction']:.1%} of errors removed, "
              f"{e['baseline_errors_remaining']:.1%} remain")
        print(f"  lift/chance   {c['baseline_lift']:.2f}x -> {c['trained_lift']:.2f}x "
              f"(floor {c['floor']:.4f})")
        for metric, m in entry["other_metrics"].items():
            print(f"  {metric:<13} {m['baseline']:.4f} -> {m['trained']:.4f}  "
                  f"{m['relative_change']:+.1%}")
        for dimension in ("modality", "ood_axis", "primitive"):
            slices = entry["dimensions"].get(dimension) or {}
            if not slices:
                continue
            print(f"  by {dimension}:")
            print("    %-18s %5s %7s %7s %7s %8s %8s %8s"
                  % ("slice", "n", "chance", "base", "trained", "gap", "errRemoved", "lift"))
            for name, s in sorted(slices.items(), key=lambda kv: kv[1]["gap"]):
                reduction = s["error_reduction"]
                print("    %-18s %5d %7.4f %7.4f %7.4f %+8.4f %8s %.2fx->%.2fx"
                      % (name[:18], s["count"], s["chance"], s["baseline"], s["trained"],
                         s["gap"], f"{reduction:.1%}" if reduction is not None else "-",
                         s["baseline_lift"], s["trained_lift"]))
    print(f"\nreceipt: {args.output}")


if __name__ == "__main__":
    main()
