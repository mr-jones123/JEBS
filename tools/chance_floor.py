"""Chance floor and lift over chance for the locked JEBS evaluation.

The locked report records observed accuracy per slice but no floor to compare it
against. A slice at 0.38 accuracy reads as a partial success until you notice its
chance rate is 0.333. This recomputes the floor from the prepared corpus (exact
1/k per question, averaged within each slice) and pairs it with the observed
accuracy, so each slice carries an interpretable lift.

Every slice computes its own n and asserts agreement with the report's n, so a
mismatched grouping fails loudly instead of quietly comparing different sets.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / ".audit" / "final-evaluation.json"
OUTPUT = ROOT / ".audit" / "chance-floor.json"
DIMENSIONS = ("candidate_count", "family", "modality", "source", "origin", "ood_axis", "primitive")


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


def load(split: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / "data" / "prepared" / f"{split}.jsonl").read_text().splitlines() if line.strip()]


def binomial_lower_tail(hits: int, trials: int, probability: float) -> float:
    """P(X <= hits) for X ~ Binomial(trials, probability), exactly."""
    from math import comb
    if trials <= 0:
        return float("nan")
    return sum(comb(trials, k) * probability ** k * (1 - probability) ** (trials - k)
               for k in range(0, max(-1, hits) + 1))


def analyse() -> dict:
    report = json.loads(REPORT.read_text())
    receipt = {
        "report": str(REPORT.relative_to(ROOT)),
        "artifact_id": report["artifact_id"],
        "temperature": report["temperature"],
        "method": "chance is the exact mean of 1/candidate_count over the slice's questions",
        "splits": {},
    }
    for split, body in report["splits"].items():
        rows = load(split)
        observed = body["metrics"]["all"]["slices"]
        groups: dict[str, dict[str, list[float]]] = {
            dimension: defaultdict(list) for dimension in DIMENSIONS}
        for row in rows:
            keys = slice_keys(row)
            floor = 1.0 / len(row["question"]["criteria"])
            for dimension in DIMENSIONS:
                if keys[dimension] is not None:
                    groups[dimension][keys[dimension]].append(floor)

        dimensions = {}
        for dimension in DIMENSIONS:
            slices = observed.get(dimension) or {}
            recorded = {}
            for name, entry in slices.items():
                floors = groups[dimension].get(name)
                if floors is None:
                    continue
                if len(floors) != entry["count"]:
                    raise SystemExit(
                        f"{split}/{dimension}/{name}: computed n={len(floors)} "
                        f"but report n={entry['count']}")
                chance = statistics.mean(floors)
                recorded[name] = {
                    "count": len(floors),
                    "chance": chance,
                    "observed": entry["accuracy"],
                    "lift": entry["accuracy"] / chance,
                    "reference": slice_keys(next(
                        row for row in rows if slice_keys(row)[dimension] == name)),
                }
            dimensions[dimension] = recorded
        by_cardinality = dimensions["candidate_count"]
        hard = [entry for name, entry in by_cardinality.items() if float(name) >= 4]
        hard_count = sum(entry["count"] for entry in hard)
        hard_hits = sum(entry["observed"] * entry["count"] for entry in hard)
        hard_chance = sum(entry["chance"] * entry["count"] for entry in hard) / hard_count

        below = []
        for dimension, slices in dimensions.items():
            for name, entry in slices.items():
                if entry["lift"] >= 1.0:
                    continue
                hits = round(entry["observed"] * entry["count"])
                below.append({
                    "dimension": dimension, "slice": name, "count": entry["count"],
                    "chance": entry["chance"], "observed": entry["observed"],
                    "lift": entry["lift"], "hits": hits,
                    "p_below_chance": binomial_lower_tail(hits, entry["count"], entry["chance"]),
                })
        below.sort(key=lambda item: item["lift"])

        receipt["splits"][split] = {
            "count": body["metrics"]["all"]["count"],
            "observed": body["metrics"]["all"]["accuracy"],
            "overall_chance": statistics.mean(
                1.0 / len(row["question"]["criteria"]) for row in rows),
            "dimensions_not_computed": {
                name: "the report provides this dimension but the prepared records carry no "
                      "equivalent field, so no floor is derived for it"
                for name in observed if name not in DIMENSIONS},
            "hard_subset": {
                "description": "questions with four or more candidates, so chance is at most 0.25",
                "count": hard_count,
                "chance": hard_chance,
                "observed": hard_hits / hard_count,
                "lift": (hard_hits / hard_count) / hard_chance,
            },
            "below_chance": below,
            "dimensions": dimensions,
        }
    return receipt


def main() -> None:
    receipt = analyse()
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    for split, body in receipt["splits"].items():
        print(f"\n{split}: observed {body['observed']:.4f} against chance "
              f"{body['overall_chance']:.4f} over {body['count']} questions "
              f"({body['observed'] / body['overall_chance']:.2f}x)")
        weakest = []
        for dimension, slices in body["dimensions"].items():
            for name, entry in slices.items():
                weakest.append((entry["lift"], dimension, name, entry))
        weakest.sort()
        print("  weakest lift over chance:")
        for lift, dimension, name, entry in weakest[:14]:
            print("    %-16s %-40s n=%-4d chance=%.4f observed=%.4f  %.2fx"
                  % (dimension, name[:40], entry["count"], entry["chance"],
                     entry["observed"], lift))
        print("    weakest beyond 1.6x: %d of %d slices"
              % (sum(1 for lift, _, _, _ in weakest if lift > 1.6), len(weakest)))
        hard = body["hard_subset"]
        print("  hard subset, 4 or more candidates: n=%d chance=%.4f observed=%.4f  %.2fx"
              % (hard["count"], hard["chance"], hard["observed"], hard["lift"]))
        below = body["below_chance"]
        print("  slices below chance: %d" % len(below))
        for item in below[:6]:
            print("    %-16s %-40s n=%-4d chance=%.4f observed=%.4f  %.2fx  p=%.4f"
                  % (item["dimension"], item["slice"][:40], item["count"], item["chance"],
                     item["observed"], item["lift"], item["p_below_chance"]))
    print(f"\nreceipt: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
