#!/usr/bin/env python3
"""Score mdtrain prediction JSONL against an external evaluation answer key."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ECE_BINS = 15
PROBABILITY_TOLERANCE = 1e-6


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _index(rows: list[dict], name: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError(f"{name} requires unique nonempty string IDs")
        result[identifier] = row
    return result


def _validate_probability_row(prediction: dict, key: dict) -> tuple[str, float]:
    probabilities = prediction.get("probabilities")
    expected_keys = key.get("candidate_keys")
    if not isinstance(probabilities, dict) or list(probabilities) != expected_keys:
        raise ValueError(f"candidate alignment differs for {key['id']}")
    values = list(probabilities.values())
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 or value > 1 for value in values):
        raise ValueError(f"invalid probabilities for {key['id']}")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=PROBABILITY_TOLERANCE):
        raise ValueError(f"probabilities do not sum to one for {key['id']}")
    predicted = prediction.get("predicted_key")
    computed = max(probabilities, key=probabilities.get)
    if predicted != computed:
        raise ValueError(f"predicted_key disagrees with probabilities for {key['id']}")
    if key.get("label") not in probabilities:
        raise ValueError(f"answer key label is not a candidate for {key['id']}")
    return predicted, float(probabilities[predicted])


def _ece(rows: list[tuple[bool, float]]) -> float:
    bins: list[list[tuple[bool, float]]] = [[] for _ in range(ECE_BINS)]
    for correct, confidence in rows:
        index = min(ECE_BINS - 1, int(confidence * ECE_BINS))
        bins[index].append((correct, confidence))
    total = len(rows)
    return sum(
        len(bucket) / total
        * abs(sum(float(correct) for correct, _ in bucket) / len(bucket)
              - sum(confidence for _, confidence in bucket) / len(bucket))
        for bucket in bins if bucket
    )


def _group_field(dataset: str) -> str:
    if dataset == "rule_probe":
        return "template_variant"
    if dataset == "hard_v1":
        return "family"
    if dataset == "doclaynet":
        return "class"
    if dataset == "rule_structures_heldout":
        return "combinator"
    raise ValueError(f"unknown key dataset: {dataset}")


def score_rows(prediction_rows: list[dict], key_rows: list[dict]) -> dict:
    predictions = _index(prediction_rows, "predictions")
    keys = _index(key_rows, "answer key")
    if set(predictions) != set(keys):
        missing = sorted(set(keys) - set(predictions))
        extra = sorted(set(predictions) - set(keys))
        raise ValueError(f"prediction/key IDs differ; missing={missing[:3]}, extra={extra[:3]}")
    datasets = {row.get("dataset") for row in key_rows}
    if len(datasets) != 1:
        raise ValueError("an answer key must name exactly one dataset")
    dataset = datasets.pop()
    group_field = _group_field(dataset)

    observations = []
    groups: dict[str, list[bool]] = defaultdict(list)
    paired = []
    heldout_pairs: dict[str, dict[str, bool]] = defaultdict(dict)
    for identifier, key in keys.items():
        predicted, confidence = _validate_probability_row(predictions[identifier], key)
        correct = predicted == key["label"]
        observations.append((correct, confidence))
        if dataset == "rule_probe":
            group = f"{key['template']} x {key['variant']}"
            if key["variant"] == "edit_changes_label":
                if key["label"] == key["original_label"]:
                    raise ValueError(f"changed-label key did not change label: {identifier}")
                paired.append((predicted == key["original_label"], correct))
            elif key["variant"] == "edit_keeps_label" and key["label"] != key["original_label"]:
                raise ValueError(f"kept-label key changed label: {identifier}")
        elif dataset == "rule_structures_heldout":
            group = key[group_field]
            variant = key.get("variant")
            if variant not in {"original", "edit_changes_label"}:
                raise ValueError(f"unknown held-out pair variant: {identifier}")
            if variant == "edit_changes_label" and key["label"] == key["original_label"]:
                raise ValueError(f"held-out sibling did not change label: {identifier}")
            base_id = key.get("base_id")
            if not isinstance(base_id, str) or variant in heldout_pairs[base_id]:
                raise ValueError(f"invalid or duplicate held-out pair member: {identifier}")
            heldout_pairs[base_id][variant] = correct
        else:
            group = key[group_field]
        groups[group].append(correct)

    result = {
        "dataset": dataset,
        "questions": len(observations),
        "accuracy": sum(correct for correct, _ in observations) / len(observations),
        "ece_15_bin": _ece(observations),
        "group_dimension": "template x variant" if dataset == "rule_probe" else group_field,
        "per_group_accuracy": {
            group: {"questions": len(values), "accuracy": sum(values) / len(values)}
            for group, values in sorted(groups.items())
        },
    }
    if dataset == "rule_probe":
        result["paired_edit_changes_label"] = {
            "questions": len(paired),
            "rule_ignoring_rate_original_label": sum(ignored for ignored, _ in paired) / len(paired),
            "new_correct_label_rate": sum(correct for _, correct in paired) / len(paired),
        }
    if dataset == "doclaynet":
        counts = Counter(row["class"] for row in key_rows)
        majority_class, majority_count = max(sorted(counts.items()), key=lambda item: item[1])
        baseline = majority_count / len(key_rows)
        if baseline != 0.495:
            raise ValueError(f"DocLayNet majority-class baseline is {baseline:.6f}, expected 0.495000")
        result["majority_class_baseline"] = {
            "class": majority_class,
            "questions": majority_count,
            "accuracy": baseline,
        }
    if dataset == "rule_structures_heldout":
        if any(set(members) != {"original", "edit_changes_label"}
               for members in heldout_pairs.values()):
            raise ValueError("held-out structure keys must contain complete original/edit pairs")
        both_correct = sum(all(members.values()) for members in heldout_pairs.values())
        result["paired_original_and_edit_changes_label"] = {
            "pairs": len(heldout_pairs),
            "both_correct": both_correct / len(heldout_pairs),
        }
    return result


def compare(prediction_sets: list[tuple[str, list[dict]]], key_rows: list[dict]) -> dict:
    if not prediction_sets:
        raise ValueError("at least one prediction set is required")
    names = [name for name, _ in prediction_sets]
    if len(set(names)) != len(names):
        raise ValueError("prediction set names must be unique")
    models = {name: score_rows(rows, key_rows) for name, rows in prediction_sets}
    baseline_name = names[0]
    baseline = models[baseline_name]
    deltas = {}
    for name in names[1:]:
        current = models[name]
        delta = {
            "accuracy": current["accuracy"] - baseline["accuracy"],
            "ece_15_bin": current["ece_15_bin"] - baseline["ece_15_bin"],
        }
        if "paired_edit_changes_label" in current:
            current_paired = current["paired_edit_changes_label"]
            baseline_paired = baseline["paired_edit_changes_label"]
            delta["rule_ignoring_rate_original_label"] = (
                current_paired["rule_ignoring_rate_original_label"]
                - baseline_paired["rule_ignoring_rate_original_label"]
            )
            delta["new_correct_label_rate"] = (
                current_paired["new_correct_label_rate"]
                - baseline_paired["new_correct_label_rate"]
            )
        deltas[name] = delta
        if "paired_original_and_edit_changes_label" in current:
            current_paired = current["paired_original_and_edit_changes_label"]
            baseline_paired = baseline["paired_original_and_edit_changes_label"]
            delta["both_correct"] = current_paired["both_correct"] - baseline_paired["both_correct"]
    return {"comparison_baseline": baseline_name, "models": models, "delta_vs_baseline": deltas}


def _synthetic_key() -> list[dict]:
    rows = []
    specifications = (
        ("base-a", "false", "true"),
        ("base-b", "true", "false"),
    )
    for base_id, original, changed in specifications:
        for variant, label in (("original", original), ("edit_changes_label", changed),
                               ("edit_keeps_label", original)):
            rows.append({
                "id": f"{base_id}-{variant}",
                "dataset": "rule_probe",
                "variant": variant,
                "template": "synthetic_template",
                "base_id": base_id,
                "label": label,
                "original_label": original,
                "candidate_keys": ["false", "true"],
            })
    return rows


def _synthetic_predictions(keys: list[dict], *, always_original: bool) -> list[dict]:
    rows = []
    for key in keys:
        predicted = key["original_label"] if always_original else key["label"]
        other = "true" if predicted == "false" else "false"
        probabilities = {candidate: (0.9 if candidate == predicted else 0.1)
                         for candidate in key["candidate_keys"]}
        rows.append({"id": key["id"], "probabilities": probabilities,
                     "predicted_key": predicted})
    return rows

def _synthetic_structure_key() -> list[dict]:
    return [
        {
            "id": f"structure-{base}-{variant}",
            "dataset": "rule_structures_heldout",
            "variant": variant,
            "base_id": f"structure-{base}",
            "combinator": "all_of",
            "domain": "synthetic",
            "label": label,
            "original_label": original,
            "candidate_keys": ["false", "true"],
        }
        for base, original in ((0, "false"), (1, "true"))
        for variant, label in (("original", original),
                               ("edit_changes_label", "true" if original == "false" else "false"))
    ]


def _key_predictions(keys: list[dict], wrong_ids: set[str] | None = None) -> list[dict]:
    wrong_ids = wrong_ids or set()
    rows = []
    for key in keys:
        predicted = key["label"]
        if key["id"] in wrong_ids:
            predicted = next(candidate for candidate in key["candidate_keys"] if candidate != predicted)
        rows.append({
            "id": key["id"],
            "probabilities": {
                candidate: 0.9 if candidate == predicted else 0.1
                for candidate in key["candidate_keys"]
            },
            "predicted_key": predicted,
        })
    return rows


def self_test() -> dict:
    keys = _synthetic_key()
    result = compare([
        ("all_correct", _synthetic_predictions(keys, always_original=False)),
        ("always_original_label", _synthetic_predictions(keys, always_original=True)),
    ], keys)
    correct = result["models"]["all_correct"]
    ignored = result["models"]["always_original_label"]
    if (correct["accuracy"] != 1.0
            or correct["paired_edit_changes_label"]["rule_ignoring_rate_original_label"] != 0.0
            or correct["paired_edit_changes_label"]["new_correct_label_rate"] != 1.0
            or ignored["accuracy"] != 2 / 3
            or ignored["paired_edit_changes_label"]["rule_ignoring_rate_original_label"] != 1.0
            or ignored["paired_edit_changes_label"]["new_correct_label_rate"] != 0.0):
        raise AssertionError("synthetic scorer expectations failed")
    structure_keys = _synthetic_structure_key()
    structure = score_rows(
        _key_predictions(structure_keys, {"structure-1-edit_changes_label"}),
        structure_keys,
    )
    if structure["paired_original_and_edit_changes_label"] != {
        "pairs": 2,
        "both_correct": 0.5,
    }:
        raise AssertionError("synthetic paired structure scorer expectations failed")
    return {"status": "SELF_TEST_PASSED", **result,
            "rule_structures_heldout_self_test": structure}


def _prediction_argument(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, raw_path = value.split("=", 1)
    else:
        raw_path = value
        name = Path(value).stem
    if not name or not raw_path:
        raise ValueError("prediction argument must be NAME=PATH or PATH")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--predictions", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        if arguments.key or arguments.predictions:
            parser.error("--self-test cannot be combined with --key/--predictions")
        result = self_test()
    else:
        if arguments.key is None or not arguments.predictions:
            parser.error("--key and at least one --predictions are required")
        sets = [(name, read_jsonl(path)) for name, path in
                (_prediction_argument(value) for value in arguments.predictions)]
        result = compare(sets, read_jsonl(arguments.key))
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
