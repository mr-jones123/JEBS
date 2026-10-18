#!/usr/bin/env python3
"""Build counterfactual rule probes and external evaluation request/key files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdprep.common import stable_id
from mdprep.native import MAX_POSITIONS, NativeCompiler
from mdprep.records import parse_request
from mdprep.text import _policy_choice_row, _policy_noul_row, _policy_score_row

BASES_PER_TEMPLATE = 120
START_INDEX = 1_000_000
VARIANTS = ("original", "edit_changes_label", "edit_keeps_label")
TEMPLATES = (
    "id_transfer_release_constraints",
    "id_card_purchase_all_constraints",
    "id_incident_additive_severity",
)


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _request(row: dict, *, identifier: str | None = None, state: str | None = None) -> dict:
    question = row["question"]
    request = {
        "id": identifier or row["id"],
        "state": row["state"] if state is None else state,
        "kind": question["type"],
        "instructions": question["instructions"],
        "candidates": [
            {"key": key, "description": description}
            for key, description in question["criteria"].items()
        ],
        "images": row["images"],
        "videos": row["videos"],
    }
    parse_request(request)
    return request


def _choice_key(row: dict, semantic_label: str) -> str:
    return f"option_{row['audit']['original_candidate_keys'].index(semantic_label)}"


def _evidence_suffix(state: str, marker: str) -> str:
    before, separator, after = state.partition(marker)
    if not separator or not before or not after:
        raise ValueError(f"state does not contain evidence marker {marker!r}")
    return separator + after


def transfer_label(facts: dict, parameters: dict) -> str:
    if None in (facts["identity_verified"], facts["recipient_age_days"], facts["amount_usd"]):
        return parameters["missing_label"]
    if ((parameters["decline_on_failed_identity"] and facts["identity_verified"] is False)
            or facts["amount_usd"] > parameters["decline_above_usd"]):
        return "decline"
    if (facts["identity_verified"] is True
            and facts["recipient_age_days"] >= parameters["minimum_age_days"]
            and facts["amount_usd"] <= parameters["allow_ceiling_usd"]):
        return "allow"
    return "review"


def _transfer_rule(parameters: dict) -> str:
    identity_decline = ("identity failed" if parameters["decline_on_failed_identity"]
                        else "identity status is not provided")
    missing = ("insufficient" if parameters["missing_label"] == "insufficient"
               else "sent for manual review")
    return (
        "Identity status, recipient age, and amount are required. Decline if "
        f"{identity_decline} or amount exceeds USD {parameters['decline_above_usd']:,}. "
        "Allow when identity is verified, the recipient is at least "
        f"{parameters['minimum_age_days']} days old, and amount is at most USD "
        f"{parameters['allow_ceiling_usd']:,}. Send any other complete request for manual "
        f"review. Missing required evidence is {missing}."
    )


def _transfer_edits(row: dict) -> tuple[tuple[str, str], tuple[str, str]]:
    facts = row["audit"]["facts"]
    original = row["audit"]["original_target_key"]
    standard = {
        "allow_ceiling_usd": 1000,
        "decline_above_usd": 2000,
        "minimum_age_days": 2,
        "decline_on_failed_identity": True,
        "missing_label": "insufficient",
    }
    if transfer_label(facts, standard) != original:
        raise AssertionError("stored transfer label disagrees with fixed rule")
    changed = dict(standard)
    if original == "allow":
        changed["allow_ceiling_usd"] = facts["amount_usd"] - 1
    elif original == "review" and facts["recipient_age_days"] in (0, 1):
        changed["minimum_age_days"] = facts["recipient_age_days"]
    elif original == "review":
        changed["allow_ceiling_usd"] = facts["amount_usd"]
    elif original == "decline" and facts["identity_verified"] is False:
        changed["decline_on_failed_identity"] = False
    elif original == "decline":
        changed["decline_above_usd"] = facts["amount_usd"]
    else:
        changed["missing_label"] = "review"
    changed_label = transfer_label(facts, changed)

    kept = dict(standard)
    kept["allow_ceiling_usd"] = 1001 if original == "allow" else 999
    kept_label = transfer_label(facts, kept)
    return (_transfer_rule(changed), changed_label), (_transfer_rule(kept), kept_label)


def card_label(facts: dict, parameters: dict) -> str:
    checks = (
        facts["card_active"] is parameters["required_card_active"],
        facts["country_supported"] is parameters["required_country_supported"],
        facts["amount_usd"] <= facts["daily_remaining_usd"] + parameters["limit_offset_usd"],
    )
    return str(all(checks)).lower()


def _card_rule(parameters: dict) -> str:
    card = "active" if parameters["required_card_active"] else "inactive"
    country = "supported" if parameters["required_country_supported"] else "unsupported"
    offset = parameters["limit_offset_usd"]
    if offset == 0:
        limit = "the remaining daily limit"
    elif offset > 0:
        limit = f"the remaining daily limit plus USD {offset}"
    else:
        limit = f"the remaining daily limit minus USD {-offset}"
    return (
        f"A card purchase is authorized only when the card is {card}, the purchase country "
        f"is {country}, and the purchase amount does not exceed {limit}."
    )


def _card_edits(row: dict) -> tuple[tuple[str, str], tuple[str, str]]:
    facts = row["audit"]["facts"]
    original = row["target"]["label"]
    standard = {
        "required_card_active": True,
        "required_country_supported": True,
        "limit_offset_usd": 0,
    }
    if card_label(facts, standard) != original:
        raise AssertionError("stored card label disagrees with fixed rule")
    changed = dict(standard)
    if original == "true":
        changed["required_card_active"] = False
    elif not facts["card_active"]:
        changed["required_card_active"] = False
    elif not facts["country_supported"]:
        changed["required_country_supported"] = False
    else:
        changed["limit_offset_usd"] = facts["amount_usd"] - facts["daily_remaining_usd"]
    changed_label = card_label(facts, changed)

    kept = dict(standard)
    amount_failure = facts["amount_usd"] > facts["daily_remaining_usd"]
    kept["limit_offset_usd"] = -1 if amount_failure else 1
    kept_label = card_label(facts, kept)
    return (_card_rule(changed), changed_label), (_card_rule(kept), kept_label)


def score_label(facts: dict, parameters: dict) -> str:
    points = (
        parameters["unauthorized_weight"] * int(facts["unauthorized"])
        + parameters["missing_weight"] * int(facts["card_missing"])
        + parameters["amount_weight"] * int(facts["amount_usd"] >= parameters["amount_threshold_usd"])
        + parameters["repeated_weight"] * int(facts["repeated"])
    )
    return str(min(parameters["cap"], points))


def _score_rule(parameters: dict) -> str:
    return (
        f"Add {parameters['unauthorized_weight']} points for an unauthorized transaction; add "
        f"{parameters['missing_weight']} for a missing card, {parameters['amount_weight']} for "
        f"amount at least USD {parameters['amount_threshold_usd']:g}, and "
        f"{parameters['repeated_weight']} for repeated occurrence. Cap the total at severity "
        f"{parameters['cap']}."
    )


def _present_one_point_factor(facts: dict) -> str:
    present = []
    if facts["card_missing"]:
        present.append("missing_weight")
    if facts["amount_usd"] >= 500:
        present.append("amount_weight")
    if facts["repeated"]:
        present.append("repeated_weight")
    if len(present) != 1:
        raise AssertionError(f"expected exactly one one-point factor, got {present}")
    return present[0]


def _score_edits(row: dict) -> tuple[tuple[str, str], tuple[str, str]]:
    facts = row["audit"]["facts"]
    original = row["target"]["label"]
    standard = {
        "unauthorized_weight": 2,
        "missing_weight": 1,
        "amount_weight": 1,
        "repeated_weight": 1,
        "amount_threshold_usd": 500,
        "cap": 3,
    }
    if score_label(facts, standard) != original:
        raise AssertionError("stored score label disagrees with fixed rule")
    changed = dict(standard)
    if original == "0":
        changed["amount_threshold_usd"] = facts["amount_usd"]
    elif original == "1":
        changed[_present_one_point_factor(facts)] = 2
    else:
        changed["unauthorized_weight"] = 1
    changed_label = score_label(facts, changed)

    kept = dict(standard)
    if original in {"0", "1"}:
        kept["unauthorized_weight"] = 3
    elif original == "2":
        kept["amount_threshold_usd"] = 501
    else:
        kept[_present_one_point_factor(facts)] = 2
    kept_label = score_label(facts, kept)
    return (_score_rule(changed), changed_label), (_score_rule(kept), kept_label)


def _template_spec(template: str) -> tuple[Callable[[str, int, bool], dict], str, Callable]:
    if template == "id_transfer_release_constraints":
        return _policy_choice_row, "\n\nEvidence:\n", _transfer_edits
    if template == "id_card_purchase_all_constraints":
        return _policy_noul_row, "\n\nRequest evidence:\n", _card_edits
    if template == "id_incident_additive_severity":
        return _policy_score_row, "\n\nObserved evidence:\n", _score_edits
    raise ValueError(f"unknown template: {template}")

def _semantic_original(row: dict) -> str:
    return (row["audit"]["original_target_key"] if row["question"]["type"] == "choice"
            else row["target"]["label"])




def build_probe_rows(existing_group_ids: set[str], *, bases_per_template: int = BASES_PER_TEMPLATE,
                     start_index: int = START_INDEX) -> tuple[list[dict], list[dict], dict]:
    if bases_per_template <= 0 or bases_per_template % 4:
        raise ValueError("bases_per_template must be a positive multiple of four")
    requests: list[dict] = []
    keys: list[dict] = []
    selected_group_ids: set[str] = set()
    proof = Counter()
    answer_counts = {}
    index_ranges = {}


    for template in TEMPLATES:
        builder, marker, edit = _template_spec(template)
        label_count = 2 if template == "id_card_purchase_all_constraints" else 4
        per_label = bases_per_template // label_count
        label_counts: Counter[str] = Counter()
        bases = []
        index = start_index
        while len(bases) < bases_per_template:
            row = builder("test_id", index, False)
            if row["provenance"]["family"] != template:
                raise AssertionError(f"generator returned the wrong family for {template}")
            semantic = _semantic_original(row)
            if (label_counts[semantic] < per_label
                    and row["group_id"] not in existing_group_ids
                    and row["group_id"] not in selected_group_ids):
                bases.append(row)
                label_counts[semantic] += 1
                selected_group_ids.add(row["group_id"])
            index += 1
            if index - start_index > bases_per_template * 1000:
                raise RuntimeError(f"could not find a fresh balanced range for {template}")
        if set(label_counts.values()) != {per_label}:
            raise AssertionError(f"probe answers are not balanced for {template}: {label_counts}")
        answer_counts[template] = dict(sorted(label_counts.items()))
        index_ranges[template] = {"start_inclusive": start_index, "end_exclusive": index}

        for row in bases:
            original_label = row["target"]["label"]
            changed, kept = edit(row)
            if row["question"]["type"] == "choice":
                changed = (changed[0], _choice_key(row, changed[1]))
                kept = (kept[0], _choice_key(row, kept[1]))
            if changed[1] == original_label:
                raise AssertionError(f"changed edit did not change {template} label")
            if kept[1] != original_label:
                raise AssertionError(f"kept edit changed {template} label")

            suffix = _evidence_suffix(row["state"], marker)
            prefix = row["state"].split("\n", 1)[0] + "\n"
            variants = {
                "original": (row["state"], original_label),
                "edit_changes_label": (prefix + changed[0] + suffix, changed[1]),
                "edit_keeps_label": (prefix + kept[0] + suffix, kept[1]),
            }
            base_id = f"{template}:{row['provenance']['variant']}"
            observed_suffixes = []
            for variant in VARIANTS:
                state, label = variants[variant]
                observed_suffixes.append(_evidence_suffix(state, marker).encode("utf-8"))
                identifier = stable_id("rule-probe", base_id, variant)
                request = _request(row, identifier=identifier, state=state)
                requests.append(request)
                keys.append({
                    "id": identifier,
                    "dataset": "rule_probe",
                    "variant": variant,
                    "template": template,
                    "base_id": base_id,
                    "source_group_id": row["group_id"],
                    "label": label,
                    "original_label": original_label,
                    "original_semantic_label": _semantic_original(row),
                    "candidate_keys": [item["key"] for item in request["candidates"]],
                })
                proof[(template, variant)] += 1
            if len(set(observed_suffixes)) != 1:
                raise AssertionError(f"evidence bytes changed for {base_id}")

    if len({request["id"] for request in requests}) != len(requests):
        raise AssertionError("probe request IDs are not unique")
    changes = sum(key["variant"] == "edit_changes_label" and key["label"] != key["original_label"] for key in keys)
    keeps = sum(key["variant"] == "edit_keeps_label" and key["label"] == key["original_label"] for key in keys)
    bases = bases_per_template * len(TEMPLATES)
    return requests, keys, {
        "bases": bases,
        "questions": len(requests),
        "group_id_collision_count": 0,
        "original_answer_counts": answer_counts,
        "index_ranges": index_ranges,
        "evidence_byte_identical_bases": bases,
        "edit_changes_label_verified": changes,
        "edit_keeps_label_verified": keeps,
        "counts": {f"{template} x {variant}": proof[(template, variant)]
                   for template in TEMPLATES for variant in VARIANTS},
    }


def prepared_group_ids(root: Path) -> set[str]:
    result: set[str] = set()
    for path in sorted((root / "data" / "prepared").glob("*.jsonl")):
        for row in _jsonl(path):
            group_id = row["group_id"]
            if not isinstance(group_id, str):
                raise ValueError(f"invalid group_id in {path}")
            result.add(group_id)
    return result


def _external_rows(paths: list[Path], dataset: str) -> tuple[list[dict], list[dict]]:
    requests = []
    keys = []
    for path in paths:
        for row in _jsonl(path):
            request = _request(row)
            label = row["target"]["label"]
            if dataset == "hard_v1":
                group = row["provenance"]["family"]
                key = {"id": row["id"], "dataset": dataset, "family": group,
                       "label": label, "candidate_keys": [item["key"] for item in request["candidates"]]}
            elif dataset == "doclaynet":
                class_name = row["question"]["criteria"][label]
                key = {"id": row["id"], "dataset": dataset, "class": class_name,
                       "label": label, "candidate_keys": [item["key"] for item in request["candidates"]]}
            else:
                raise ValueError(f"unknown external dataset: {dataset}")
            requests.append(request)
            keys.append(key)
    if not requests or len({row["id"] for row in requests}) != len(requests):
        raise ValueError(f"{dataset} requests must be nonempty with unique IDs")
    return requests, keys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_request_files(root: Path, files: list[tuple[Path, Path]], *, artifact: Path,
                          code_archive: Path) -> dict:
    """Compile through the artifact-bound v1 NativeCompiler implementation."""
    helper = r'''
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from mdprep.native import MAX_POSITIONS, NativeCompiler
from mdprep.records import parse_request
artifact = Path(sys.argv[2])
compiler = NativeCompiler.from_artifact(artifact, media_root=Path(sys.argv[4]))
results = {}
args = sys.argv[3:]
for index in range(0, len(args), 2):
    request_path = Path(args[index])
    compiler.media_root = Path(args[index + 1])
    count = 0
    maximum = 0
    with request_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = parse_request(json.loads(line))
            batch = compiler.compile((item,))
            positions = int(batch.expanded_lengths[0].item())
            if positions > MAX_POSITIONS:
                raise AssertionError(f"{item.id} compiled to {positions} positions")
            maximum = max(maximum, positions)
            count += 1
    results[str(request_path)] = {"questions": count, "max_positions": maximum,
                                  "position_limit": MAX_POSITIONS}
print(json.dumps({"processor_id": compiler.processor_id, "files": results}, sort_keys=True))
'''
    with tempfile.TemporaryDirectory(prefix="rule-probe-v1-code-") as temporary:
        with tarfile.open(code_archive, "r:gz") as archive:
            archive.extractall(temporary, filter="data")
        source = Path(temporary) / "src"
        command = [str(root / ".venv" / "bin" / "python"), "-c", helper,
                   str(source), str(artifact)]
        for request_path, media_root in files:
            command.extend((str(request_path), str(media_root)))
        result = subprocess.run(command, cwd=root, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"native compilation failed:\n{result.stdout}\n{result.stderr}")
        compiled = json.loads(result.stdout.strip().splitlines()[-1])
    compiled["files"] = {
        str(Path(path).relative_to(root)): value for path, value in compiled["files"].items()
    }
    compiled["compiler_code_archive"] = str(code_archive.relative_to(root))
    compiled["compiler_code_archive_sha256"] = _sha256(code_archive)
    compiled["artifact"] = str(artifact.relative_to(root))
    return compiled

def compile_current_tree_request_files(root: Path, files: list[tuple[Path, Path]]) -> dict:
    compiler = NativeCompiler.from_pretrained(root, media_root=files[0][1])
    results = {}
    for request_path, media_root in files:
        compiler.media_root = media_root
        maximum = 0
        rows = _jsonl(request_path)
        for row in rows:
            batch = compiler.compile((parse_request(row),))
            positions = int(batch.expanded_lengths[0].item())
            if positions > MAX_POSITIONS:
                raise AssertionError(f"{row['id']} compiled to {positions} positions")
            maximum = max(maximum, positions)
        results[str(request_path.relative_to(root))] = {
            "questions": len(rows),
            "max_positions": maximum,
            "position_limit": MAX_POSITIONS,
        }
    return {"processor_id": compiler.processor_id, "method": "NativeCompiler.from_pretrained",
            "files": results}



def build(root: Path, output: Path, *, compile_requests: bool = True) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    probe_requests, probe_keys, proof = build_probe_rows(prepared_group_ids(root))
    hard_requests, hard_keys = _external_rows(
        [root / ".audit" / "hardv1-converted" / "test_ood.jsonl"], "hard_v1")
    doc_paths = sorted((root / ".audit" / "doclaynet-candidate" / "records").glob("*.jsonl"))
    doc_requests, doc_keys = _external_rows(doc_paths, "doclaynet")

    outputs = {
        "rule_probe": (output / "rule-probe.requests.jsonl", output / "rule-probe.key.jsonl",
                       probe_requests, probe_keys),
        "hard_v1_test": (output / "hard-v1-test.requests.jsonl", output / "hard-v1-test.key.jsonl",
                         hard_requests, hard_keys),
        "doclaynet": (output / "doclaynet.requests.jsonl", output / "doclaynet.key.jsonl",
                      doc_requests, doc_keys),
    }
    for request_path, key_path, requests, keys in outputs.values():
        _write_jsonl(request_path, requests)
        _write_jsonl(key_path, keys)

    doc_counts = Counter(key["class"] for key in doc_keys)
    majority = max(doc_counts.values()) / len(doc_keys)
    if majority != 0.495:
        raise AssertionError(f"DocLayNet majority baseline changed: {majority:.6f}")
    receipt = {
        "status": "VERIFIED",
        "probe": proof,
        "external": {
            "hard_v1_test_questions": len(hard_requests),
            "doclaynet_questions": len(doc_requests),
            "doclaynet_media_root_local": ".audit/doclaynet-candidate",
            "doclaynet_media_root_pod": "/workspace/external/doclaynet-candidate",
            "doclaynet_majority_class_baseline": majority,
        },
        "files": {
            name: {"requests": str(values[0].relative_to(root)), "key": str(values[1].relative_to(root))}
            for name, values in outputs.items()
        },
    }
    if compile_requests:
        compile_files = [
            (outputs["rule_probe"][0], root / "data"),
            (outputs["hard_v1_test"][0], root / "data"),
            (outputs["doclaynet"][0], root / ".audit" / "doclaynet-candidate"),
        ]
        receipt["compilation"] = compile_request_files(
            root, compile_files,
            artifact=root / ".audit" / "finalized" / "finalized-model",
            code_archive=root / "dist-accepted-2026-10-16" / "training-code.tar.gz",
        )
        receipt["current_tree_compilation"] = compile_current_tree_request_files(root, compile_files)
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-compile", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = (arguments.output or root / ".audit" / "rule-probe").resolve()
    print(json.dumps(build(root, output, compile_requests=not arguments.skip_compile),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
