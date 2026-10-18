"""Behavioral checks for counterfactual rule variants and held-out structures."""

import copy
import hashlib
import json
from pathlib import Path


from mdprep.rule_variants import (
    DOMAIN_FIELDS,
    FORBIDDEN_WORDING,
    HELD_OUT_COMBINATIONS,
    HELD_OUT_RENDERING,
    INCLUSIVE_AT_LEAST,
    INCLUSIVE_AT_MOST,
    NUMERIC_FIELD_SPECS,
    _render_numeric_condition,
    build_heldout_rows,
    build_structure_rows,
    build_template_rows,
    leakage_overlaps,
    wording_lint,
)
from mdprep.records import validate_record
from tools.score_predictions import _key_predictions, _synthetic_structure_key, score_rows

ROOT = Path(__file__).resolve().parents[1]


def _semantic_target(row):
    if row["question"]["type"] == "choice":
        return row["audit"]["original_target_key"]
    return row["target"]["label"]


def _independent_template_answer(row):
    facts = row["audit"]["facts"]
    parameters = row["audit"]["rule_parameters"]
    domain = row["audit"]["domain"]
    if domain == "transfer":
        if any(facts[name] is None for name in ("identity_verified", "recipient_age_days", "amount_usd")):
            return parameters["missing_label"]
        if (facts["identity_verified"] == parameters["declined_identity_status"]
                or facts["amount_usd"] > parameters["decline_above_usd"]):
            return "decline"
        if (facts["identity_verified"] is True
                and facts["recipient_age_days"] >= parameters["minimum_age_days"]
                and facts["amount_usd"] <= parameters["allow_ceiling_usd"]):
            return "allow"
        return "review"
    if domain == "card_purchase":
        checks = []
        if "card_active" in parameters["required_constraints"]:
            checks.append(facts["card_active"] == parameters["required_card_active"])
        if "country_supported" in parameters["required_constraints"]:
            checks.append(facts["country_supported"] == parameters["required_country_supported"])
        if "amount_limit" in parameters["required_constraints"]:
            checks.append(facts["amount_usd"] <= facts["daily_remaining_usd"] + parameters["limit_offset_usd"])
        return str(all(checks)).lower()
    points = (
        parameters["unauthorized_weight"] * int(facts["unauthorized"])
        + parameters["missing_weight"] * int(facts["card_missing"])
        + parameters["amount_weight"] * int(facts["amount_usd"] >= parameters["amount_threshold_usd"])
        + parameters["repeated_weight"] * int(facts["repeated"])
    )
    return str(min(parameters["cap"], points))


def _independent_condition(condition, facts):
    actual = facts[condition["field"]]
    if condition["operator"] in {"is", "equals"}:
        return actual == condition["value"]
    if condition["operator"] == "at_least":
        return actual >= condition["value"]
    if condition["operator"] == "at_most":
        return actual <= condition["value"]
    raise AssertionError(condition["operator"])


def _independent_structure_answer(row):
    facts = row["audit"]["facts"]
    parameters = row["audit"]["rule_parameters"]
    results = [_independent_condition(condition, facts) for condition in parameters["conditions"]]
    combinator = parameters["combinator"]
    if combinator == "all_of":
        return str(all(results)).lower()
    if combinator == "any_of":
        return str(any(results)).lower()
    if combinator == "unless_exception":
        return str(all(results[:-1]) and not results[-1]).lower()
    if combinator == "first_match":
        return next((action for result, action in zip(results, parameters["actions"]) if result),
                    parameters["default_action"])
    return str(min(parameters["cap"], sum(
        weight for result, weight in zip(results, parameters["weights"]) if result
    )))


def test_part_a_labels_are_independently_recomputed_and_pairs_change_mostly():
    rows = build_template_rows("dev", 150)
    for row in rows[::7]:
        validate_record(row)
        assert _independent_template_answer(row) == _semantic_target(row)
    pairs = {}
    for row in rows:
        pairs.setdefault(row["audit"]["pair_id"], []).append(row)
    changed = 0
    same = 0
    for members in pairs.values():
        assert len(members) == 2
        assert members[0]["audit"]["facts"] == members[1]["audit"]["facts"]
        labels = {_semantic_target(member) for member in members}
        changed += len(labels) == 2
        same += len(labels) == 1
    assert changed > same > 0


def test_part_b_labels_are_independently_recomputed_and_pairs_change_mostly():
    rows = build_structure_rows("dev", 200)
    for row in rows[::9]:
        validate_record(row)
        assert _independent_structure_answer(row) == _semantic_target(row)
        assert 2 <= row["audit"]["condition_count"] <= 5
    pairs = {}
    for row in rows:
        pairs.setdefault(row["audit"]["pair_id"], []).append(row)
    changed = sum(len({_semantic_target(row) for row in members}) == 2 for members in pairs.values())
    same = len(pairs) - changed
    assert changed > same > 0


def test_leakage_guard_reports_a_planted_five_gram_overlap():
    overlap = leakage_overlaps(
        ["A harmless prefix then alpha beta gamma delta epsilon and a suffix."],
        [("locked-row", "Another rule contains alpha beta gamma delta epsilon exactly.")],
    )
    assert overlap == [{
        "generated_index": 0,
        "reference_ids": ["locked-row"],
        "five_gram": "alpha beta gamma delta epsilon",
    }]


def test_heldout_structure_combinations_and_rendering_are_reserved():
    train = build_structure_rows("train", 400)
    heldout = build_heldout_rows(2)
    train_combinations = {(row["audit"]["domain"], row["audit"]["combinator"]) for row in train}
    heldout_combinations = {(row["audit"]["domain"], row["audit"]["combinator"]) for row in heldout}
    assert heldout_combinations == set(HELD_OUT_COMBINATIONS)
    assert not train_combinations & heldout_combinations
    assert {row["audit"]["rendering"] for row in heldout} == {HELD_OUT_RENDERING}
    assert HELD_OUT_RENDERING not in {row["audit"]["rendering"] for row in train}


def test_wording_lint_rejects_machine_phrasing_and_ambiguous_numeric_conditions():
    rows = [
        *build_template_rows("dev", 150),
        *build_structure_rows("train", 400),
        *build_heldout_rows(2),
    ]
    receipt = wording_lint(rows)
    assert receipt["status"] == "VERIFIED"
    assert set(receipt["forbidden_phrase_counts"]) == set(FORBIDDEN_WORDING)
    assert set(receipt["forbidden_phrase_counts"].values()) == {0}
    assert receipt["numeric_comparisons_without_inclusivity"] == 0
    assert receipt["numeric_boundary_coverage_pass"]
    boundary = receipt["numeric_boundary_facts"]
    assert all(boundary[key] > 0 for key in ("at_threshold", "one_below", "one_above", "farther"))
    assert 0.30 <= boundary["near_boundary_share"] <= 0.40
    assert boundary["farther_share"] >= 0.60

    numeric = next(
        row for row in rows
        if any(condition["operator"] in {"at_most", "at_least"}
               for condition in row["audit"]["rule_parameters"].get("conditions", []))
    )
    ambiguous = copy.deepcopy(numeric)
    ambiguous["audit"]["rule"] = "Use a numeric limit for this decision."
    ambiguous_receipt = wording_lint([ambiguous])
    assert ambiguous_receipt["numeric_comparisons_without_inclusivity"] > 0

    machine_like = copy.deepcopy(rows[0])
    machine_like["audit"]["rule"] += " If it matches => allow."
    assert wording_lint([machine_like])["forbidden_phrase_counts"]["=>"] == 1


def test_numeric_facts_and_thresholds_use_declared_realistic_ranges():
    declared_numeric_fields = {
        name
        for fields in DOMAIN_FIELDS.values()
        for name, kind, _ in fields
        if kind == "number"
    }
    assert declared_numeric_fields == set(NUMERIC_FIELD_SPECS)
    rows = [*build_structure_rows("train", 400), *build_heldout_rows(2)]
    observed_fields = set()
    for row in rows:
        facts = row["audit"]["facts"]
        for condition in row["audit"]["rule_parameters"]["conditions"]:
            field = condition["field"]
            if field not in NUMERIC_FIELD_SPECS:
                continue
            observed_fields.add(field)
            spec = NUMERIC_FIELD_SPECS[field]
            assert spec["minimum"] <= facts[field] <= spec["maximum"]
            assert spec["minimum"] <= condition["value"] <= spec["maximum"]
            assert spec["unit"]
            assert "_" not in row["audit"]["rule"]
    assert observed_fields == declared_numeric_fields

    malformed_verb_chains = (
        "active for does",
        "age of does",
        "includes does",
        "lasts does",
        "allows does",
        "weighs does",
        "totals does",
        "requests does",
        "reaches does",
        "active for is not",
        "age of is not",
        "includes is not",
        "lasts is not",
        "allows is not",
        "weighs is not",
        "totals is not",
        "requests is not",
        "reaches is not",
    )
    for field, spec in NUMERIC_FIELD_SPECS.items():
        threshold = (spec["minimum"] + spec["maximum"]) // 2
        rendered = []
        for operator, inclusive_phrases in (
            ("at_most", INCLUSIVE_AT_MOST),
            ("at_least", INCLUSIVE_AT_LEAST),
        ):
            for style in range(5):
                clause = _render_numeric_condition(field, operator, threshold, style)
                rendered.append(clause)
                assert any(phrase in clause for phrase in inclusive_phrases)
                assert not any(fragment in clause for fragment in malformed_verb_chains)
                assert "_" not in clause
        assert len(set(rendered)) == 10


def test_structure_scorer_reports_both_correct_per_pair():
    keys = _synthetic_structure_key()
    predictions = _key_predictions(keys, {"structure-1-edit_changes_label"})
    result = score_rows(predictions, keys)
    assert result["paired_original_and_edit_changes_label"] == {
        "pairs": 2,
        "both_correct": 0.5,
    }


def test_locked_splits_and_policy_match_the_admission_baseline():
    receipt_path = ROOT / ".audit" / "rule-variants-admission-baseline" / "hashes.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for relative in ("data/prepared/test_id.jsonl", "data/prepared/test_ood.jsonl",
                     "data/prepared/policy.jsonl"):
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == receipt["files"][relative]["sha256"]
    assert receipt["processor_id"] == "9cf4edc09256eb21aaddb7002d6cf753a3389065b924884d8eb6e24016d7f024"
