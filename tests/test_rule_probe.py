"""Counterfactual probes preserve evidence while changing only rule parameters."""

from collections import Counter, defaultdict

from tools.rule_probe import TEMPLATES, VARIANTS, build_probe_rows


MARKERS = {
    "id_transfer_release_constraints": "\n\nEvidence:\n",
    "id_card_purchase_all_constraints": "\n\nRequest evidence:\n",
    "id_incident_additive_severity": "\n\nObserved evidence:\n",
}


def test_probe_edits_recompute_labels_and_preserve_evidence_bytes():
    requests, keys, proof = build_probe_rows(set(), bases_per_template=8)
    request_by_id = {row["id"]: row for row in requests}
    grouped = defaultdict(list)
    for key in keys:
        grouped[key["base_id"]].append(key)

    assert len(grouped) == 8 * len(TEMPLATES)
    assert proof["evidence_byte_identical_bases"] == len(grouped)
    assert proof["edit_changes_label_verified"] == len(grouped)
    assert proof["edit_keeps_label_verified"] == len(grouped)
    for base_keys in grouped.values():
        assert {key["variant"] for key in base_keys} == set(VARIANTS)
        by_variant = {key["variant"]: key for key in base_keys}
        assert by_variant["edit_changes_label"]["label"] != by_variant["original"]["label"]
        assert by_variant["edit_keeps_label"]["label"] == by_variant["original"]["label"]
        marker = MARKERS[base_keys[0]["template"]]
        evidence = {
            request_by_id[key["id"]]["state"].partition(marker)[2].encode("utf-8")
            for key in base_keys
        }
        assert len(evidence) == 1


def test_probe_is_balanced_across_original_answers():
    _, keys, _ = build_probe_rows(set(), bases_per_template=8)
    originals = [row for row in keys if row["variant"] == "original"]
    counts = Counter((row["template"], row["original_semantic_label"]) for row in originals)
    for template in TEMPLATES:
        family_counts = [count for (family, _), count in counts.items() if family == template]
        assert max(family_counts) == min(family_counts)
