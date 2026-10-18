"""Behavioral tests for hard-v1 staging and null-option recovery."""

import collections
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mdprep.hardv1 import (
    LicenseInfo,
    SourceFile,
    _question_record,
    convert_files,
    decision_input,
    render_state,
    split_development_rows,
    resolve_option_text,
    slug,
    validate_hardv1_record,
)
from mdprep.records import validate_record


LICENSE = LicenseInfo(
    "Apache-2.0",
    "https://example.test/LICENSE",
    "Copyright 2026 Jared Palmer",
    "Jared Palmer, Kev, 2026",
    {"test": True},
)


class _Length:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeCompiler:
    def compile(self, items):
        item = items[0]
        if "OVER BUDGET" in item.state:
            raise ValueError(f"Complete input exceeds 2048 positions: {item.id} (2107)")
        return SimpleNamespace(expanded_lengths=[_Length(137)])


def source_file(path, split="train"):
    return SourceFile(
        path,
        split,
        f"hard-v1/{split}.jsonl",
        f"https://example.test/{split}.jsonl",
        "https://example.test/repository",
        "revision",
        "https://example.test/revision",
    )


def source_record(split="train", state="Evidence"):
    return {
        "state": state,
        "questions": {
            "choice": {
                "type": "choice",
                "instructions": "Which service should be chosen?",
                "criteria": {
                    "tidedb": None,
                    "none_qualifies": "Reject all options: none meets the requirements",
                    "granite_db": None,
                    "lumen_data": None,
                },
                "src": "hard_tradeoff",
                "label": "lumen_data",
            },
            "meets": {
                "type": "noul",
                "instructions": "Does Granite DB pass the hard requirements?",
                "src": "hard_tradeoff",
                "label": False,
            },
            "rating": {
                "type": "score",
                "instructions": "How strong is the result?",
                "criteria": ["Zero", "One", "Two", "Three", "Four", "Five"],
                "src": "hard_tradeoff",
                "label": 4,
            },
        },
        "_meta": {
            "id": f"hard-v1/tradeoff/{split}/00736",
            "source": "hard_tradeoff",
            "group_id": f"hard-v1/tradeoff/{split}/00736",
            "variant": "clean",
            "split": split,
            "template": "tradeoff/t0" if split == "train" else "tradeoff/t4",
            "family": "tradeoff",
            "facts": {
                "options": {
                    "Lumen Data": {"cost": 750},
                    "Granite DB": {"cost": 3800},
                    "TideDB": {"cost": 1800},
                }
            },
            "text_sha256": "state-hash",
        },
    }


class HardV1ConversionTests(unittest.TestCase):
    def test_structured_state_renderer_is_deterministic_and_preserves_names(self):
        state = {
            "decision": "Choose a service.",
            "hard_requirements": ["cost must be at most $1,250", "backups must be included"],
            "selection_rule": "Choose the lowest latency.",
            "options": {
                "TideDB": {"cost": "$1,800", "backups": False},
                "Granite DB": {"cost": "$900", "backups": True},
            },
        }
        expected = (
            "Choose a service.\n\nHard requirements:\n"
            "- cost must be at most $1,250\n- backups must be included\n\n"
            "Choose the lowest latency.\n\n"
            "| option | cost | backups |\n|---|---|---|\n"
            "| TideDB | $1,800 | no |\n| Granite DB | $900 | yes |"
        )
        self.assertEqual(render_state(state), expected)
        self.assertEqual(render_state(state), render_state(json.loads(json.dumps(state))))
        self.assertEqual(
            render_state({"topic": "Invoice timing", "facts": ["Issued Monday.", "Paid Friday."]}),
            "topic: Invoice timing\nfacts:\n  - Issued Monday.\n  - Paid Friday.",
        )

    def test_null_resolution_uses_mapping_scan_then_title_case(self):
        mappings = ({"tidedb": "TideDB"}, {})
        self.assertEqual(resolve_option_text("tidedb", "not present", mappings),
                         ("TideDB", "exact_mapping"))
        self.assertEqual(resolve_option_text("granite_db", "Use Granite DB today.", ({}, {})),
                         ("Granite DB", "rendered_state_scan"))
        self.assertEqual(resolve_option_text("lumen_data", "not present", ({}, {})),
                         ("Lumen Data", "title_case_fallback"))
        self.assertEqual(resolve_option_text("___", "not present", ({}, {})),
                         (None, "unresolved"))

    def test_crux_preserves_order_casing_and_target_position(self):
        source = source_record(
            state=(
                "We need to choose a managed database service.\n"
                "| option | cost |\n|---|---|\n| Lumen Data | $750 |\n"
                "| Granite DB | $3,800 |\n| TideDB | $1,800 |"
            )
        )
        tiers = collections.Counter()
        row = _question_record(source, source_file(Path("unused")), 2, "choice",
                               source["questions"]["choice"], LICENSE, tiers)
        validate_hardv1_record(row)
        self.assertEqual(
            row["question"]["criteria"],
            {
                "option_0": "TideDB",
                "option_1": "Reject all options: none meets the requirements",
                "option_2": "Granite DB",
                "option_3": "Lumen Data",
            },
        )
        self.assertEqual(row["target"], {"label": "option_3"})
        parsed = decision_input(row)
        self.assertEqual(parsed.candidates[3].key, "lumen_data")
        self.assertEqual(tiers, {"exact_mapping": 3})

    def test_noul_and_score_shapes_round_trip(self):
        source = source_record()
        tiers = collections.Counter()
        noul = _question_record(source, source_file(Path("unused")), 1, "meets",
                                source["questions"]["meets"], LICENSE, tiers)
        score = _question_record(source, source_file(Path("unused")), 1, "rating",
                                 source["questions"]["rating"], LICENSE, tiers)
        validate_hardv1_record(noul)
        validate_hardv1_record(score)
        self.assertEqual(noul["question"]["criteria"], {"false": "No", "true": "Yes"})
        self.assertEqual(noul["target"]["label"], "false")
        self.assertEqual(list(score["question"]["criteria"]), ["0", "1", "2", "3", "4", "5"])
        self.assertEqual(score["target"]["label"], "4")
        self.assertEqual([candidate.key for candidate in decision_input(noul).candidates],
                         ["false", "true"])
        self.assertEqual([candidate.key for candidate in decision_input(score).candidates],
                         ["0", "1", "2", "3", "4", "5"])
        test_source = source_record(split="test")
        test_noul = _question_record(
            test_source,
            source_file(Path("unused"), "test"),
            7,
            "meets",
            test_source["questions"]["meets"],
            LICENSE,
            tiers,
        )
        validate_hardv1_record(test_noul)
        self.assertEqual(
            {
                key: test_noul["provenance"][key]
                for key in ("source", "source_domain", "upstream_id", "template",
                            "state_sha256", "ood_axis")
            },
            {
                "source": "hard-v1",
                "source_domain": "hard-v1",
                "upstream_id": "hard-v1/tradeoff/test/00736",
                "template": "tradeoff/t4",
                "state_sha256": "state-hash",
                "ood_axis": "template",
            },
        )

    def test_admitted_source_validates_and_unknown_source_is_rejected(self):
        source = source_record()
        tiers = collections.Counter()
        row = _question_record(source, source_file(Path("unused")), 1, "choice",
                               source["questions"]["choice"], LICENSE, tiers)
        validate_hardv1_record(row)
        validate_record(row)
        mutated = json.loads(json.dumps(row))
        mutated["provenance"]["source"] = "hard-v1-mutated"
        mutated["provenance"]["source_domain"] = "hard-v1-mutated"
        with self.assertRaisesRegex(ValueError, "^Unsupported canonical data source$"):
            validate_record(mutated)

    def test_development_groups_are_split_deterministically(self):
        rows = []
        for index in range(4):
            source = source_record(split="development")
            source["_meta"]["group_id"] = f"group-{index // 2}"
            source["_meta"]["id"] = f"row-{index}"
            question = source["questions"]["choice"]
            rows.append(_question_record(
                source, source_file(Path("unused"), "development"), index + 1,
                "choice", question, LICENSE, collections.Counter(),
            ))
        first = split_development_rows(rows)
        second = split_development_rows(json.loads(json.dumps(rows)))
        self.assertEqual(first, second)
        self.assertEqual(
            {row["split"] for row in first if row["group_id"] == "group-0"},
            {first[0]["split"]},
        )
        self.assertEqual(collections.Counter(row["split"] for row in first),
                         {"dev": 2, "calibration": 2})

    def test_checked_in_receipt_reports_family_coverage(self):
        root = Path(__file__).resolve().parents[1]
        receipt = json.loads((root / ".audit" / "hardv1-conversion.json").read_text())
        ratios = {
            family: counts["admitted_source_ratio"]
            for family, counts in receipt["per_family"].items()
        }
        self.assertEqual(ratios, {
            "ambiguous": "1058/1058",
            "judge": "1983/1983",
            "long_policy": "358/2114",
            "multi_hop": "1317/1317",
            "probability": "1662/1662",
            "temporal_numeric": "1233/1233",
            "tradeoff": "2114/2114",
        })
        partial = [
            family for family, counts in receipt["per_family"].items()
            if counts["coverage"] == "partial"
        ]
        self.assertEqual(partial, ["long_policy"])
        self.assertIn(
            "short tail of a family defined by long states",
            receipt["per_family"]["long_policy"]["coverage_note"],
        )

    def test_conversion_receipts_drops_and_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for split in ("train", "development", "test"):
                path = root / f"{split}.jsonl"
                record = source_record(split=split)
                if split == "test":
                    record["questions"] = {
                        "over": {
                            "type": "noul",
                            "instructions": "Is this too long?",
                            "src": "hard_tradeoff",
                            "label": True,
                        }
                    }
                    record["state"] = "OVER BUDGET"
                path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                sources.append(source_file(path, split))
            output = root / "outputs"
            receipt_path = root / "audit" / "hardv1-conversion.json"
            first = convert_files(sources, output, receipt_path, LICENSE, FakeCompiler())
            hashes = {path.name: path.read_bytes() for path in output.iterdir()}
            receipt_bytes = receipt_path.read_bytes()
            second = convert_files(sources, output, receipt_path, LICENSE, FakeCompiler())
            self.assertEqual(first, second)
            self.assertEqual(hashes, {path.name: path.read_bytes() for path in output.iterdir()})
            self.assertEqual(receipt_bytes, receipt_path.read_bytes())
            self.assertEqual(first["staging"], {
                "dataset_status": "qualified_conversion",
                "canonical_corpus": "admitted_by_mdprep_prepare",
                "admitted_to_canonical_corpus": True,
                "admission_scope": (
                    "train and development only; upstream test remains external"
                ),
                "record_schema_sha256": hashlib.sha256(
                    (Path(__file__).resolve().parents[1] / "src/mdprep/records.py").read_bytes()
                ).hexdigest(),
            })
            self.assertEqual(
                first["per_family"]["tradeoff"]["admitted_source_ratio"],
                "6/7",
            )
            self.assertEqual(first["null_option_descriptions"], {
                "source": 6, "resolved": 6, "unresolved": 0, "output_null_or_empty": 0,
            })
            self.assertEqual(first["compiled"]["max_positions"], 137)
            self.assertEqual(first["compiled"]["over_budget_by_family"], {
                "tradeoff": {"questions": 1, "max_positions": 2107}
            })
            self.assertEqual(first["dropped_questions"][0]["reason"]["code"],
                             "over_position_budget")
            for path in output.iterdir():
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    validate_hardv1_record(row)
                    descriptions = row["question"]["criteria"].values()
                    self.assertTrue(all(isinstance(value, str) and value for value in descriptions))
                    if row["question"]["type"] == "choice":
                        index = int(row["target"]["label"].removeprefix("option_"))
                        key = row["audit"]["original_candidate_keys"][index]
                        if source_record()["questions"]["choice"]["criteria"][key] is None:
                            self.assertEqual(slug(list(descriptions)[index]), key)


if __name__ == "__main__":
    unittest.main()
