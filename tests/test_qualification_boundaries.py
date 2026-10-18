"""Behavioral guards for provenance, complete trajectories, and train-only subsets."""

import copy
import tempfile
import unittest
from pathlib import Path

from mdprep.common import choice, record, sha256_file, write_json, write_jsonl
from mdprep.records import validate_record
from mdprep.verify import object_hash, reconcile_temporal_frames, validate_verification_contract


def specimen(split="train", index=0, source="mdprep_text_rules"):
    return record(split=split, source=source, group_id=f"{split}-{index}",
                  family="id_transfer_release_constraints", modality="text",
                  state=f"Evidence {split} {index}", question=choice("Choose", {"allow": "Allow", "decline": "Decline"}),
                  label="decline", provenance={"license": "Project-created"}, variant=index)


def publish_receipt(root, train, dev, subset):
    primary={"train": train, "dev": dev}
    for split, rows in primary.items():
        write_jsonl(root / "data/prepared" / f"{split}.jsonl", rows)
    derived=root / "data/prepared/pilot.jsonl"
    write_jsonl(derived, subset)
    source_receipt={"status": "VERIFIED", "questions": len(train)+len(dev)}
    contract={"status": "VERIFIED", "all_questions_checked": True,
              "questions_checked": len(train)+len(dev), "scope": list(primary),
              "prepared_hashes": {split: sha256_file(root / "data/prepared" / f"{split}.jsonl") for split in primary},
              "media_hashes": {}, "row_receipts": {row["id"]: object_hash(row) for rows in primary.values() for row in rows},
              "derived_hashes": {"prepared/pilot.jsonl": sha256_file(derived)},
              "source_reconciliation": source_receipt, "source_receipt_sha256": object_hash(source_receipt),
              "human_review": {"status": "NOT_VERIFIED"}}
    contract["qualification_sha256"]=object_hash(contract)
    write_json(root / "data/verification-contract.json", contract)


class QualificationBoundaryTests(unittest.TestCase):
    def test_source_contract_rejects_fabricated_real_origin_and_domain(self):
        for field, value in (("origin", "real_human_label"), ("source_domain", "unrelated-source")):
            with self.subTest(field=field):
                row=specimen()
                row["provenance"][field]=value
                with self.assertRaises(ValueError):
                    validate_record(row)
        row=specimen(split="test_ood")
        row["provenance"]["ood_axis"]="source"
        with self.assertRaises(ValueError):
            validate_record(row)
        with self.assertRaises(ValueError):
            validate_record(specimen(source="unsupported-source"))

    def test_derived_sets_are_unique_complete_train_subsets(self):
        train=[specimen(index=i) for i in range(2)]
        dev=[specimen(split="dev", index=i) for i in range(2)]
        cases=(("valid", train, True), ("evaluation-row", [train[0], dev[0]], False),
               ("duplicate", [train[0], train[0]], False), ("truncated", train[:1], False))
        for name, subset, accepted in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                publish_receipt(root, train, dev, subset)
                if accepted:
                    self.assertEqual(validate_verification_contract(root)["questions_checked"], 4)
                else:
                    with self.assertRaises(ValueError):
                        validate_verification_contract(root)


class CompleteTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.facts=[{"focus_color": color} for color in ("red", "blue", "green", "yellow")]
        self.row={"videos": [{"fps": 2.0}], "audit": {
            "sample_times_s": [0.0, 0.5, 1.0, 1.5],
            "frame_facts": [{"sample_index": index, "time_s": index/2, **fact}
                            for index, fact in enumerate(self.facts)]}}

    def test_canonical_identity_uses_all_decoded_observations(self):
        self.assertEqual(reconcile_temporal_frames(self.row, self.facts),
                         [(("focus_color", "red"),), (("focus_color", "blue"),),
                          (("focus_color", "green"),), (("focus_color", "yellow"),)])

    def test_missing_extra_reordered_or_retimed_audit_cannot_shorten_identity(self):
        mutations=[]
        row=copy.deepcopy(self.row); row["audit"]["frame_facts"].pop(); mutations.append(row)
        row=copy.deepcopy(self.row); row["audit"]["frame_facts"].append(copy.deepcopy(row["audit"]["frame_facts"][-1])); mutations.append(row)
        row=copy.deepcopy(self.row); row["audit"]["frame_facts"].reverse(); mutations.append(row)
        row=copy.deepcopy(self.row); row["audit"]["frame_facts"][2]["time_s"]=1.25; mutations.append(row)
        row=copy.deepcopy(self.row); row["audit"]["frame_facts"][1]["sample_index"]=True; mutations.append(row)
        row=copy.deepcopy(self.row); row["audit"]["sample_times_s"][0]=0.25; mutations.append(row)
        for index, row in enumerate(mutations):
            with self.subTest(case=index), self.assertRaises(ValueError):
                reconcile_temporal_frames(row, self.facts)


if __name__ == "__main__":
    unittest.main()
