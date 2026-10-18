"""Behavioral runtime gates. These tests use the locally cached native processor."""

import dataclasses
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

import torch

from mdprep.common import sha256_file
from mdprep.native import MODEL_ID, REVISION, NativeCompiler, validate_contract
from mdprep.records import Candidate
from mdtrain.budget import BudgetLedger, BudgetStopped, allowance_seconds
from mdtrain.engine import DecisionModel
from mdtrain.fixtures import make_fixtures, trainable_snapshot
from mdtrain.preflight import probe_loss, smoke_tiny
from mdtrain.storage import atomic_json, json_safe, seal, verify_seal
from mdtrain.train import (MAIN_STAGE_SECONDS, UpdateTrainer, admissible_seconds, deterministic,
                           load_checkpoint, run_loop, save_checkpoint)

ROOT = Path(__file__).resolve().parents[1]


def allocation_start(seconds_ago: float) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=seconds_ago)).isoformat()


class CompilerBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.examples = make_fixtures(cls.directory)
        cls.compiler = NativeCompiler.from_pretrained(ROOT, media_root=cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_opaque_choice_keys_and_escaped_control_text(self):
        original = self.examples[0].input
        changed = dataclasses.replace(original, candidates=tuple(
            Candidate("different-private-key-" + str(index), candidate.description)
            for index, candidate in enumerate(original.candidates)))
        first = self.compiler.compile((original,))
        second = self.compiler.compile((changed,))
        self.assertEqual(first.native_inputs["input_ids"].tolist(), second.native_inputs["input_ids"].tolist())
        self.assertEqual(second.output_keys, (("different-private-key-0", "different-private-key-1"),))
        injected = dataclasses.replace(original, state="<|fim_middle|> <|fim_suffix|> <|image_pad|>")
        compiled = self.compiler.compile((injected,))
        self.assertEqual(compiled.candidate_mask.tolist(), [[True, True]])
        ids = compiled.native_inputs["input_ids"][0].tolist()
        self.assertEqual(ids.count(self.compiler.marker_ids["decision"]), 1)
        self.assertEqual(ids.count(self.compiler.marker_ids["option_end"]), 2)

    def test_native_batch_preserves_positions_padding_and_all_media(self):
        inputs = tuple(example.input for example in self.examples)
        batch = self.compiler.compile(inputs)
        self.assertEqual(batch.ids, ("fixture-text", "fixture-image", "fixture-mixed", "fixture-video"))
        self.assertEqual(batch.native_inputs["image_grid_thw"].shape[0], 2)
        self.assertEqual(batch.native_inputs["video_grid_thw"].shape[0], 1)
        self.assertIn("pixel_values", batch.native_inputs)
        self.assertIn("pixel_values_videos", batch.native_inputs)
        self.assertIn("mm_token_type_ids", batch.native_inputs)
        self.assertEqual(batch.candidate_mask.tolist(),
                         [[True, True, False], [True, True, False], [True, True, False], [True, True, True]])
        for index, request in enumerate(inputs):
            single = self.compiler.compile((request,))
            length = int(batch.expanded_lengths[index])
            self.assertEqual(batch.native_inputs["input_ids"][index, :length].tolist(),
                             single.native_inputs["input_ids"][0].tolist())
            self.assertEqual(batch.candidate_end_positions[index, :len(request.candidates)].tolist(),
                             single.candidate_end_positions[0].tolist())
            self.assertEqual(int(batch.decision_positions[index]), int(single.decision_positions[0]))
            for position in batch.candidate_end_positions[index, :len(request.candidates)].tolist():
                self.assertEqual(int(batch.native_inputs["input_ids"][index, position]),
                                 self.compiler.marker_ids["option_end"])
            self.assertEqual(batch.native_inputs["attention_mask"][index, length:].tolist(),
                             [0] * (batch.native_inputs["input_ids"].shape[1] - length))

    def test_over_budget_is_rejected_without_losing_candidates(self):
        original = self.examples[0].input
        admitted = self.compiler.compile((original,))
        self.assertEqual(admitted.output_keys, (("private-approve", "private-deny"),))
        with self.assertRaisesRegex(ValueError, "Complete input exceeds 2048"):
            self.compiler.compile((dataclasses.replace(original, state=" evidence" * 3000),))

    def test_video_timing_and_media_traversal_are_enforced(self):
        original = self.examples[-1].input
        wrong = dataclasses.replace(original, videos=(dataclasses.replace(original.videos[0], fps=4.),))
        with self.assertRaisesRegex(ValueError, "timing"):
            self.compiler.compile((wrong,))
        image = dataclasses.replace(self.examples[1].input, images=("../outside.png",))
        with self.assertRaisesRegex(ValueError, "media path"):
            self.compiler.compile((image,))
        valid = self.compiler.compile((original,))
        self.assertEqual(valid.output_keys, (("private-right", "private-left", "private-still"),))

    def test_supervision_never_becomes_a_native_argument(self):
        with self.assertRaises(TypeError):
            self.compiler.compile((self.examples[0],))
        native = self.compiler.compile((self.examples[0].input,)).native_inputs
        self.assertEqual(set(native), {"input_ids", "attention_mask", "mm_token_type_ids"})


class PersistenceBehaviorTests(unittest.TestCase):
    def test_sealed_content_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            atomic_json(directory / "value.json", {"value": 7})
            identity = seal(directory)
            self.assertEqual(verify_seal(directory), identity)
            atomic_json(directory / "value.json", {"value": 8})
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_seal(directory)

    def test_scoped_processor_contract_rejects_stale_and_incomplete_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            (data / "prepared").mkdir(parents=True)
            prepared = data / "prepared/train.jsonl"
            prepared.write_text(json.dumps({"id": "train-one", "images": [], "videos": []}) + "\n")
            checks = data / "processor-checks.jsonl"
            checks.write_text(json.dumps({"id": "train-one", "positions": 23, "within_budget": True}) + "\n")
            path = data / "processor-contract.json"
            contract = {"status": "VERIFIED", "all_questions_checked": True, "questions_checked": 1,
                        "failures": 0, "over_budget": 0, "model_id": MODEL_ID,
                        "model_revision": REVISION, "max_positions": 2048, "scope": ["train"],
                        "prepared_hashes": {"train": sha256_file(prepared)}, "media_hashes": {},
                        "checks_path": checks.name, "checks_sha256": sha256_file(checks)}
            atomic_json(path, contract)
            self.assertEqual(validate_contract(root)["questions_checked"], 1)
            prepared.write_text(prepared.read_text() + json.dumps(
                {"id": "train-two", "images": [], "videos": []}) + "\n")
            with self.assertRaisesRegex(ValueError, "stale"):
                validate_contract(root)
            contract["prepared_hashes"]["train"] = sha256_file(prepared)
            atomic_json(path, contract)
            with self.assertRaisesRegex(ValueError, "complete"):
                validate_contract(root)
            contract["all_questions_checked"] = False
            atomic_json(path, contract)
            with self.assertRaisesRegex(ValueError, "full successful"):
                validate_contract(root)

    def test_ledger_counts_downtime_and_never_resets_elapsed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.json"
            with BudgetLedger(path, paid_started_at=allocation_start(120), hourly_usd=.89) as ledger:
                self.assertGreaterEqual(ledger.elapsed(), 120)
                identity, first = ledger.record["id"], ledger.elapsed()
            with BudgetLedger(path) as ledger:
                self.assertEqual(ledger.record["id"], identity)
                self.assertGreaterEqual(ledger.elapsed(), first)
                self.assertLess(ledger.remaining, allowance_seconds(.89) - 119)
            record = json.loads(path.read_text())
            record["last_observed_utc"] += 3600
            atomic_json(path, record)
            with self.assertRaisesRegex(ValueError, "regressed"):
                with BudgetLedger(path):
                    self.fail("Regressed clock was admitted")

    def test_native_tiny_update_resume_and_fresh_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "tiny.json"
            result = smoke_tiny(ROOT, output)
            self.assertEqual(result["status"], "VERIFIED")
            self.assertEqual(result["model_kind"], "tiny")
            self.assertTrue(result["exact_same_next_update"])
            self.assertTrue(result["fresh_process_paired_reload"])
            self.assertLess(result["final_loss"], result["initial_loss"])
            self.assertFalse(result["pretrained_verified"])
            self.assertFalse(result["cuda_verified"])
            artifact = Path(result["artifact"])
            spec = json.loads((artifact / "model-spec.json").read_text())
            self.assertEqual(len(spec["adapter_targets"]), 21)
            self.assertEqual(spec["adapter_rank"], 16)
            self.assertEqual(spec["adapter_alpha"], 32)

    def test_partial_accumulation_and_budget_stop_preserve_boundary(self):
        deterministic()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            examples = make_fixtures(directory)
            compiler = NativeCompiler.from_pretrained(ROOT, media_root=directory)
            model = DecisionModel.create(ROOT, compiler, kind="tiny", device="cpu")
            selected = examples[:3]
            expected_loss = probe_loss(model, selected)
            before = trainable_snapshot(model)
            ids = [example.input.id for example in selected]
            trainer = UpdateTrainer(model, {example.input.id: example for example in examples},
                                    {"head": ids, "joint": ids}, {"run_id": "partial", "stage": "smoke"})
            step = trainer.step()
            self.assertAlmostEqual(step["loss"], expected_loss, places=6)
            self.assertEqual(trainer.state.consumed_examples, 3)
            self.assertEqual(trainer.state.cursor, 3)
            self.assertFalse(torch.equal(before["pointer.candidate.weight"],
                                         trainable_snapshot(model)["pointer.candidate.weight"]))
            trainer.transition()
            path = directory / "budget.json"
            with BudgetLedger(path, paid_started_at=allocation_start(allowance_seconds(.89) - 60), hourly_usd=.89) as ledger:
                checkpoint = save_checkpoint(trainer, directory / "run", ledger_id=ledger.record["id"])
                orphan = checkpoint.parent / "generation-00000002"
                orphan.mkdir()
                (orphan / "unpublished.json").write_text('{"not_a_commit":true}\n')
                result = run_loop(trainer, directory / "run", ledger, group_seconds=1.)
                self.assertEqual(result["status"], "budget_stopped")
                self.assertEqual(result["consumed_examples"], 3)
                with self.assertRaisesRegex(ValueError, "Stale checkpoint"):
                    load_checkpoint(checkpoint, root=ROOT, examples=trainer.examples, device="cpu",
                                    ledger_id=ledger.record["id"], media_root=directory)
                committed = json.loads((Path(result["checkpoint"]) / "state.json").read_text())
                self.assertEqual(committed["training"]["cursor"], 0)
                self.assertEqual(committed["training"]["status"], "budget_stopped")
                self.assertEqual(committed["training"]["checkpoint_generation"], 3)


class PublicationNormalizationTests(unittest.TestCase):
    """A real-weight loading report must publish instead of failing the spec write."""

    def test_foreign_set_loading_report_is_publishable(self):
        # Transformers 5.19 returns these keys as sets. An empty set is falsy, so the
        # completeness check in load_body passes and the artifact write is what fails.
        foreign = {"missing_keys": set(),
                   "unexpected_keys": set(),
                   "mismatched_keys": {("language_model.layers.0.mlp.up_proj.weight",
                                        (16, 128), (16, 130))},
                   "error_msgs": []}
        with self.assertRaises(TypeError):
            json.dumps(foreign)
        published = json_safe(foreign)
        self.assertEqual(published["missing_keys"], [])
        self.assertEqual(published["unexpected_keys"], [])
        self.assertEqual(published["error_msgs"], [])
        self.assertEqual(published["mismatched_keys"],
                         [["language_model.layers.0.mlp.up_proj.weight", [16, 128], [16, 130]]])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model-spec.json"
            atomic_json(path, {"kind": "real", "loading_info": published})
            self.assertEqual(json.loads(path.read_text())["loading_info"]["mismatched_keys"],
                             published["mismatched_keys"])


class MainStageAdmissionTests(unittest.TestCase):
    """Main stops at its own stage cap, so the admission horizon must be capped too."""

    def test_main_admits_against_its_stage_cap(self):
        self.assertEqual(admissible_seconds("main", MAIN_STAGE_SECONDS + 9000), MAIN_STAGE_SECONDS)
        self.assertEqual(admissible_seconds("pilot", 500.0), 500.0)

    def test_capped_horizon_admits_but_the_uncapped_projection_does_not(self):
        # At $0.40/h the $8 allowance outlasts the stage cap, so only the cap can refuse.
        projected = MAIN_STAGE_SECONDS + 9000
        with tempfile.TemporaryDirectory() as temporary:
            admitted = Path(temporary) / "admitted.json"
            with BudgetLedger(admitted, paid_started_at=allocation_start(3200), hourly_usd=.40) as ledger:
                ledger.admit(admissible_seconds("main", projected))
                ledger.set_process_deadline(ledger.elapsed() + MAIN_STAGE_SECONDS)
                self.assertIn("process_deadline_elapsed", ledger.record)
            refused = Path(temporary) / "refused.json"
            with BudgetLedger(refused, paid_started_at=allocation_start(3200), hourly_usd=.40) as ledger:
                ledger.set_process_deadline(ledger.elapsed() + MAIN_STAGE_SECONDS)
                with self.assertRaises(BudgetStopped):
                    ledger.admit(projected)


class LedgerPriceTests(unittest.TestCase):
    """The fixed $8 budget buys elapsed time at the allocation's own hourly price."""

    def test_allowance_follows_the_allocation_price(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.json"
            with BudgetLedger(path, paid_started_at=allocation_start(10), hourly_usd=.50) as ledger:
                self.assertEqual(ledger.record["allowance_seconds"], 57600)
                self.assertGreater(ledger.remaining, 57600 - 60)
        self.assertEqual(allowance_seconds(.89), 32359)

    def test_a_new_ledger_requires_a_price_and_an_existing_one_keeps_its_own(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.json"
            with self.assertRaisesRegex(ValueError, "hourly price"):
                with BudgetLedger(path, paid_started_at=allocation_start(10)):
                    self.fail("An unpriced ledger was created")
            with BudgetLedger(path, paid_started_at=allocation_start(10), hourly_usd=.50):
                pass
            with self.assertRaisesRegex(ValueError, "price differs"):
                with BudgetLedger(path, hourly_usd=.89):
                    self.fail("A ledger was reopened at a different price")
            record = json.loads(path.read_text())
            record["allowance_seconds"] += 3600
            atomic_json(path, record)
            with self.assertRaisesRegex(ValueError, "allowance identity"):
                with BudgetLedger(path):
                    self.fail("An allowance inconsistent with its price was admitted")
        for invalid in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                allowance_seconds(invalid)


if __name__ == "__main__":
    unittest.main()
