"""Isolated identity, learning, report, and process-boundary regression tests."""

import copy
import dataclasses
import datetime as dt
import json
import math
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mdprep.common import noul, record, sha256_file
from mdprep.native import MODEL_ID, REVISION, digest_json
from mdprep.records import Candidate, DecisionInput, TrainingExample
from mdprep.verify import object_hash
from mdtrain.budget import BudgetLedger
from mdtrain.data import DatasetView, qualified_inputs
from mdtrain.deadline import ProcessDeadline, ledger_deadline, supervise
from mdtrain.metrics import score_predictions
from mdtrain.preflight import assert_exact_replay
from mdtrain.storage import atomic_json, report_json, seal
from mdtrain.train import (TrainingState, backend_identity, checkpoint_disk_admission, code_identity,
                           fixed_probe, learning_evidence, load_checkpoint, record_run_receipt,
                           require_learning_receipts, resume, retain_checkpoints, validate_preflight)


def started_at() -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=10)).isoformat()


def qualified_fixture(root: Path, *, processor_id: str = "processor-a", qualification: str = "first") -> dict:
    data = root / "data"
    prepared = data / "prepared/train.jsonl"
    if not prepared.exists():
        prepared.parent.mkdir(parents=True)
        row = record(split="train", source="synthetic-visuals-v1", group_id="scene-one", family="lookup",
                     modality="image", state="", question=noul("Is the visible value one?"), label="true",
                     images=["image.bin"], provenance={"license": "CC0-1.0"})
        prepared.write_text(json.dumps(row) + "\n")
        (data / "image.bin").write_bytes(b"first image bytes")
        (root / "pyproject.toml").write_text("")
        (root / "uv.lock").write_text("")
    row = json.loads(prepared.read_text())
    media = {"image.bin": sha256_file(data / "image.bin")}
    prepared_hashes = {"train": sha256_file(prepared)}
    checks = data / "processor-checks.jsonl"
    checks.write_text(json.dumps({"id": row["id"], "positions": 32, "within_budget": True}) + "\n")
    processor = {"status": "VERIFIED", "all_questions_checked": True, "questions_checked": 1,
                 "failures": 0, "over_budget": 0, "model_id": MODEL_ID, "model_revision": REVISION,
                 "max_positions": 2048, "scope": ["train"], "prepared_hashes": prepared_hashes,
                 "media_hashes": media, "checks_path": checks.name, "checks_sha256": sha256_file(checks),
                 "processor_id": processor_id}
    atomic_json(data / "processor-contract.json", processor)
    source = {"status": "VERIFIED", "method": "isolated fixture"}
    verification = {"status": "VERIFIED", "all_questions_checked": True, "scope": ["train"],
                    "questions_checked": 1, "prepared_hashes": prepared_hashes, "media_hashes": media,
                    "source_reconciliation": source, "source_receipt_sha256": object_hash(source),
                    "human_review": {"status": "NOT_VERIFIED"}, "row_receipts": {row["id"]: object_hash(row)},
                    "qualification": qualification}
    verification["qualification_sha256"] = object_hash(verification)
    atomic_json(data / "verification-contract.json", verification)
    return qualified_inputs(root)


def preflight_fixture(root: Path, path: Path, ledger: BudgetLedger, commitment: dict) -> None:
    atomic_json(path, {"status": "VERIFIED", "device": "cuda", "recipe": "mixed",
                       "ledger_id": ledger.record["id"], "code_hashes": code_identity(root),
                       "cuda_device": {"uuid": "gpu-one"}, "group_seconds": 1., **commitment})


def probe_examples() -> tuple[TrainingExample, ...]:
    return tuple(TrainingExample(
        DecisionInput(identity, "evidence", "choice", "Choose", (Candidate("yes", "Yes"), Candidate("no", "No")), (), ()),
        0, identity, "generated_rule", "mdprep_text_rules", "fixture", "none")
        for identity in ("q1", "q2"))


class ProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.zeros(2, 2))
        self.dropout = torch.nn.Dropout(p=1.)
        self.compiler = SimpleNamespace(compile=lambda items: items[0].id)
        self.device = torch.device("cpu")

    def forward(self, identity):
        return self.dropout(self.values[0 if identity == "q1" else 1]).unsqueeze(0)


class HistoricalIdentityTests(unittest.TestCase):
    def test_requalified_media_invalidates_view_preflight_stages_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = qualified_fixture(root)
            before = DatasetView.open(root, split="train")
            rows = (root / "data/prepared/train.jsonl").read_bytes()
            with BudgetLedger(root / "budget.json", paid_started_at=started_at(), hourly_usd=.89) as ledger:
                receipt = root / "preflight.json"
                preflight_fixture(root, receipt, ledger, original)
                with patch("mdtrain.train.cuda_device_identity", return_value={"uuid": "gpu-one"}):
                    self.assertEqual(validate_preflight(root, receipt, "mixed", ledger)["processor_id"], "processor-a")
                baseline = {"ids": [before.examples[0].input.id], "target_indices": [1], "target_keys": ["true"],
                            "mode": "eval", "statistic": "question_mean_uncalibrated_nll", "nll": 1.}
                final = {**baseline, "nll": .5}
                for stage in ("diagnostic", "pilot"):
                    ledger.receipt("mixed/" + stage, {"accepted": True, "dataset_id": before.identity,
                        "qualified_inputs": original, "learning_evidence": learning_evidence(baseline, final)})
                require_learning_receipts(ledger, "mixed", before, original)
                checkpoint = root / "run/checkpoints/generation-00000001"
                checkpoint.mkdir(parents=True)
                orders = {"head": [before.examples[0].input.id], "joint": [before.examples[0].input.id]}
                spec = {"stage": "pilot", "recipe": "mixed", "dataset_id": before.identity,
                        "code_hashes": code_identity(root), "qualified_inputs": original, **original}
                atomic_json(checkpoint / "state.json", {"spec": spec, "ledger_id": ledger.record["id"],
                    "orders": orders, "orders_id": digest_json(orders), "backend": backend_identity("cpu")})
                seal(checkpoint)
                (root / "data/image.bin").write_bytes(b"changed and requalified image")
                current = qualified_fixture(root)
                after = DatasetView.open(root, split="train")
                self.assertEqual((root / "data/prepared/train.jsonl").read_bytes(), rows)
                self.assertEqual(after.examples[0].input.id, before.examples[0].input.id)
                self.assertNotEqual(after.identity, before.identity)
                with patch("mdtrain.train.cuda_device_identity", return_value={"uuid": "gpu-one"}):
                    with self.assertRaisesRegex(ValueError, "commitment is stale"):
                        validate_preflight(root, receipt, "mixed", ledger)
                with self.assertRaisesRegex(ValueError, "fixed-question diagnostic"):
                    require_learning_receipts(ledger, "mixed", after, current)
                with self.assertRaisesRegex(ValueError, "Checkpoint media"):
                    load_checkpoint(checkpoint, root=root, examples=after.by_id(), device="cpu",
                                    ledger_id=ledger.record["id"], require_latest=False)
                preflight_fixture(root, receipt, ledger, current)
            with patch("mdtrain.train.cuda_device_identity", return_value={"uuid": "gpu-one"}):
                with self.assertRaisesRegex(ValueError, "changed since checkpoint"):
                    resume(root, checkpoint=checkpoint, preflight=receipt, budget_ledger=root / "budget.json")

    def test_processor_or_successful_qualification_change_invalidates_historical_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = qualified_fixture(root)
            before = DatasetView.open(root, split="train")
            changed = qualified_fixture(root, processor_id="processor-b")
            after = DatasetView.open(root, split="train")
            self.assertEqual(changed["media_hashes"], first["media_hashes"])
            self.assertNotEqual(after.identity, before.identity)
            requalified = qualified_fixture(root, processor_id="processor-b", qualification="second successful audit")
            self.assertEqual(requalified["processor_id"], "processor-b")
            self.assertNotEqual(DatasetView.open(root, split="train").identity, after.identity)
            with BudgetLedger(root / "budget.json", paid_started_at=started_at(), hourly_usd=.89) as ledger:
                receipt = root / "preflight.json"
                preflight_fixture(root, receipt, ledger, first)
                with patch("mdtrain.train.cuda_device_identity", return_value={"uuid": "gpu-one"}):
                    with self.assertRaisesRegex(ValueError, "commitment is stale"):
                        validate_preflight(root, receipt, "mixed", ledger)
                preflight_fixture(root, receipt, ledger, requalified)
                checkpoint = root / "run/checkpoints/generation-00000001"
                checkpoint.mkdir(parents=True)
                atomic_json(checkpoint / "state.json", {"spec": {"stage": "pilot", "recipe": "mixed",
                    "dataset_id": before.identity, "qualified_inputs": first, **first,
                    "code_hashes": code_identity(root)}})
            with patch("mdtrain.train.cuda_device_identity", return_value={"uuid": "gpu-one"}):
                with self.assertRaisesRegex(ValueError, "changed since checkpoint"):
                    resume(root, checkpoint=checkpoint, preflight=receipt, budget_ledger=root / "budget.json")

    def test_preflight_is_specific_to_the_current_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commitment = qualified_fixture(root)
            with BudgetLedger(root / "budget.json", paid_started_at=started_at(), hourly_usd=.89) as ledger:
                receipt = root / "preflight.json"
                preflight_fixture(root, receipt, ledger, commitment)
                with patch("mdtrain.train.cuda_device_identity", return_value={"uuid": "gpu-one"}):
                    self.assertEqual(validate_preflight(root, receipt, "mixed", ledger)["status"], "VERIFIED")
                with patch("mdtrain.train.cuda_device_identity", return_value={"uuid": "gpu-two"}):
                    with self.assertRaisesRegex(ValueError, "device is required"):
                        validate_preflight(root, receipt, "mixed", ledger)


class ExactReplayTests(unittest.TestCase):
    def test_replay_rejects_subtolerance_loss_tensor_optimizer_and_schedule_changes(self):
        expected = {"ids": ["q1"], "loss": .5, "weights": {"pointer": torch.tensor([1.])},
                    "optimizer": {"moment": torch.tensor([.25])}, "scheduler": {"last_epoch": 2},
                    "training": {"cursor": 16, "phase": "joint"}}
        assert_exact_replay(expected, copy.deepcopy(expected))
        for field in ("loss", "weights", "optimizer", "scheduler", "training"):
            changed = copy.deepcopy(expected)
            if field == "loss":
                changed[field] = math.nextafter(.5, 1.)
            elif field in ("weights", "optimizer"):
                key = "pointer" if field == "weights" else "moment"
                changed[field][key] = torch.nextafter(changed[field][key], torch.tensor([2.]))
            elif field == "scheduler":
                changed[field]["last_epoch"] = 3
            else:
                changed[field]["cursor"] = 17
            with self.subTest(field=field), self.assertRaises(AssertionError):
                assert_exact_replay(expected, changed)
        self.assertEqual(expected["training"], {"cursor": 16, "phase": "joint"})
        changed_dtype = copy.deepcopy(expected)
        changed_dtype["weights"]["pointer"] = changed_dtype["weights"]["pointer"].double()
        with self.assertRaisesRegex(AssertionError, "Exact next-update tensor"):
            assert_exact_replay(expected, changed_dtype)


class FixedLearningTests(unittest.TestCase):
    def test_probe_uses_same_targets_eval_mode_and_counts_durable_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            with BudgetLedger(Path(temporary) / "budget.json", paid_started_at=started_at(), hourly_usd=.89) as ledger:
                model = ProbeModel()
                before_elapsed = ledger.record["elapsed_seconds"]
                rng = torch.get_rng_state().clone()
                baseline = fixed_probe(model, probe_examples(), ledger=ledger, question_seconds=.01)
                with torch.no_grad():
                    model.values[:, 0] = math.log(3)
                final = fixed_probe(model, probe_examples(), ledger=ledger, question_seconds=.01)
                evidence = learning_evidence(baseline, final)
                self.assertAlmostEqual(baseline["nll"], math.log(2), places=6)
                self.assertAlmostEqual(final["nll"], math.log(4 / 3), places=6)
                self.assertAlmostEqual(evidence["reduction"], math.log(1.5), places=6)
                self.assertEqual(evidence["ids"], ["q1", "q2"])
                self.assertEqual(evidence["target_keys"], ["yes", "yes"])
                self.assertEqual(model.training, True)
                self.assertEqual(torch.equal(rng, torch.get_rng_state()), True)
                self.assertGreater(ledger.record["elapsed_seconds"], before_elapsed)
                with self.assertRaisesRegex(ValueError, "changed its fixed questions"):
                    learning_evidence(baseline, {**final, "ids": ["q2", "q1"]})

    def test_different_question_update_trace_cannot_authorize_learning(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with BudgetLedger(directory / "budget.json", paid_started_at=started_at(), hourly_usd=.89) as ledger:
                model = ProbeModel()
                examples = probe_examples()
                baseline = fixed_probe(model, examples, ledger=ledger, question_seconds=.01)
                trainer = SimpleNamespace(model=model, examples={item.input.id: item for item in examples},
                    state=TrainingState(losses=[20.] * 4 + [.001] * 4, learning_probe={"baseline": baseline}),
                    spec={"run_id": "fixed-probe-fixture", "stage": "pilot", "recipe": "mixed", "dataset_id": "fixed-inputs", "qualified_inputs": {},
                          "probe_question_seconds": .01, "gate": {"minimum_loss_reduction": 0.}})
                with patch("mdtrain.train.save_checkpoint", return_value=directory / "checkpoint"):
                    result = record_run_receipt(trainer, {"status": "completed"}, directory, ledger)
                self.assertEqual(result["accepted_learning_gate"], False)
                self.assertEqual(result["learning_evidence"]["reduction"], 0.)
                self.assertEqual(ledger.record["receipts"]["mixed/pilot"]["accepted"], False)
                self.assertEqual(result["update_losses_are_diagnostic_only"], True)


class StrictReportTests(unittest.TestCase):
    def test_infinite_nll_is_lossless_standard_json_at_report_boundary(self):
        rows = [{"id": "zero-target", "group_id": "g1", "probabilities": {"a": 0., "b": 1.},
                 "target": "a", "source": "fixture", "origin": "generated_rule", "modality": "text",
                 "primitive": "choice", "family": "fixture", "ood_axis": "none"}]
        metrics = score_predictions(rows)
        self.assertEqual(metrics["nll"], math.inf)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            atomic_json(path, report_json({"metrics": metrics}))
            def reject_constant(value):
                raise AssertionError("Nonstandard JSON constant " + value)
            parsed = json.loads(path.read_text(), parse_constant=reject_constant)
            self.assertEqual(parsed["metrics"]["nll"], None)
            self.assertEqual(parsed["nonfinite_values"]["/metrics/nll"], "positive_infinity")
            self.assertEqual(parsed["metrics"]["accuracy"], 0.)
            self.assertEqual(metrics["nll"], math.inf)
            with self.assertRaises(ValueError):
                atomic_json(path, {"nll": math.inf})
            self.assertEqual(json.loads(path.read_text())["metrics"]["accuracy"], 0.)

    def test_nested_infinite_bounds_are_explicit_and_nan_remains_an_error(self):
        encoded = report_json({"a/b": {"~bounds": [1., math.inf, -math.inf]}})
        self.assertEqual(encoded, {"a/b": {"~bounds": [1., None, None]}, "nonfinite_values": {
            "/a~1b/~0bounds/1": "positive_infinity", "/a~1b/~0bounds/2": "negative_infinity"}})
        with self.assertRaisesRegex(ValueError, "NaN report value"):
            report_json({"nll": math.nan})


class CheckpointDiskTests(unittest.TestCase):
    def test_low_disk_refuses_publication_without_deleting_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary)
            previous = checkpoints / "generation-00000001"
            previous.mkdir()
            (previous / "valid").write_text("last recovery")
            with patch("mdtrain.train.shutil.disk_usage", return_value=SimpleNamespace(free=1)):
                with self.assertRaisesRegex(OSError, "preserving latest"):
                    checkpoint_disk_admission(checkpoints, previous)
            self.assertEqual((previous / "valid").read_text(), "last recovery")

    def test_retention_keeps_latest_and_previous_published_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary)
            for name in ("generation-00000001", "generation-00000002", "generation-00000003", ".pending-orphan"):
                (checkpoints / name).mkdir()
                (checkpoints / name / "value").write_text(name)
            previous, current = checkpoints / "generation-00000002", checkpoints / "generation-00000003"
            (checkpoints / "latest").symlink_to(current.name, target_is_directory=True)
            retain_checkpoints(checkpoints, current=current, previous=previous)
            self.assertEqual(sorted(path.name for path in checkpoints.iterdir()),
                             ["generation-00000002", "generation-00000003", "latest"])
            self.assertEqual((checkpoints / "latest/value").read_text(), "generation-00000003")
            self.assertEqual((previous / "value").read_text(), "generation-00000002")


class ProcessDeadlineTests(unittest.TestCase):
    def test_cumulative_deadline_counts_prior_allocations_downtime_and_reserve(self):
        record = {"allowance_seconds": 1000, "reserve_seconds": 120, "elapsed_seconds": 140,
                  "allocations": [{"started_utc": 0, "closed_utc": 100}, {"started_utc": 200, "closed_utc": None}]}
        deadline = ledger_deadline(record, now_utc=250, now_monotonic=10)
        self.assertEqual(dataclasses.asdict(deadline), {"hard_monotonic": 859., "reserve_seconds": 120})
        record["process_deadline_elapsed"] = 200
        self.assertEqual(ledger_deadline(record, now_utc=250, now_monotonic=10).hard_monotonic, 59.)

    def test_graceful_reserve_allows_a_boundary_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "checkpoint"
            code = ("import pathlib,signal,sys,time\n"
                    "path=pathlib.Path(sys.argv[1])\n"
                    "def save(signum,frame):\n"
                    "    path.write_text('published boundary')\n"
                    "    sys.exit(0)\n"
                    "signal.signal(signal.SIGTERM,save)\n"
                    "time.sleep(60)\n")
            result = supervise([sys.executable, "-c", code, str(marker)],
                               deadline=ProcessDeadline(time.monotonic() + 2., 1.))
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(result["graceful_requested"], True)
            self.assertEqual(result["hard_stopped"], False)
            self.assertEqual(marker.read_text(), "published boundary")
            self.assertEqual(result["timer_is_billing_guarantee"], False)

    def test_hard_deadline_kills_blocked_group_and_preserves_previous_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "checkpoint"
            checkpoint.write_text("previous complete optimizer boundary")
            heartbeat = directory / "heartbeat"
            graceful = directory / "graceful"
            descendant = ("import pathlib,sys,time\npath=pathlib.Path(sys.argv[1])\n"
                          "while True:\n    path.write_text(str(time.monotonic_ns()))\n    time.sleep(.03)\n")
            code = ("import pathlib,signal,subprocess,sys,time\n"
                    "signal.signal(signal.SIGTERM,lambda s,f:pathlib.Path(sys.argv[2]).write_text('requested'))\n"
                    "subprocess.Popen([sys.executable,'-c',sys.argv[3],sys.argv[1]])\n"
                    "time.sleep(60)\n")
            start = time.monotonic()
            result = supervise([sys.executable, "-c", code, str(heartbeat), str(graceful), descendant],
                               deadline=ProcessDeadline(start + 2., 1.))
            self.assertEqual(result["returncode"], -signal.SIGKILL)
            self.assertEqual(result["hard_stopped"], True)
            self.assertEqual(graceful.read_text(), "requested")
            first = heartbeat.read_text()
            time.sleep(.15)
            self.assertEqual(heartbeat.read_text(), first)
            self.assertEqual(checkpoint.read_text(), "previous complete optimizer boundary")
            self.assertLess(time.monotonic() - start, 4.)

    def test_guardian_does_not_compete_for_worker_ledger_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "budget.json"
            with BudgetLedger(path, paid_started_at=started_at(), hourly_usd=.89) as ledger:
                ledger.record["reserve_seconds"] = 1.
                identity = ledger.record["id"]
            acquired = directory / "acquired"
            code = ("import fcntl,pathlib,signal,sys,time\n"
                    "stream=open(sys.argv[1],'a+')\n"
                    "fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
                    "pathlib.Path(sys.argv[2]).write_text('worker owns ledger')\n"
                    "signal.signal(signal.SIGTERM,lambda s,f:sys.exit(0))\n"
                    "time.sleep(60)\n")
            result = supervise([sys.executable, "-c", code, str(path.with_suffix('.json.lock')), str(acquired)],
                               deadline=ProcessDeadline(time.monotonic() + 2., 1.),
                               ledger_path=path, ledger_id=identity)
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(acquired.read_text(), "worker owns ledger")


if __name__ == "__main__":
    unittest.main()
