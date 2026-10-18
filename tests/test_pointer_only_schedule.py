"""Pointer-only schedule planning, identity, and phase-boundary regressions."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from mdprep.native import digest_json
from mdtrain.__main__ import parser
from mdtrain.data import DatasetView, plan_orders
from mdtrain.train import (UpdateTrainer, projected_training_seconds, schedule_fields,
                           validate_preflight)


class PhaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pointer = torch.nn.Linear(1, 1)
        self.body = torch.nn.Linear(1, 1)
        self.adapter_names = ()
        self.phase = "head"

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.body.requires_grad_(phase == "joint")
        self.pointer.requires_grad_(True)


class PointerOnlyScheduleTests(unittest.TestCase):
    def view(self, count: int = 20) -> tuple[DatasetView, dict[str, int]]:
        examples = tuple(SimpleNamespace(
            origin="fixture", input=SimpleNamespace(id=f"q-{index:03d}", kind="text"))
            for index in range(count))
        rows = tuple({"modality": "text"} for _ in examples)
        return DatasetView(examples, rows, "train", "mixed", "fixture"), {
            example.input.id: 32 + index for index, example in enumerate(examples)}

    def test_pointer_only_orders_are_deterministic_and_preserve_paired_orders(self):
        view, lengths = self.view()
        paired = plan_orders(view, "main", lengths)
        legacy = {
            "head": [paired["selection"][index % len(view.examples)] for index in range(100 * 16)],
            "joint": paired["selection"], "selection": paired["selection"],
            "head_updates": 100, "joint_updates": 2, "without_replacement_joint": True,
        }
        self.assertEqual(digest_json(paired), digest_json(legacy))
        first = plan_orders(view, "main", lengths, "pointer-only")
        second = plan_orders(view, "main", lengths, "pointer-only")
        self.assertEqual(digest_json(first), digest_json(second))
        self.assertEqual(len(first["head"]), 1600 + len(view.examples))
        self.assertEqual(first["joint"], [])
        self.assertEqual(first["joint_updates"], 0)
        self.assertTrue({example.input.id for example in view.examples} <= set(first["head"]))

    def test_transition_and_projection_follow_bound_schedule(self):
        trainer = UpdateTrainer(PhaseModel(), {}, {"head": ["q"], "joint": []},
                                {"run_id": "pointer", "stage": "smoke",
                                 "schedule": "pointer-only"})
        trainer.state.cursor = 1
        self.assertTrue(trainer.transition())
        self.assertEqual(trainer.state.status, "completed")
        self.assertEqual(trainer.state.phase, "head")
        orders = {"head": ["q"] * 32, "joint": ["q"] * 16}
        receipt = {"group_seconds": 99., "phase_seconds": {"head": 2., "joint": 7.}}
        self.assertEqual(projected_training_seconds(orders, receipt, "pointer-only"), 4.)
        self.assertEqual(projected_training_seconds(orders, receipt, "paired"), 11.)

    def test_schedule_identity_and_cli_default(self):
        legacy_spec = {"recipe": "mixed", "stage": "main"}
        paired_spec = {**legacy_spec, **schedule_fields("paired")}
        self.assertEqual(digest_json(legacy_spec), digest_json(paired_spec))
        self.assertEqual(schedule_fields("pointer-only"), {"schedule": "pointer-only"})
        default = parser().parse_args(["smoke", "--kind", "tiny", "--output", "out.json"])
        selected = parser().parse_args(["smoke", "--kind", "tiny", "--output", "out.json",
                                        "--schedule", "pointer-only"])
        self.assertEqual(default.schedule, "paired")
        self.assertEqual(selected.schedule, "pointer-only")

    def test_cross_schedule_preflight_is_rejected_before_device_use(self):
        ledger = SimpleNamespace(record={"id": "ledger"})
        for recorded, requested in (("paired", "pointer-only"), ("pointer-only", "paired")):
            with self.subTest(recorded=recorded, requested=requested):
                with tempfile.TemporaryDirectory() as temporary:
                    receipt = Path(temporary) / "preflight.json"
                    receipt.write_text('{"status":"VERIFIED","schedule":"%s"}' % recorded)
                    with self.assertRaisesRegex(ValueError, "schedule does not match"):
                        validate_preflight(Path(temporary), receipt, "mixed", ledger,
                                           schedule=requested)


if __name__ == "__main__":
    unittest.main()
