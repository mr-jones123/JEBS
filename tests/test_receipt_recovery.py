"""Recover a published learning checkpoint after its ledger publication crashes."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from mdprep.native import NativeCompiler
from mdtrain.budget import BudgetLedger
from mdtrain.data import DatasetView
from mdtrain.engine import DecisionModel
from mdtrain.train import (UpdateTrainer, code_identity, deterministic, fixed_probe,
                           load_checkpoint, record_run_receipt, resume)
from test_training_guards import preflight_fixture, qualified_fixture, started_at

ROOT=Path(__file__).resolve().parents[1]


class PublishedLearningRecoveryTests(unittest.TestCase):
    def test_resume_reconciles_receipt_without_retraining_and_rejects_conflicts(self):
        deterministic()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            qualified_fixture(root)
            Image.new("RGB",(256,256),(20,80,140)).save(root/"data/image.bin",format="PNG")
            compiler=NativeCompiler.from_pretrained(ROOT,media_root=root/"data")
            commitment=qualified_fixture(root,processor_id=compiler.processor_id)
            view=DatasetView.open(root,split="train")
            examples=view.by_id()
            ids=list(examples)
            model=DecisionModel.create(ROOT,compiler,kind="tiny",device="cpu")
            run_dir=root/"run"
            receipt=root/"preflight.json"
            ledger_path=root/"budget.json"
            with BudgetLedger(ledger_path, paid_started_at=started_at(), hourly_usd=.89) as ledger:
                preflight_fixture(root,receipt,ledger,commitment)
                spec={"run_id":"receipt-crash","stage":"pilot","recipe":"mixed",
                      "dataset_id":view.identity,"qualified_inputs":commitment,
                      **commitment,"code_hashes":code_identity(root),
                      "probe_question_seconds":.01,"gate":{"minimum_loss_reduction":0.}}
                trainer=UpdateTrainer(model,examples,{"head":ids*32,"joint":ids*16},spec)
                trainer.state.learning_probe={"baseline":fixed_probe(model,view.examples,ledger=ledger,question_seconds=.01)}
                trainer.step(); trainer.step(); trainer.transition(); trainer.step(); trainer.transition()
                self.assertEqual(trainer.state.status,"completed")
                with patch.object(ledger,"receipt",side_effect=OSError("receipt publication crash")):
                    with self.assertRaisesRegex(OSError,"receipt publication crash"):
                        record_run_receipt(trainer,{"status":"completed"},run_dir,ledger)
                checkpoint=(run_dir/"checkpoints/latest").resolve()
                state_before=(checkpoint/"state.json").read_bytes()
                saved=json.loads(state_before)
                self.assertGreater(saved["training"]["learning_probe"]["evidence"]["reduction"],0.)
                self.assertNotIn("mixed/pilot",ledger.record["receipts"])
                expected_updates=trainer.state.global_updates
                expected_exposures=trainer.state.consumed_examples

            def load_on_cpu(checkpoint,**kwargs):
                kwargs["device"]="cpu"
                return load_checkpoint(checkpoint,**kwargs)

            with patch("mdtrain.train.cuda_device_identity",return_value={"uuid":"gpu-one"}), \
                    patch("mdtrain.train.load_checkpoint",side_effect=load_on_cpu):
                first=resume(root,checkpoint=checkpoint,preflight=receipt,budget_ledger=ledger_path)
                second=resume(root,checkpoint=checkpoint,preflight=receipt,budget_ledger=ledger_path)
                self.assertEqual(first["accepted_learning_gate"],True)
                self.assertEqual(second,first)
                self.assertEqual(first["updates"],expected_updates)
                self.assertEqual(first["consumed_examples"],expected_exposures)
                self.assertEqual((checkpoint/"state.json").read_bytes(),state_before)
                self.assertEqual((run_dir/"checkpoints/latest").resolve(),checkpoint)
                with BudgetLedger(ledger_path) as ledger:
                    published=copy.deepcopy(ledger.record["receipts"]["mixed/pilot"])
                    self.assertEqual(published["learning_evidence"],saved["training"]["learning_probe"]["evidence"])
                    ledger.record["receipts"]["mixed/pilot"]["learning_evidence"]["reduction"]=999.
                with self.assertRaises(ValueError):
                    resume(root,checkpoint=checkpoint,preflight=receipt,budget_ledger=ledger_path)
                current=json.loads(ledger_path.read_text())["receipts"]["mixed/pilot"]
                self.assertEqual(current["learning_evidence"]["reduction"],999.)


if __name__=="__main__":
    unittest.main()
