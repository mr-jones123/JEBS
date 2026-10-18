"""Basic grouped scoring and optional scalar-temperature finalization. Policy stays disabled."""

import contextlib
import json
import os
import tempfile
from pathlib import Path

from mdtrain.budget import BudgetLedger
from mdtrain.data import DatasetView, EVALUATION_SPLITS
from mdtrain.engine import DecisionModel
from mdtrain.fixtures import read_requests
from mdtrain.metrics import fit_temperature, score_predictions
from mdtrain.storage import atomic_json, report_json, verify_seal


def prediction_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def gpu_allowance(device: str, path: Path | None):
    # Rented pods pass a ledger to account paid time; a user's own GPU needs none.
    if device == "cuda" and path is not None:
        return BudgetLedger(path)
    return contextlib.nullcontext(None)


def evaluate_model(model: DecisionModel, view: DatasetView, *, ledger: BudgetLedger | None = None) -> tuple[list[dict], dict]:
    rows = []
    for example, canonical in zip(view.examples, view.rows):
        if ledger is not None:
            ledger.admit(60)
        batch = model.compiler.compile((example.input,))
        prediction = model.predict((example.input,))[0]
        expected_keys = tuple(candidate.key for candidate in example.input.candidates)
        if prediction["id"] != example.input.id or tuple(prediction["probabilities"]) != expected_keys:
            raise ValueError("Prediction identity/candidate alignment differs from requested view")
        rows.append({**prediction, "group_id": example.group_id,
                     "target": expected_keys[example.target_index], "source": example.source,
                     "origin": example.origin, "modality": canonical["modality"],
                     "primitive": example.input.kind, "family": example.family, "ood_axis": example.ood_axis,
                     "position_count": int(batch.expanded_lengths[0])})
        if ledger is not None:
            ledger.commit()
    metrics = score_predictions(rows)
    real_rows = [row for row in rows if row["origin"] == "real_human_label"]
    generated_rows = [row for row in rows if row["origin"] != "real_human_label"]
    return rows, {"all": metrics, "real": score_predictions(real_rows) if real_rows else None,
                  "generated": score_predictions(generated_rows) if generated_rows else None}


def predict(root: Path, *, artifact: Path, input_path: Path, media_root: Path, output: Path,
            device: str, budget_ledger: Path | None = None) -> dict:
    requests = read_requests(input_path)
    with gpu_allowance(device, budget_ledger) as ledger:
        if ledger:
            ledger.admit(300)
        model = DecisionModel.load(artifact, root=root, device=device, media_root=media_root)
        rows = []
        for item in requests:
            if ledger:
                ledger.admit(60)
            rows.extend(model.predict((item,)))
        prediction_file(output, rows)
        return {"status": "completed", "questions": len(rows), "artifact_id": model.artifact_id,
                "output": str(output), "policy": "disabled"}


def evaluate(root: Path, *, artifact: Path, splits: tuple[str, ...], output: Path,
             device: str, budget_ledger: Path | None = None) -> dict:
    if not splits or len(set(splits)) != len(splits) or any(split not in EVALUATION_SPLITS for split in splits):
        raise ValueError("Evaluation requires distinct declared evaluation splits")
    with gpu_allowance(device, budget_ledger) as ledger:
        if ledger:
            ledger.admit(300)
        model = DecisionModel.load(artifact, root=root, device=device)
        if any(split.startswith("test_") for split in splits) and not model.finalized:
            raise ValueError("Locked final-test evaluation requires a finalized paired artifact")
        result = {"status": "completed", "artifact_id": model.artifact_id,
                  "temperature": model.temperature, "policy": "disabled", "splits": {}}
        for split in splits:
            view = DatasetView.open(root, split=split, recipe="mixed")
            rows, metrics = evaluate_model(model, view, ledger=ledger)
            prediction_file(output.with_name(output.stem + f"-{split}-predictions.jsonl"), rows)
            result["splits"][split] = {"view_identity": view.identity, "metrics": metrics}
        result = report_json(result)
        atomic_json(output, result)
        return result


def finalize(root: Path, *, checkpoint: Path, output: Path, device: str,
             fit_scalar_temperature: bool = False, budget_ledger: Path | None = None) -> dict:
    checkpoint = checkpoint.resolve(strict=True)
    checkpoint_id = verify_seal(checkpoint)
    state = json.loads((checkpoint / "state.json").read_text())
    if state["training"]["status"] != "completed" or state["spec"]["stage"] != "main":
        raise ValueError("Only a completed main checkpoint may be finalized")
    with gpu_allowance(device, budget_ledger) as ledger:
        if ledger:
            ledger.admit(300)
            if state["ledger_id"] != ledger.record["id"]:
                raise ValueError("Finalization must use the same durable budget ledger")
        model = DecisionModel.load(checkpoint / "inference", root=root, device=device)
        calibration_identity = None
        if fit_scalar_temperature:
            view = DatasetView.open(root, split="calibration", recipe="mixed")
            logits, targets = [], []
            for example in view.examples:
                if ledger:
                    ledger.admit(60)
                logits.append(model.logits((example.input,))[0])
                targets.append(example.target_index)
            model.temperature = fit_temperature(logits, targets)
            model.temperature_fitted = True
            calibration_identity = view.identity
        model.finalized = True
        model.save(output, metadata={"checkpoint_id": checkpoint_id, "training_status": "completed",
                                     "calibration_view_id": calibration_identity, "policy": "disabled"})
        receipt = {"status": "completed", "artifact": str(output), "artifact_id": model.artifact_id,
                   "checkpoint_id": checkpoint_id, "temperature": model.temperature,
                   "temperature_fitted": fit_scalar_temperature, "calibration_view_id": calibration_identity,
                   "policy": "disabled", "weights_selected_using_final_tests": False}
        atomic_json(output.with_name(output.name + "-freeze-receipt.json"), receipt)
        return receipt
