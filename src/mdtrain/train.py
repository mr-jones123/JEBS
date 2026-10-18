"""One explicit microbatch-one, accumulation-sixteen update loop and recovery path."""

import dataclasses
import json
import math
import os
import random
import shutil
import signal
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mdprep.common import sha256_file
from mdprep.native import NativeCompiler, digest_json, validate_contract
from mdprep.records import TrainingExample
from mdtrain.budget import BudgetLedger, BudgetStopped
from mdtrain.data import (DEFAULT_SCHEDULE, OPTIMIZER_PHASES, DatasetView, bound_schedule, plan_orders,
                          qualified_inputs, training_schedule)
from mdtrain.engine import DecisionModel
from mdtrain.storage import atomic_json, seal, sync_directory, verify_seal, writer_lock


MAIN_STAGE_SECONDS = 10 * 3600


def admissible_seconds(stage: str, projected: float) -> float:
    """Return the horizon an admission check has to fit.

    Main stops itself at its stage cap and publishes a checkpoint there, so only
    the capped horizon needs to fit. Comparing the uncapped projection against a
    deadline already tightened to that cap refused every main stage.
    """
    return min(projected, MAIN_STAGE_SECONDS) if stage == "main" else projected


def schedule_fields(schedule: str) -> dict:
    training_schedule(schedule)
    return {} if schedule == DEFAULT_SCHEDULE else {"schedule": schedule}


def stage_receipt_key(recipe: str, schedule: str, stage: str) -> str:
    return (f"{recipe}/{stage}" if schedule == DEFAULT_SCHEDULE
            else f"{recipe}/{schedule}/{stage}")


def projected_training_seconds(orders: dict, receipt: dict, schedule: str) -> float:
    selected = training_schedule(schedule)
    phase_seconds = receipt.get("phase_seconds", {})
    return sum(math.ceil(len(orders[phase]) / 16)
               * phase_seconds.get(phase, receipt["group_seconds"])
               for phase in selected.phases)


def cuda_device_identity() -> dict:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    identity = getattr(properties, "uuid", None)
    if identity is None:
        raise ValueError("CUDA device UUID is required to reuse hardware qualification")
    return {"index": index, "uuid": str(identity), "name": properties.name,
            "total_memory": properties.total_memory,
            "capability": [properties.major, properties.minor]}


def backend_identity(device: str) -> dict:
    return {"device": device, "torch": torch.__version__, "cuda": torch.version.cuda,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cuda_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "cuda_device": cuda_device_identity() if device == "cuda" else None,
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "threads": torch.get_num_threads()}


def deterministic(seed: int = 1729) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@dataclass
class TrainingState:
    phase: str = "head"
    status: str = "running"
    cursor: int = 0
    phase_updates: int = 0
    global_updates: int = 0
    consumed_examples: int = 0
    checkpoint_generation: int = 0
    elapsed_allowance: float = 0.0
    main_elapsed: float = 0.0
    losses: list[float] = field(default_factory=list)
    learning_probe: dict = field(default_factory=dict)


class UpdateTrainer:
    def __init__(self, model: DecisionModel, examples: dict[str, TrainingExample], orders: dict,
                 spec: dict, *, state: TrainingState | None = None):
        self.model, self.examples, self.orders, self.spec = model, examples, orders, spec
        self.schedule = training_schedule(bound_schedule(spec))
        if (any(phase not in orders for phase in self.schedule.phases)
                or any(orders.get(phase) for phase in OPTIMIZER_PHASES
                       if phase not in self.schedule.phases)):
            raise ValueError("Training orders do not implement the bound schedule")
        self.state = state or TrainingState(phase=self.schedule.phases[0])
        if self.state.phase not in self.schedule.phases:
            raise ValueError("Training state phase is outside the bound schedule")
        self.optimizer = None
        self.scheduler = None
        self.begin_phase(self.state.phase)

    def begin_phase(self, phase: str) -> None:
        self.model.set_phase(phase)
        if phase == "head":
            groups = [{"params": list(self.model.pointer.parameters()), "lr": 1e-3}]
        elif phase == "joint":
            adapters = [parameter for name, parameter in self.model.body.named_parameters()
                        if name in self.model.adapter_names]
            groups = [{"params": adapters, "lr": 5e-5},
                      {"params": list(self.model.pointer.parameters()), "lr": 1e-4}]
        else:
            raise ValueError("Unknown optimizer phase")
        self.optimizer = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8, weight_decay=.01)
        updates = max(1, math.ceil(len(self.orders[phase]) / 16))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lambda index: max(0., 1 - index / updates))

    def transition(self) -> bool:
        if self.state.cursor < len(self.orders[self.state.phase]):
            return False
        phase_index = self.schedule.phases.index(self.state.phase)
        if phase_index + 1 < len(self.schedule.phases):
            self.state.phase = self.schedule.phases[phase_index + 1]
            self.state.cursor = 0
            self.state.phase_updates = 0
            self.begin_phase(self.state.phase)
            return True
        self.state.status = "completed"
        return True

    def next_ids(self) -> list[str]:
        return self.orders[self.state.phase][self.state.cursor:self.state.cursor + 16]

    def step(self) -> dict:
        ids = self.next_ids()
        if not ids or len(ids) > 16:
            raise ValueError("No admissible accumulation group remains")
        self.model.train(True)
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        start = time.monotonic()
        for identity in ids:
            example = self.examples[identity]
            batch = self.model.compiler.compile((example.input,))
            logits = self.model(batch)
            target = torch.tensor([example.target_index], dtype=torch.long, device=self.model.device)
            with torch.autocast(device_type=self.model.device.type, enabled=False):
                loss = F.cross_entropy(logits.float(), target) / len(ids)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite candidate cross-entropy")
            loss.backward()
            losses.append(float(loss.detach()) * len(ids))
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        missing = [name for name, parameter in self.model.named_parameters()
                   if parameter.requires_grad and parameter.grad is None]
        if missing:
            raise ValueError(f"Intended trainables have no gradient: {missing}")
        if any(parameter.grad is not None for parameter in self.model.parameters() if not parameter.requires_grad):
            raise ValueError("A frozen base parameter received a gradient")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        if not norm > 0:
            raise ValueError("All intended gradients are zero")
        head_nonzero = all(torch.count_nonzero(p.grad) for p in self.model.pointer.parameters())
        adapter_nonzero = any(torch.count_nonzero(p.grad) for name, p in self.model.body.named_parameters()
                              if name in self.model.adapter_names and p.grad is not None)
        if not head_nonzero or (self.state.phase == "joint" and not adapter_nonzero):
            raise ValueError("Expected nonzero pointer and joint adapter gradients")
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.model.device.type == "cuda":
            torch.cuda.synchronize()
        self.state.cursor += len(ids)
        self.state.phase_updates += 1
        self.state.global_updates += 1
        self.state.consumed_examples += len(ids)
        self.state.losses.append(sum(losses) / len(losses))
        return {"ids": ids, "loss": self.state.losses[-1], "seconds": time.monotonic() - start,
                "gradient_norm": float(norm), "head_gradient_nonzero": bool(head_nonzero),
                "adapter_gradient_nonzero": bool(adapter_nonzero)}

def fixed_probe(model: DecisionModel, examples: tuple[TrainingExample, ...], *,
                ledger: BudgetLedger, question_seconds: float) -> dict:
    """Measure uncalibrated question-mean NLL on fixed targets in evaluation mode."""
    if not examples:
        raise ValueError("Fixed learning probe requires admitted questions")
    rng, training = capture_rng(), model.training
    losses = []
    try:
        model.eval()
        with torch.no_grad():
            for example in examples:
                ledger.admit(question_seconds)
                logits = model(model.compiler.compile((example.input,)))
                target = torch.tensor([example.target_index], dtype=torch.long, device=model.device)
                loss = float(F.cross_entropy(logits.float(), target))
                if not math.isfinite(loss):
                    raise FloatingPointError("Nonfinite fixed-question NLL")
                losses.append(loss)
                ledger.commit()
    finally:
        model.train(training)
        restore_rng(rng)
    return {"ids": [example.input.id for example in examples],
            "target_indices": [example.target_index for example in examples],
            "target_keys": [example.input.candidates[example.target_index].key for example in examples],
            "mode": "eval", "statistic": "question_mean_uncalibrated_nll",
            "nll": sum(losses) / len(losses)}


def learning_evidence(baseline: dict, final: dict) -> dict:
    for field in ("ids", "target_indices", "target_keys", "mode", "statistic"):
        if baseline[field] != final[field]:
            raise ValueError("Learning probe changed its fixed questions, targets, or evaluation conditions")
    return {key: baseline[key] for key in ("ids", "target_indices", "target_keys", "mode", "statistic")} | {
        "baseline_nll": baseline["nll"], "final_nll": final["nll"],
        "reduction": baseline["nll"] - final["nll"]}


def require_learning_receipts(ledger: BudgetLedger, recipe: str, view: DatasetView,
                              commitment: dict, schedule: str = DEFAULT_SCHEDULE) -> None:
    for prerequisite in ("diagnostic", "pilot"):
        gate = ledger.record["receipts"].get(stage_receipt_key(recipe, schedule, prerequisite))
        evidence = gate.get("learning_evidence", {}) if gate else {}
        if (not gate or not gate["accepted"] or gate["dataset_id"] != view.identity
                or gate.get("qualified_inputs") != commitment
                or evidence.get("mode") != "eval"
                or evidence.get("statistic") != "question_mean_uncalibrated_nll"
                or not evidence.get("ids") or len(evidence["ids"]) != len(set(evidence["ids"]))
                or len(evidence.get("target_indices", [])) != len(evidence["ids"])
                or len(evidence.get("target_keys", [])) != len(evidence["ids"])
                or not all(math.isfinite(evidence.get(key, math.nan))
                           for key in ("baseline_nll", "final_nll", "reduction"))
                or evidence["reduction"] != evidence["baseline_nll"] - evidence["final_nll"]
                or evidence["reduction"] <= 0):
            raise ValueError(f"Main requires accepted matching fixed-question {prerequisite} evidence")
        current = view.by_id()
        for identity, target, key in zip(evidence["ids"], evidence["target_indices"], evidence["target_keys"]):
            if (identity not in current or current[identity].target_index != target
                    or current[identity].input.candidates[target].key != key):
                raise ValueError("Learning evidence no longer names the admitted targets")


def checkpoint_disk_admission(checkpoints: Path, previous: Path | None) -> None:
    previous_bytes = (sum(path.stat().st_size for path in previous.rglob("*") if path.is_file())
                      if previous is not None else 0)
    required = max(3 * 1024 ** 3, math.ceil(previous_bytes * 1.25) + 512 * 1024 ** 2)
    if shutil.disk_usage(checkpoints).free < required:
        raise OSError(f"Checkpoint publication requires {required} free bytes while preserving latest")


def retain_checkpoints(checkpoints: Path, *, current: Path, previous: Path | None) -> None:
    """Keep the new and previous published boundaries, only after latest is durable."""
    keep = {current.resolve(), previous.resolve() if previous is not None else None}
    for path in checkpoints.iterdir():
        if path.is_dir() and not path.is_symlink() and path.resolve() not in keep:
            if ((path.name.startswith("generation-") and path.name.removeprefix("generation-").isdigit())
                    or path.name.startswith(".pending-")):
                shutil.rmtree(path)
    sync_directory(checkpoints)



def save_checkpoint(trainer: UpdateTrainer, run_dir: Path, *, ledger_id: str | None = None) -> Path:
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest = checkpoints / "latest"
    previous = latest.resolve(strict=True) if latest.exists() else None
    if latest.is_symlink() or latest.exists():
        published = json.loads((latest / "state.json").read_text())
        if (published["training"]["checkpoint_generation"] != trainer.state.checkpoint_generation
                or published["spec"]["run_id"] != trainer.spec["run_id"]):
            raise ValueError("Stale in-memory state cannot fork a published checkpoint")
    elif trainer.state.checkpoint_generation != 0:
        raise ValueError("Committed generation has lost its authoritative latest pointer")
    occupied = [int(path.name.removeprefix("generation-")) for path in checkpoints.glob("generation-*")
                if path.name.removeprefix("generation-").isdigit()]
    generation = max([trainer.state.checkpoint_generation, *occupied]) + 1
    checkpoint_disk_admission(checkpoints, previous)
    destination = checkpoints / f"generation-{generation:08d}"
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=checkpoints))
    rng = capture_rng()
    trainer.model._write(temporary / "inference", {"training_status": trainer.state.status,
                                                  "run_id": trainer.spec["run_id"]})
    committed = dataclasses.replace(trainer.state, checkpoint_generation=generation)
    atomic_json(temporary / "state.json", {"training": dataclasses.asdict(committed),
                                           "spec": trainer.spec, "orders": trainer.orders,
                                           "orders_id": digest_json(trainer.orders),
                                           "backend": backend_identity(trainer.model.device.type),
                                           "ledger_id": ledger_id})
    torch.save({"optimizer": trainer.optimizer.state_dict(), "scheduler": trainer.scheduler.state_dict(),
                "rng": rng}, temporary / "trusted-local-state.pt")
    seal(temporary)
    os.rename(temporary, destination)
    sync_directory(checkpoints)
    link = checkpoints / ".latest-pending"
    link.unlink(missing_ok=True)
    link.symlink_to(destination.name, target_is_directory=True)
    os.replace(link, checkpoints / "latest")
    sync_directory(checkpoints)
    trainer.state.checkpoint_generation = generation
    restore_rng(rng)
    retain_checkpoints(checkpoints, current=destination, previous=previous)
    return destination


def load_checkpoint(checkpoint: Path, *, root: Path, examples: dict[str, TrainingExample],
                    device: str, ledger_id: str | None = None, require_latest: bool = True,
                    media_root: Path | None = None) -> UpdateTrainer:
    checkpoint = checkpoint.resolve(strict=True)
    verify_seal(checkpoint)
    if require_latest and (checkpoint.parent / "latest").resolve() != checkpoint:
        raise ValueError("Stale checkpoint cannot fork an existing run")
    payload = json.loads((checkpoint / "state.json").read_text())
    if payload["ledger_id"] != ledger_id:
        raise ValueError("Budget ledger identity differs from checkpoint")
    if payload["backend"] != backend_identity(device):
        raise ValueError("Exact resume requires the same backend and determinism settings")
    if payload["orders_id"] != digest_json(payload["orders"]):
        raise ValueError("Saved schedule identity mismatch")
    schedule = training_schedule(bound_schedule(payload["spec"]))
    saved_phase = payload.get("training", {}).get("phase", schedule.phases[0])
    if (saved_phase not in schedule.phases
            or any(phase not in payload["orders"] for phase in schedule.phases)
            or any(payload["orders"].get(phase) for phase in OPTIMIZER_PHASES
                   if phase not in schedule.phases)):
        raise ValueError("Checkpoint orders and state contradict the bound schedule")
    if any(identity not in examples for phase in schedule.phases for identity in payload["orders"][phase]):
        raise ValueError("Resume schedule references absent examples")
    saved_commitment = payload["spec"].get("qualified_inputs")
    if saved_commitment is not None and saved_commitment != qualified_inputs(root):
        raise ValueError("Checkpoint media, qualification, or processor commitment is stale")
    model = DecisionModel.load(checkpoint / "inference", device=device, root=root, media_root=media_root)
    expected_processor = payload["spec"].get("processor_id")
    if expected_processor is not None and model.compiler.processor_id != expected_processor:
        raise ValueError("Checkpoint processor differs from its saved training specification")
    state = TrainingState(**payload["training"])
    trainer = UpdateTrainer(model, examples, payload["orders"], payload["spec"], state=state)
    recovery = torch.load(checkpoint / "trusted-local-state.pt", map_location="cpu", weights_only=False)
    trainer.optimizer.load_state_dict(recovery["optimizer"])
    trainer.scheduler.load_state_dict(recovery["scheduler"])
    restore_rng(recovery["rng"])
    return trainer


def code_identity(root: Path) -> dict:
    paths = (list((root / "src/mdtrain").glob("*.py")) + list((root / "src/mdprep").glob("*.py"))
             + [root / "pyproject.toml", root / "uv.lock"])
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def validate_preflight(root: Path, path: Path, recipe: str, ledger: BudgetLedger,
                       schedule: str = DEFAULT_SCHEDULE) -> dict:
    receipt = json.loads(path.read_text())
    training_schedule(schedule)
    if bound_schedule(receipt) != schedule:
        raise ValueError("Preflight schedule does not match the requested training schedule")
    commitment = qualified_inputs(root)
    if (receipt.get("status") != "VERIFIED" or receipt.get("device") != "cuda"
            or receipt.get("recipe") != recipe or receipt.get("ledger_id") != ledger.record["id"]
            or receipt.get("code_hashes") != code_identity(root)
            or receipt.get("cuda_device") != cuda_device_identity()):
        raise ValueError("A current successful CUDA preflight for this recipe/ledger/code/device is required")
    if any(receipt.get(key) != value for key, value in commitment.items()):
        raise ValueError("Preflight media, qualification, or processor commitment is stale")
    return receipt


def run_loop(trainer: UpdateTrainer, run_dir: Path, ledger: BudgetLedger, *,
             group_seconds: float | dict[str, float]) -> dict:
    interrupted = False

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    old = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    latest = run_dir / "checkpoints/latest"
    last_save = time.monotonic()
    try:
        while trainer.state.status != "completed":
            if interrupted:
                trainer.state.status = "interrupted"
                break
            if trainer.state.cursor == len(trainer.orders[trainer.state.phase]):
                trainer.state.elapsed_allowance = ledger.elapsed()
                if trainer.state.phase != trainer.schedule.phases[-1]:
                    save_checkpoint(trainer, run_dir, ledger_id=ledger.record["id"])
                trainer.transition()
                latest = save_checkpoint(trainer, run_dir, ledger_id=ledger.record["id"])
                last_save = time.monotonic()
                continue
            update_seconds = (group_seconds[trainer.state.phase]
                              if isinstance(group_seconds, dict) else group_seconds)
            try:
                if trainer.spec["stage"] == "main":
                    trainer.state.main_elapsed = ledger.elapsed() - trainer.spec["main_started_elapsed"]
                ledger.admit(update_seconds)
                if trainer.spec["stage"] == "main" and trainer.state.main_elapsed + update_seconds >= MAIN_STAGE_SECONDS:
                    raise BudgetStopped("Five-hour main-stage allowance reached")
            except BudgetStopped:
                trainer.state.status = "budget_stopped"
                break
            update = trainer.step()
            if isinstance(group_seconds, dict):
                group_seconds[trainer.state.phase] = max(update_seconds, update["seconds"] * 1.25)
            else:
                group_seconds = max(update_seconds, update["seconds"] * 1.25)
            trainer.state.elapsed_allowance = ledger.elapsed()
            ledger.commit()
            atomic_json(run_dir / "update.json", {**update, "state": dataclasses.asdict(trainer.state)})
            if trainer.state.global_updates % 100 == 0 or time.monotonic() - last_save >= 300:
                save_start = time.monotonic()
                latest = save_checkpoint(trainer, run_dir, ledger_id=ledger.record["id"])
                ledger.observe_save(time.monotonic() - save_start)
                last_save = time.monotonic()
        trainer.state.elapsed_allowance = ledger.elapsed()
        if trainer.spec["stage"] == "main":
            trainer.state.main_elapsed = ledger.elapsed() - trainer.spec["main_started_elapsed"]
        save_start = time.monotonic()
        latest = save_checkpoint(trainer, run_dir, ledger_id=ledger.record["id"])
        ledger.observe_save(time.monotonic() - save_start)
        result = {"status": trainer.state.status, "stop_reason": trainer.state.status,
                  "checkpoint": str(latest), "artifact": str(latest / "inference"),
                  "updates": trainer.state.global_updates, "consumed_examples": trainer.state.consumed_examples,
                  "elapsed_allowance": ledger.elapsed(), "ledger_id": ledger.record["id"],
                  "timer_is_billing_guarantee": False}
        atomic_json(run_dir / "result.json", result)
        return result
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def record_run_receipt(trainer: UpdateTrainer, result: dict, run_dir: Path, ledger: BudgetLedger) -> dict:
    evidence = None
    if trainer.spec["stage"] != "main" and result["status"] == "completed":
        try:
            baseline = trainer.state.learning_probe["baseline"]
            final = trainer.state.learning_probe.get("final")
            if final is None:
                examples = tuple(trainer.examples[identity] for identity in baseline["ids"])
                final = fixed_probe(trainer.model, examples, ledger=ledger,
                                    question_seconds=trainer.spec["probe_question_seconds"])
                evidence = learning_evidence(baseline, final)
                trainer.state.learning_probe["final"] = final
                trainer.state.learning_probe["evidence"] = evidence
                trainer.state.elapsed_allowance = ledger.elapsed()
                save_started = time.monotonic()
                latest = save_checkpoint(trainer, run_dir, ledger_id=ledger.record["id"])
                ledger.observe_save(time.monotonic() - save_started)
                result.update(checkpoint=str(latest), artifact=str(latest / "inference"))
            else:
                evidence = learning_evidence(baseline, final)
                if evidence != trainer.state.learning_probe.get("evidence"):
                    raise ValueError("Saved learning evidence contradicts its fixed probes")
        except BudgetStopped as error:
            result.update(status="budget_stopped", reason=str(error))
    accepted = (result["status"] == "completed"
                and (trainer.spec["stage"] == "main" or
                     (evidence is not None and evidence["reduction"] > trainer.spec["gate"]["minimum_loss_reduction"])))
    result["accepted_learning_gate"] = accepted
    result["learning_evidence"] = evidence
    result["update_losses_are_diagnostic_only"] = True
    result["elapsed_allowance"] = ledger.elapsed()
    key = stage_receipt_key(trainer.spec["recipe"], bound_schedule(trainer.spec), trainer.spec["stage"])
    value = {"accepted": accepted, "dataset_id": trainer.spec["dataset_id"], "result": result,
             "qualified_inputs": trainer.spec["qualified_inputs"],
             "learning_evidence": evidence, "criterion": trainer.spec["gate"],
             "run_id": trainer.spec["run_id"],
             "checkpoint_generation": trainer.state.checkpoint_generation}
    previous = ledger.record["receipts"].get(key)
    if result.get("receipt_reconciled") and previous is not None:
        if any(previous.get(field) != content for field,content in value.items() if field != "result"):
            raise ValueError("Conflicting completed-stage receipt cannot be overwritten")
        result = previous["result"]
    else:
        ledger.receipt(key, value)
    atomic_json(run_dir / "result.json", result)
    return result


def train(root: Path, *, recipe: str, stage: str, run_dir: Path, preflight: Path,
          budget_ledger: Path, schedule: str = DEFAULT_SCHEDULE) -> dict:
    deterministic()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("train requires an empty run directory; use resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    with writer_lock(run_dir / ".writer.lock"), BudgetLedger(budget_ledger) as ledger:
        receipt = validate_preflight(root, preflight, recipe, ledger, schedule)
        view = DatasetView.open(root, split="train", recipe=recipe)
        contract = validate_contract(root)
        commitment = qualified_inputs(root, processor=contract)
        if stage == "main":
            require_learning_receipts(ledger, recipe, view, commitment, schedule)
        lengths = {}
        with (root / "data" / contract["checks_path"]).open() as stream:
            for line in stream:
                row = json.loads(line)
                lengths[row["id"]] = row["positions"]
        orders = plan_orders(view, stage, lengths, schedule)
        probe_seconds = max(1., receipt["group_seconds"] / 16)
        probe_ids = orders["selection"][:64] if stage != "main" else []
        projected = (projected_training_seconds(orders, receipt, schedule)
                     + 2 * len(probe_ids) * probe_seconds)
        horizon = admissible_seconds(stage, projected)
        try:
            ledger.admit(horizon)
        except BudgetStopped as error:
            result = {"status": "budget_stopped", "reason": str(error), "checkpoint": None,
                      "artifact": None, "updates": 0, "consumed_examples": 0,
                      "accepted_learning_gate": False, "projected_seconds": projected,
                      "admitted_seconds": horizon,
                      "ledger_id": ledger.record["id"], "timer_is_billing_guarantee": False}
            atomic_json(run_dir / "result.json", result)
            return result
        if stage == "main":
            ledger.set_process_deadline(ledger.elapsed() + MAIN_STAGE_SECONDS)
        run_identity = {"path": str(run_dir.resolve()), "dataset": view.identity,
                        "recipe": recipe, "stage": stage, **schedule_fields(schedule)}
        spec = {"run_id": digest_json(run_identity),
                "recipe": recipe, "stage": stage, "dataset_id": view.identity,
                **schedule_fields(schedule),
                **commitment, "qualified_inputs": commitment, "code_hashes": code_identity(root),
                "probe_question_seconds": probe_seconds,
                "seed": 1729, "accumulation": 16, "microbatch": 1,
                "preflight_sha256": sha256_file(preflight), "ledger_id": ledger.record["id"],
                "main_started_elapsed": ledger.elapsed(),
                "gate": {"criterion": "completed_same_questions_eval_mode_nll_reduction",
                         "minimum_loss_reduction": 0.0}}
        model = DecisionModel.create(root, NativeCompiler.load(root / "data/processor-contract.json"), device="cuda")
        trainer = UpdateTrainer(model, view.by_id(), orders, spec)
        if probe_ids:
            trainer.state.learning_probe["baseline"] = fixed_probe(
                model, tuple(trainer.examples[identity] for identity in probe_ids),
                ledger=ledger, question_seconds=probe_seconds)
        if model.compiler.processor_id != commitment["processor_id"]:
            raise ValueError("Current compiler differs from qualified training inputs")
        trainer.state.elapsed_allowance = ledger.elapsed()
        save_checkpoint(trainer, run_dir, ledger_id=ledger.record["id"])
        timing = receipt.get("phase_seconds", receipt["group_seconds"])
        result = run_loop(trainer, run_dir, ledger, group_seconds=timing)
        return record_run_receipt(trainer, result, run_dir, ledger)


def resume(root: Path, *, checkpoint: Path, preflight: Path, budget_ledger: Path,
           schedule: str = DEFAULT_SCHEDULE) -> dict:
    deterministic()
    resolved = checkpoint.resolve(strict=True)
    payload = json.loads((resolved / "state.json").read_text())
    run_dir = resolved.parent.parent
    spec = payload["spec"]
    if bound_schedule(spec) != schedule:
        raise ValueError("Requested resume schedule differs from the checkpoint")
    with writer_lock(run_dir / ".writer.lock"), BudgetLedger(budget_ledger) as ledger:
        if spec["stage"] == "main":
            ledger.set_process_deadline(spec["main_started_elapsed"] + MAIN_STAGE_SECONDS)
        receipt = validate_preflight(root, preflight, spec["recipe"], ledger, schedule)
        view = DatasetView.open(root, split="train", recipe=spec["recipe"])
        commitment = qualified_inputs(root)
        if (view.identity != spec["dataset_id"] or code_identity(root) != spec["code_hashes"]
                or commitment != spec.get("qualified_inputs")
                or any(receipt.get(key) != spec.get(key) for key in commitment)):
            raise ValueError("Dataset media, qualification, processor, or executable code changed since checkpoint")
        trainer = load_checkpoint(resolved, root=root, examples=view.by_id(), device="cuda",
                                  ledger_id=ledger.record["id"])
        if trainer.state.status == "completed":
            if spec["stage"] == "main":
                raise ValueError("Completed main runs cannot be resumed")
            if trainer.state.learning_probe.get("evidence") is not None:
                result = {"status": "completed", "stop_reason": "completed",
                          "checkpoint": str(resolved), "artifact": str(resolved / "inference"),
                          "updates": trainer.state.global_updates,
                          "consumed_examples": trainer.state.consumed_examples,
                          "elapsed_allowance": ledger.elapsed(), "ledger_id": ledger.record["id"],
                          "timer_is_billing_guarantee": False, "receipt_reconciled": True}
                return record_run_receipt(trainer, result, run_dir, ledger)
        else:
            trainer.state.status = "running"
        timing = receipt.get("phase_seconds", receipt["group_seconds"])
        result = run_loop(trainer, run_dir, ledger, group_seconds=timing)
        return record_run_receipt(trainer, result, run_dir, ledger)
