"""Separate native-tiny, bounded pretrained-CPU, and actual-CUDA proof surfaces."""

import copy
import dataclasses
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch
from torch.nn import functional as F

from mdprep.native import MAX_POSITIONS, NativeCompiler, cached_snapshot, validate_contract
from mdprep.records import TrainingExample
from mdtrain.budget import BudgetLedger, BudgetStopped
from mdtrain.data import DEFAULT_SCHEDULE, DatasetView, qualified_inputs, training_schedule
from mdtrain.engine import DecisionModel
from mdtrain.fixtures import (compare_states, make_fixtures, parameter_digest,
                              trainable_snapshot, write_requests)
from mdtrain.storage import atomic_json
from mdtrain.train import (UpdateTrainer, code_identity, cuda_device_identity, deterministic,
                           load_checkpoint, save_checkpoint, schedule_fields, stage_receipt_key)

RSS_CEILING = 11 * 1024 ** 3


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu_state(item) for item in value]
    return copy.deepcopy(value)

def replay_snapshot(trainer: UpdateTrainer, update: dict) -> dict:
    return {"ids": update["ids"], "loss": update["loss"],
            "weights": trainable_snapshot(trainer.model),
            "optimizer": cpu_state(trainer.optimizer.state_dict()),
            "scheduler": cpu_state(trainer.scheduler.state_dict()),
            "training": dataclasses.asdict(trainer.state)}


def assert_exact_replay(expected: dict, actual: dict) -> None:
    """Same-backend training recovery permits no inference-style tolerance."""
    compare_states(expected, actual, exact=True)



def bounded_child(command: list[str], *, directory: Path, rss_limit: int = RSS_CEILING,
                  timeout: float = 900) -> dict:
    start, peak = time.monotonic(), 0
    log_path = directory / f"child-{time.time_ns()}.log"
    with log_path.open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        process = psutil.Process(child.pid)
        reason = None
        while child.poll() is None:
            try:
                rss = sum(item.memory_info().rss for item in [process, *process.children(recursive=True)]
                          if item.is_running())
                peak = max(peak, rss)
                if rss > rss_limit:
                    reason = f"Child RSS exceeded explicit {rss_limit}-byte ceiling"
                elif time.monotonic() - start > timeout:
                    reason = f"Child exceeded {timeout}-second watchdog"
                if reason:
                    for descendant in process.children(recursive=True):
                        descendant.kill()
                    child.kill()
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.1)
        returncode = child.wait()
    return {"returncode": returncode, "reason": reason,
            "peak_rss_bytes": peak, "seconds": time.monotonic() - start, "log": str(log_path)}


def reload_command(root: Path, artifact: Path, inputs: Path, media_root: Path,
                   output: Path, device: str) -> list[str]:
    return [sys.executable, "-m", "mdtrain._reload", "--root", str(root), "--artifact", str(artifact),
            "--input", str(inputs), "--media-root", str(media_root), "--output", str(output),
            "--device", device]


def assert_predictions(expected: list[dict], actual: list[dict], *, tolerance: float) -> None:
    if [row["id"] for row in expected] != [row["id"] for row in actual]:
        raise AssertionError("Fresh process changed prediction identity/order")
    for left, right in zip(expected, actual):
        if tuple(left["probabilities"]) != tuple(right["probabilities"]):
            raise AssertionError("Fresh process changed opaque candidate routing")
        for key, probability in left["probabilities"].items():
            if not math.isclose(probability, right["probabilities"][key], abs_tol=tolerance, rel_tol=tolerance):
                raise AssertionError("Fresh process changed paired probabilities")


def probe_loss(model: DecisionModel, examples: tuple[TrainingExample, ...]) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for example in examples:
            logits = model(model.compiler.compile((example.input,)))
            losses.append(float(F.cross_entropy(logits, torch.tensor([example.target_index], device=model.device))))
    return sum(losses) / len(losses)


def smoke_tiny(root: Path, output: Path, schedule: str = DEFAULT_SCHEDULE) -> dict:
    deterministic()
    selected_schedule = training_schedule(schedule)
    directory = output.with_suffix(".files")
    if directory.exists():
        raise FileExistsError(f"Smoke evidence directory must be new: {directory}")
    examples = make_fixtures(directory)
    compiler = NativeCompiler.from_pretrained(root, media_root=directory)
    model = DecisionModel.create(root, compiler, device="cpu", kind="tiny")
    inputs = tuple(example.input for example in examples)
    combined = compiler.compile(inputs)
    model.eval()
    with torch.no_grad():
        batched = model(combined)
        if not torch.isneginf(batched[~combined.candidate_mask]).all():
            raise AssertionError("Padded candidate slots were not masked to negative infinity")
        for index, item in enumerate(inputs):
            single = model(compiler.compile((item,)))
            torch.testing.assert_close(single[0], batched[index, :len(item.candidates)], rtol=2e-5, atol=2e-5)
    frozen_before = parameter_digest(model, frozen_only=True)
    initial = trainable_snapshot(model)
    initial_loss = probe_loss(model, examples)
    ids = [example.input.id for example in examples]
    components = {"warmup": ids * 4, "training": ids * 12}
    phase_orders = selected_schedule.compose(components)
    orders = {phase: phase_orders.get(phase, []) for phase in ("head", "joint")}
    trainer = UpdateTrainer(model, {e.input.id: e for e in examples}, orders,
                            {"run_id": "native-tiny-structural", "stage": "smoke",
                             **schedule_fields(schedule)})
    head = trainer.step()
    after_head = trainable_snapshot(model)
    if not any(not torch.equal(initial[name], after_head[name]) for name in initial if name.startswith("pointer.")):
        raise AssertionError("Head-only optimizer did not change the pointer")
    if any(not torch.equal(initial[name], after_head[name]) for name in initial if not name.startswith("pointer.")):
        raise AssertionError("Head-only phase changed an adapter")
    joint = None
    if len(selected_schedule.phases) > 1:
        trainer.transition()
        joint = trainer.step()
        after_joint = trainable_snapshot(model)
        if not any(not torch.equal(after_head[name], after_joint[name]) for name in initial if ".lora_B." in name):
            raise AssertionError("Joint optimizer did not change a language adapter")
    if parameter_digest(model, frozen_only=True) != frozen_before:
        raise AssertionError("An original native body weight changed")
    checkpoint = save_checkpoint(trainer, directory / "run")
    baseline_update = trainer.step()
    expected = replay_snapshot(trainer, baseline_update)
    resumed = load_checkpoint(checkpoint, root=root, examples={e.input.id: e for e in examples},
                              device="cpu", media_root=directory)
    replay_update = resumed.step()
    assert_exact_replay(expected, replay_snapshot(resumed, replay_update))
    completed_updates = [head, replay_update]
    completed_updates.append(resumed.step())
    while resumed.state.status != "completed":
        if resumed.state.cursor == len(resumed.orders[resumed.state.phase]):
            resumed.transition()
        else:
            completed_updates.append(resumed.step())
    final_loss = probe_loss(resumed.model, examples)
    if not final_loss < initial_loss:
        raise AssertionError("Tiny fixed-fixture training loss did not decrease")
    if parameter_digest(resumed.model, frozen_only=True) != frozen_before:
        raise AssertionError("Resumed training changed an original body weight")
    artifact = directory / "tiny-pair"
    resumed.model.save(artifact, metadata={"scope": "random native tiny structural proof"})
    predictions = [resumed.model.predict((item,))[0] for item in inputs]
    reload_output = directory / "fresh-reload.json"
    child = bounded_child(reload_command(root, artifact, directory / "requests.jsonl", directory,
                                          reload_output, "cpu"), directory=directory)
    if child["returncode"] != 0:
        raise RuntimeError(f"Fresh native tiny pair failed: {child}")
    reloaded = json.loads(reload_output.read_text())
    assert_predictions(predictions, reloaded["predictions"], tolerance=0.)
    result = {"status": "VERIFIED", "model_kind": "tiny", "schedule": schedule,
              "pretrained_verified": False, "cuda_verified": False, "device": "cpu",
              "dtype": "float32", "input_kinds": list(("text", "image", "mixed", "video")),
              "native_fields": sorted(combined.native_inputs), "padding_agreement": True,
              "frozen_weights_unchanged": True, "head_only_update": head, "joint_update": joint,
              "updates": completed_updates, "training_status": resumed.state.status,
              "global_updates": resumed.state.global_updates,
              "pointer_gradients_nonzero": all(update["head_gradient_nonzero"] for update in completed_updates),
              "adapter_gradients_absent": all(not update["adapter_gradient_nonzero"]
                                              for update in completed_updates)
              if schedule == "pointer-only" else None,
              "exact_same_next_update": True, "fresh_process_paired_reload": True,
              "initial_loss": initial_loss, "final_loss": final_loss,
              "processor_id": compiler.processor_id, "artifact": str(artifact), "child": child}
    atomic_json(output, result)
    return result


def smoke_real(root: Path, output: Path, schedule: str = DEFAULT_SCHEDULE) -> dict:
    training_schedule(schedule)
    directory = output.with_suffix(".files")
    if directory.exists():
        raise FileExistsError(f"Smoke evidence directory must be new: {directory}")
    directory.mkdir(parents=True)
    result = {"status": "NOT_VERIFIED", "model_kind": "real", "schedule": schedule,
              "device": "cpu", "dtype": "bfloat16", "cuda_verified": False,
              "backward_verified": False, "rss_ceiling_bytes": RSS_CEILING,
              "tiny_substitution": False, "implicit_downloads": False}
    try:
        snapshot = cached_snapshot(root)
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        shard_paths = [snapshot / name for name in set(index["weight_map"].values())]
        if not all(path.is_file() for path in shard_paths):
            raise FileNotFoundError("Pinned real shards are unavailable locally")
        weight_bytes = int(index["metadata"]["total_size"])
        estimated_peak = weight_bytes + 1536 * 1024 ** 2
        available = psutil.virtual_memory().available
        result["memory_admission"] = {"weight_bytes": weight_bytes, "estimated_peak_bytes": estimated_peak,
                                      "available_bytes": available, "system_reserve_bytes": 2 * 1024 ** 3}
        if estimated_peak > RSS_CEILING or estimated_peak + 2 * 1024 ** 3 > available:
            raise MemoryError("Explicit BF16 CPU memory admission refused the cached real body")
        make_fixtures(directory)
        child = bounded_child([sys.executable, "-m", "mdtrain._real_smoke", "--root", str(root),
                               "--directory", str(directory)], directory=directory)
        result["inference_child"] = child
        if child["returncode"] != 0:
            raise RuntimeError(f"Real BF16 child did not complete; see {child['log']}")
        actual = json.loads((directory / "real-child.json").read_text())
        if actual["model_kind"] != "real" or actual["dtype"] != "bfloat16":
            raise ValueError("Child did not execute the intended real BF16 model")
        fresh = directory / "real-fresh.json"
        reload_child = bounded_child(reload_command(root, directory / "real-pair", directory / "requests.jsonl",
                                                    directory, fresh, "cpu"), directory=directory)
        result["reload_child"] = reload_child
        if reload_child["returncode"] != 0:
            raise RuntimeError(f"Real paired reload did not complete; see {reload_child['log']}")
        assert_predictions(actual["predictions"], json.loads(fresh.read_text())["predictions"], tolerance=0.)
        result.update(status="VERIFIED", actual=actual, fresh_process_paired_reload=True)
    except (OSError, ValueError, RuntimeError, MemoryError, KeyError, AssertionError, subprocess.SubprocessError) as error:
        result["reason"] = str(error)
    atomic_json(output, result)
    return result


def boundary_request(compiler: NativeCompiler, example: TrainingExample) -> TrainingExample:
    lower, upper = 0, MAX_POSITIONS * 2
    best = example.input
    while lower <= upper:
        count = (lower + upper) // 2
        request = dataclasses.replace(example.input, id=example.input.id + "-boundary",
                                      state=example.input.state + "\n" + " neutral" * count)
        try:
            compiler.compile((request,))
            best, lower = request, count + 1
        except ValueError as error:
            if "Complete input exceeds" not in str(error):
                raise
            upper = count - 1
    length = int(compiler.compile((best,)).expanded_lengths[0])
    if length < MAX_POSITIONS - 1:
        raise ValueError(f"Could not form a complete near-boundary native input: {length}")
    return dataclasses.replace(example, input=best)


def preflight_cuda(root: Path, *, recipe: str, budget_ledger: Path,
                   paid_started_at: str, output: Path, schedule: str = DEFAULT_SCHEDULE) -> dict:
    deterministic()
    selected_schedule = training_schedule(schedule)
    directory = output.with_suffix(".files")
    if directory.exists():
        raise FileExistsError("Preflight evidence directory must be new")
    directory.mkdir(parents=True)
    result = {"status": "NOT_VERIFIED", "device": "cuda", "recipe": recipe,
              "schedule": schedule, "cuda_verified": False, "timer_is_billing_guarantee": False}
    with BudgetLedger(budget_ledger, paid_started_at=paid_started_at) as ledger:
        result["ledger_id"] = ledger.record["id"]
        try:
            ledger.admit(900)
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise ValueError("Actual CUDA with native BF16 support is required")
            properties = torch.cuda.get_device_properties(0)
            if (not any(model in properties.name for model in ("RTX 4090", "RTX 3090"))
                    or properties.total_memory < 23 * 1024 ** 3):
                raise ValueError("The selected preflight targets an actual 24GB RTX 4090 or RTX 3090")
            if psutil.virtual_memory().total < 28 * 1024 ** 3:
                raise MemoryError("CUDA preflight requires at least 28GiB host RAM")
            free_disk = psutil.disk_usage(directory).free
            if free_disk < 3 * 1024 ** 3:
                raise OSError("At least 3GiB persistent free disk is required for checkpoint/reload proof")
            binaries = {name: subprocess.run([name, "-version"], check=True, capture_output=True,
                                             text=True).stdout.splitlines()[0] for name in ("ffmpeg", "ffprobe")}
            contract = validate_contract(root)
            commitment = qualified_inputs(root, processor=contract)
            DatasetView.open(root, split="train", recipe=recipe)
            view = DatasetView.open(root, split="train", recipe="mixed")
            by_id = view.by_id()
            maxima = {}
            with (root / "data" / contract["checks_path"]).open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["id"] in by_id:
                        previous = maxima.get(row["modality"])
                        if previous is None or row["positions"] > previous["positions"]:
                            maxima[row["modality"]] = row
            kinds = ("text", "image", "mixed", "video")
            if set(maxima) != set(kinds):
                raise ValueError("Current training corpus must supply all four input kinds for preflight")
            compiler = NativeCompiler.load(root / "data/processor-contract.json")
            actual = tuple(by_id[maxima[kind]["id"]] for kind in kinds)
            boundary = tuple(boundary_request(compiler, example) for example in actual)
            cases = actual + boundary
            model = DecisionModel.create(root, compiler, device="cuda", kind="real")
            torch.cuda.reset_peak_memory_stats()
            frozen = parameter_digest(model, frozen_only=True)
            initial = trainable_snapshot(model)
            ids = [example.input.id for example in cases]
            components = {"warmup": ids * 2, "training": ids * 4}
            phase_orders = selected_schedule.compose(components)
            orders = {phase: phase_orders.get(phase, []) for phase in ("head", "joint")}
            trainer = UpdateTrainer(model, {e.input.id: e for e in cases}, orders,
                                    {"run_id": "cuda-preflight", "stage": "preflight",
                                     **schedule_fields(schedule),
                                     **commitment, "qualified_inputs": commitment})
            head = trainer.step()
            after_head = trainable_snapshot(model)
            if not any(not torch.equal(initial[key], after_head[key]) for key in initial if key.startswith("pointer.")):
                raise AssertionError("CUDA head-only update did not change pointer weights")
            if any(not torch.equal(initial[key], after_head[key]) for key in initial if not key.startswith("pointer.")):
                raise AssertionError("CUDA head-only update changed adapters")
            joint = None
            if len(selected_schedule.phases) > 1:
                trainer.transition()
                ledger.admit(max(900., head["seconds"] * 2))
                joint = trainer.step()
                after_joint = trainable_snapshot(model)
                if not any(not torch.equal(after_head[key], after_joint[key])
                           for key in initial if ".lora_B." in key):
                    raise AssertionError("CUDA joint update did not change language adapters")
            if parameter_digest(model, frozen_only=True) != frozen:
                raise AssertionError("CUDA update changed frozen original weights")
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            if properties.total_memory - peak_reserved < 1024 ** 3:
                raise MemoryError("Measured CUDA preflight leaves less than 1GiB reserved-memory headroom")
            save_started = time.monotonic()
            trainer.state.elapsed_allowance = ledger.elapsed()
            checkpoint = save_checkpoint(trainer, directory / "run", ledger_id=ledger.record["id"])
            save_seconds = time.monotonic() - save_started
            ledger.observe_save(save_seconds)
            replay_phase = trainer.state.phase
            replay_basis = joint if joint is not None else head
            ledger.admit(replay_basis["seconds"] * 3)
            reference = trainer.step()
            expected = replay_snapshot(trainer, reference)
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()
            recovered = load_checkpoint(checkpoint, root=root, examples={e.input.id: e for e in cases},
                                        device="cuda", ledger_id=ledger.record["id"])
            ledger.admit(replay_basis["seconds"] * 1.5)
            replay = recovered.step()
            assert_exact_replay(expected, replay_snapshot(recovered, replay))
            if parameter_digest(recovered.model, frozen_only=True) != frozen:
                raise AssertionError("CUDA recovery changed a frozen original weight")
            artifact = directory / "cuda-pair"
            recovered.model.save(artifact, metadata={"scope": "actual CUDA preflight"})
            inputs = tuple(example.input for example in actual)
            predictions = [recovered.model.predict((item,))[0] for item in inputs]
            write_requests(directory / "requests.jsonl", inputs)
            trainables = recovered.model.audit_trainables()
            del recovered
            gc.collect()
            torch.cuda.empty_cache()
            fresh = directory / "fresh-cuda.json"
            child = bounded_child(reload_command(root, artifact, directory / "requests.jsonl", root / "data",
                                                 fresh, "cuda"), directory=directory,
                                  rss_limit=psutil.virtual_memory().total - 4 * 1024 ** 3)
            if child["returncode"] != 0:
                raise RuntimeError(f"Fresh CUDA paired reload failed: {child}")
            assert_predictions(predictions, json.loads(fresh.read_text())["predictions"], tolerance=1e-5)
            phase_seconds = {"head": head["seconds"] * 1.5}
            if joint is not None:
                phase_seconds["joint"] = max(joint["seconds"], replay["seconds"]) * 1.5
            else:
                phase_seconds[replay_phase] = max(head["seconds"], replay["seconds"]) * 1.5
            group_seconds = max(phase_seconds.values())
            result.update(status="VERIFIED", cuda_verified=True, gpu=properties.name,
                          total_vram=properties.total_memory, peak_allocated=peak_allocated,
                          peak_reserved=peak_reserved, host_rss_observed_bytes=psutil.Process().memory_info().rss,
                          host_total_ram=psutil.virtual_memory().total, free_disk=free_disk, binaries=binaries,
                          maximum_examples=maxima,
                          boundary_lengths={e.input.id: int(compiler.compile((e.input,)).expanded_lengths[0]) for e in boundary},
                          head_update=head, joint_update=joint, trainables=trainables,
                          same_next_update=True, training_replay_equality="exact",
                          fresh_process_paired_reload=True, inference_reload_tolerance=1e-5,
                          phase_seconds=phase_seconds, group_seconds=group_seconds,
                          save_seconds=save_seconds, **commitment, cuda_device=cuda_device_identity(),
                          code_hashes=code_identity(root), child=child)
            ledger.receipt(stage_receipt_key(recipe, schedule, "preflight"),
                           {"accepted": True, "output": str(output), "schedule": schedule,
                            "group_seconds": group_seconds, "phase_seconds": phase_seconds,
                            "qualified_inputs": commitment, "cuda_device": result["cuda_device"]})
        except BudgetStopped as error:
            result.update(status="budget_stopped", reason=str(error))
        except (OSError, ValueError, RuntimeError, AssertionError, MemoryError, subprocess.SubprocessError) as error:
            result["reason"] = str(error)
        result["elapsed_allowance"] = ledger.elapsed()
        atomic_json(output, result)
    return result
