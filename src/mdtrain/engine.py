"""Native Qwen3-VL hidden states, exact language LoRA, and the paired pointer."""

import contextlib
import importlib.metadata
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Sequence

import psutil
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLModel

from mdprep.common import sha256_file
from mdprep.native import MODEL_ID, REVISION, NativeCompiler, PreparedBatch, cached_snapshot, digest_json
from mdprep.records import DecisionInput
from mdtrain.metrics import probabilities
from mdtrain.storage import atomic_json, json_safe, seal, sync_directory, verify_seal

PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
POINTER_EQUATION = "two-projection-dot-256-v1"
INFERENCE_DEVICES = ("cpu", "cuda", "mps")
# The base weights live in host RAM on CPU and in unified memory on Apple GPUs.
HOST_MEMORY_DEVICES = ("cpu", "mps")


def admit_host_memory(total_bytes: int, device: str) -> None:
    needed = total_bytes + 1536 * 1024 ** 2
    available = psutil.virtual_memory().available
    if needed > 11 * 1024 ** 3 or needed + 2 * 1024 ** 3 > available:
        raise MemoryError(f"Loading the BF16 base on {device} needs {(needed + 2 * 1024 ** 3) / 1024 ** 3:.1f} GiB "
                          f"of free memory; {available / 1024 ** 3:.1f} GiB is available. Close other apps and retry.")


def environment() -> dict:
    return {name: importlib.metadata.version(name).split("+")[0] for name in
            ("transformers", "torch", "peft", "accelerate", "safetensors")}


def require_versions() -> None:
    installed = environment()
    if installed["transformers"] != "5.19.0" or installed["torch"].split("+")[0] != "2.14.1":
        raise ValueError("The qualified core stack requires Transformers 5.19.0 and torch 2.14.1")


def tiny_config() -> Qwen3VLConfig:
    return Qwen3VLConfig(text_config={
        "vocab_size": 151936, "hidden_size": 128, "intermediate_size": 256,
        "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 32, "max_position_embeddings": 4096, "use_cache": False,
        "pad_token_id": 151643, "rope_parameters": {
            "rope_type": "default", "rope_theta": 5000000.0,
            "mrope_section": [6, 5, 5], "mrope_interleaved": True}},
        vision_config={"hidden_size": 64, "intermediate_size": 128, "depth": 3,
                       "num_heads": 4, "out_hidden_size": 128, "patch_size": 16,
                       "spatial_merge_size": 2, "temporal_patch_size": 2,
                       "num_position_embeddings": 2304, "deepstack_visual_indexes": [0, 1, 2]})


def adapter_targets(body: Qwen3VLModel) -> tuple[str, ...]:
    targets = tuple(f"language_model.layers.{index}.{suffix}"
                    for index in range(body.config.text_config.num_hidden_layers) for suffix in PROJECTIONS)
    for name in targets:
        if not isinstance(body.get_submodule(name), nn.Linear):
            raise ValueError(f"Native language adapter target is not Linear: {name}")
    return targets


def load_body(root: Path, *, device: str) -> tuple[Qwen3VLModel, dict]:
    require_versions()
    snapshot = cached_snapshot(root)
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not all((snapshot / shard).is_file() for shard in shards):
        raise FileNotFoundError("Pinned base shards are not completely cached; downloads are disabled")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("The mps device needs Apple Silicon and a PyTorch build with MPS support")
    if device in HOST_MEMORY_DEVICES:
        admit_host_memory(int(index["metadata"]["total_size"]), device)
    # Loading on CPU and then copying to MPS would hold two copies in unified memory.
    body, loading = Qwen3VLModel.from_pretrained(
        str(snapshot), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True, output_loading_info=True,
        device_map={"": "mps"} if device == "mps" else None)
    report = json_safe(loading)
    unexpected = [name for name in report.get("unexpected_keys", [])
                  if name not in ("lm_head.weight",)]
    if report.get("missing_keys") or report.get("mismatched_keys") or unexpected or report.get("error_msgs"):
        raise ValueError(f"Native body weights did not load completely: {report}")
    if any(parameter.dtype == torch.float32 for parameter in body.parameters()):
        raise ValueError("The real frozen body must not remain in FP32")
    body.to(device=device)
    spec = {"kind": "real", "model_id": MODEL_ID, "model_revision": REVISION,
            "config_sha256": sha256_file(snapshot / "config.json"),
            "weight_index_sha256": sha256_file(index_path),
            "shards": {name: sha256_file(snapshot / name) for name in shards},
            "base_dtype": "bfloat16", "loading_info": report}
    return body, spec


class Pointer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.candidate = nn.Linear(hidden_size, 256, bias=False, dtype=torch.float32)
        self.decision = nn.Linear(hidden_size, 256, bias=False, dtype=torch.float32)
        nn.init.xavier_uniform_(self.candidate.weight)
        nn.init.xavier_uniform_(self.decision.weight)

    def forward(self, hidden: torch.Tensor, batch: PreparedBatch) -> torch.Tensor:
        device = hidden.device
        rows = torch.arange(len(batch.ids), device=device)
        candidates = hidden[rows[:, None], batch.candidate_end_positions.to(device)]
        decisions = hidden[rows, batch.decision_positions.to(device)]
        with torch.autocast(device_type=device.type, enabled=False):
            logits = (self.candidate(candidates.float())
                      * self.decision(decisions.float())[:, None, :]).sum(-1) / math.sqrt(256)
        return logits.masked_fill(~batch.candidate_mask.to(device), -torch.inf)


class DecisionModel(nn.Module):
    def __init__(self, body: PeftModel, compiler: NativeCompiler, spec: dict, *, device: str):
        super().__init__()
        self.body = body
        self.compiler = compiler
        self.spec = spec
        self.pointer = Pointer(body.config.text_config.hidden_size).to(device=device)
        self.temperature = 1.0
        self.temperature_fitted = False
        self.finalized = False
        self.artifact_id = "unsaved"
        self.phase = "head"
        self.adapter_names = tuple(name for name, _ in body.named_parameters() if ".lora_A." in name or ".lora_B." in name)
        expected = 2 * len(spec["adapter_targets"])
        if len(self.adapter_names) != expected:
            raise ValueError("Unexpected adapter tensor count")
        config = body.peft_config["default"]
        if (config.r != 16 or config.lora_alpha != 32 or config.lora_dropout != .05
                or config.bias != "none" or config.use_rslora or config.use_dora
                or config.modules_to_save):
            raise ValueError("Paired adapter does not implement the exact selected LoRA policy")
        for target in spec["adapter_targets"]:
            module = body.get_base_model().get_submodule(target)
            output_width, input_width = spec["adapter_shapes"][target]
            if (list(module.lora_A["default"].weight.shape) != [16, input_width]
                    or list(module.lora_B["default"].weight.shape) != [output_width, 16]
                    or module.scaling["default"] != 2.):
                raise ValueError(f"Resolved native adapter shape/scaling mismatch: {target}")
        for name, parameter in body.named_parameters():
            if name in self.adapter_names:
                parameter.data = parameter.data.float()
        self.set_phase("head")

    @classmethod
    def create(cls, root: Path, compiler: NativeCompiler, *, device: str = "cpu", kind: str = "real"):
        require_versions()
        if device not in ("cpu", "cuda"):
            raise ValueError("Only cpu and cuda backends are supported")
        if kind == "tiny":
            config = tiny_config()
            config._attn_implementation = "sdpa"
            body = Qwen3VLModel(config).to(device=device)
            spec = {"kind": "tiny", "config": config.to_dict(), "base_dtype": "float32",
                    "model_id": MODEL_ID, "model_revision": REVISION}
        elif kind == "real":
            body, spec = load_body(root, device=device)
        else:
            raise ValueError("Unknown native model kind")
        body.requires_grad_(False)
        targets = adapter_targets(body)
        shapes = {name: list(body.get_submodule(name).weight.shape) for name in targets}
        configuration = LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, bias="none",
                                   target_modules="^(?:" + "|".join(re.escape(name) for name in targets) + ")$",
                                   init_lora_weights=True, use_rslora=False, use_dora=False)
        adapted = get_peft_model(body, configuration, autocast_adapter_dtype=True)
        spec.update(adapter_targets=list(targets), adapter_shapes=shapes, adapter_rank=16,
                    adapter_alpha=32, adapter_dropout=.05, hidden_size=body.config.text_config.hidden_size,
                    pointer_equation=POINTER_EQUATION, environment=environment())
        return cls(adapted, compiler, spec, device=device)

    @property
    def device(self) -> torch.device:
        return self.pointer.candidate.weight.device

    def set_phase(self, phase: str) -> None:
        if phase not in ("head", "joint"):
            raise ValueError("Optimizer phase must be head or joint")
        self.phase = phase
        for name, parameter in self.body.named_parameters():
            parameter.requires_grad_(phase == "joint" and name in self.adapter_names)
        self.pointer.requires_grad_(True)
        base = self.body.get_base_model()
        base.disable_input_require_grads()
        if phase == "joint":
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        else:
            base.gradient_checkpointing_disable()
            base.disable_input_require_grads()
        base.config.text_config.use_cache = False
        self.train(True)
        self.audit_trainables()

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "body"):
            if getattr(self, "phase", "head") == "head":
                self.body.eval()
            self.body.get_base_model().visual.eval()
        return self

    def audit_trainables(self) -> dict:
        expected = {"pointer.candidate.weight", "pointer.decision.weight"}
        if self.phase == "joint":
            expected.update("body." + name for name in self.adapter_names)
        actual = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        if actual != expected:
            raise ValueError(f"Trainability invariant failed: {actual ^ expected}")
        return {name: {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                       "count": parameter.numel()}
                for name, parameter in self.named_parameters() if parameter.requires_grad}

    def forward(self, batch: PreparedBatch) -> torch.Tensor:
        base_dtype = next(self.body.get_base_model().parameters()).dtype
        native = {name: tensor.to(device=self.device,
                                   dtype=base_dtype if tensor.is_floating_point() else tensor.dtype)
                  for name, tensor in batch.native_inputs.items()}
        context = torch.no_grad() if self.phase == "head" else contextlib.nullcontext()
        with context:
            output = self.body(**native, use_cache=False, return_dict=True)
        if not hasattr(output, "last_hidden_state"):
            raise ValueError("Native body did not return final hidden states")
        return self.pointer(output.last_hidden_state, batch)

    @torch.no_grad()
    def logits(self, requests: Sequence[DecisionInput]) -> list[list[float]]:
        self.eval()
        batch = self.compiler.compile(requests)
        output = self(batch).cpu()
        return [output[index, :len(keys)].tolist() for index, keys in enumerate(batch.output_keys)]

    def predict(self, requests: Sequence[DecisionInput]) -> list[dict]:
        requests = tuple(requests)
        logits = self.logits(requests)
        calibrated = probabilities(logits, self.temperature)
        results = []
        for request, values in zip(requests, calibrated):
            mapping = dict(zip((candidate.key for candidate in request.candidates), values))
            expected = (sum(float(key) * value for key, value in mapping.items())
                        if request.kind == "score" else None)
            results.append({"id": request.id, "probabilities": mapping,
                            "predicted_key": max(mapping, key=mapping.get), "expected_score": expected,
                            "artifact_id": self.artifact_id, "processor_id": self.compiler.processor_id,
                            "calibration_id": digest_json({"temperature": self.temperature}),
                            "policy": "disabled"})
        return results

    def _write(self, directory: Path, metadata: dict) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.body.save_pretrained(str(directory / "adapter"), safe_serialization=True,
                                  save_embedding_layers=False)
        save_file({name: value.detach().cpu().contiguous() for name, value in self.pointer.state_dict().items()},
                  str(directory / "pointer.safetensors"))
        if self.spec["kind"] == "tiny":
            base = self.body.get_base_model()
            base_state = {name: value.detach().cpu().contiguous() for name, value in base.state_dict().items()
                          if ".lora_A." not in name and ".lora_B." not in name}
            save_file(base_state, str(directory / "tiny-base.safetensors"))
        self.compiler.save(directory)
        atomic_json(directory / "model-spec.json", self.spec)
        atomic_json(directory / "calibration.json", {"kind": "temperature" if self.temperature_fitted else "identity",
                                                    "temperature": self.temperature})
        atomic_json(directory / "policy.json", {"kind": "disabled"})
        atomic_json(directory / "artifact.json", {"format": "native-decision-pair-v1",
                                                  "finalized": self.finalized, "metadata": metadata,
                                                  "processor_id": self.compiler.processor_id,
                                                  "model_spec_id": digest_json(self.spec)})
        seal(directory)

    def save(self, artifact: Path, *, metadata: dict | None = None) -> str:
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite paired artifact: {artifact}")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{artifact.name}.", dir=artifact.parent))
        temporary.rmdir()
        self._write(temporary, metadata or {})
        os.rename(temporary, artifact)
        sync_directory(artifact.parent)
        self.artifact_id = verify_seal(artifact)
        return self.artifact_id

    @classmethod
    def load(cls, artifact: Path, *, device: str, root: Path, media_root: Path | None = None):
        identity = verify_seal(artifact)
        record = json.loads((artifact / "artifact.json").read_text())
        spec = json.loads((artifact / "model-spec.json").read_text())
        if (record["format"] != "native-decision-pair-v1" or record["model_spec_id"] != digest_json(spec)
                or spec["pointer_equation"] != POINTER_EQUATION or spec["environment"] != environment()
                or spec["model_revision"] != REVISION or spec["model_id"] != MODEL_ID):
            raise ValueError("Incompatible paired artifact")
        compiler = NativeCompiler.from_artifact(artifact, media_root=media_root or root / "data")
        if compiler.processor_id != record["processor_id"]:
            raise ValueError("Artifact and processor pairing mismatch")
        if spec["kind"] == "tiny":
            config = Qwen3VLConfig.from_dict(spec["config"])
            config._attn_implementation = "sdpa"
            body = Qwen3VLModel(config).to(device=device)
        else:
            body, observed = load_body(root, device=device)
            for field in ("config_sha256", "weight_index_sha256", "shards"):
                if observed[field] != spec[field]:
                    raise ValueError(f"Pinned base identity mismatch: {field}")
        if list(adapter_targets(body)) != spec["adapter_targets"]:
            raise ValueError("Native language target inventory differs")
        body.requires_grad_(False)
        adapted = PeftModel.from_pretrained(body, str(artifact / "adapter"), is_trainable=True,
                                            local_files_only=True, autocast_adapter_dtype=True)
        if spec["kind"] == "tiny":
            result = adapted.get_base_model().load_state_dict(
                load_file(str(artifact / "tiny-base.safetensors"), device=device), strict=False)
            if result.unexpected_keys or set(result.missing_keys) != {
                    name for name, _ in adapted.get_base_model().named_parameters()
                    if ".lora_A." in name or ".lora_B." in name}:
                raise ValueError("Tiny native base reload mismatch")
        model = cls(adapted, compiler, spec, device=device)
        model.pointer.load_state_dict(load_file(str(artifact / "pointer.safetensors"), device=device), strict=True)
        calibration = json.loads((artifact / "calibration.json").read_text())
        model.temperature = float(calibration["temperature"])
        model.temperature_fitted = calibration["kind"] == "temperature"
        if not math.isfinite(model.temperature) or model.temperature <= 0:
            raise ValueError("Invalid calibration temperature")
        model.finalized = bool(record["finalized"])
        model.artifact_id = identity
        model.eval()
        return model
