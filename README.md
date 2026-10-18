# JEBS: a multimodal decision model

An experimental candidate-probability model for text, photographs, rendered documents, joint policy/image evidence, and four-frame videos. It does not generate answers token by token. It is not a validated generalist model or a production authorization system.

The trained v3 model is public at [experimentalmachines/JEBS-4B](https://huggingface.co/experimentalmachines/JEBS-4B), with its model card, results, and instructions to run it. This repository holds the code that prepared the data, trained, finalized, and evaluated it. The prepared corpus, the locked test splits, the base-model cache, and the run receipts are not published, so tests that read `data/` need a locally prepared corpus.

## Model contract

The base is `Qwen/Qwen3-VL-4B-Instruct` at revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`. The runtime loads `Qwen3VLModel`, including native vision and temporal processing, without a vocabulary head. Original weights remain frozen. Language-only PEFT LoRA uses rank 16, alpha 32, and dropout 0.05. The candidate and decision representations each pass through a bias-free FP32 projection to 256 dimensions. Their scaled dot products become candidate logits.

`DecisionInput` separates visible evidence and ordered candidate meanings from supervision and audit metadata. `PreparedBatch` contains native tensors, marker positions, masks, IDs, and output keys. Targets remain in `TrainingExample`, outside model-native arguments. Choice keys are opaque in the prompt. Noul preserves false/true ordering. Score preserves ordered numeric levels and reports an expected score.

Native compilation admits a complete input only when it fits 2,048 expanded positions. It never truncates visual tokens. One image or one four-frame video is supported per input. Native video timing is preserved. Ordinary causal attention does not provide candidate-order invariance.

## Data

The assembled corpus has 44,904 questions, 3,503 unique images, and 750 videos. It includes 8,270 real human-label-derived questions and 36,634 generated questions. The generated set includes 8,804 admitted hard-v1 questions and 6,100 `rule-variants-v1` counterfactual rule questions; hard-v1's upstream test partition and the separate 600-question held-out structure set remain external. A person did not author every real-source-derived question. Open Images annotation fidelity is not independent photographic ground truth.

| Split | Questions | Purpose |
|---|---:|---|
| train | 33,276 | Weight updates |
| dev | 2,814 | Development evaluation |
| calibration | 2,814 | Optional scalar temperature |
| policy | 2,000 | Reserved. Automation policy remains disabled. |
| test_id | 3,000 | Locked final same-family evaluation |
| test_ood | 1,000 | Locked declared holdout axes |

`pilot.jsonl` and `diagnostic.jsonl` remain byte-identical train-derived subsets of the original corpus, not additional data. The `real` recipe selects 5,600 training questions from BANKING77 and Open Images. The `mixed` recipe selects all 33,276 training questions. Real-only training has no generated-video exposure. Neither recipe silently substitutes for the other.

Read `DATA_CARD.md` for origins, licenses, label meanings, and limitations. Read `SUBAGENT_TRAINING_PLAN.md` for ownership and acceptance contracts.

## Verification artifacts

Successful current-hash receipts are required, not just files with plausible names.

- `data/manifest.json` records assembly counts and split isolation.
- `data/verification-contract.json` binds independent source, visible-rule, pixel/geometry, and decoded-temporal checks to rows, media, and verifier code.
- `data/processor-contract.json` binds every prepared input to the pinned native compiler, schema, processor, versions, marker positions, and expanded-input budget.
- `dist/bundles.json` records archive and member hashes.
- `dist/package-verification.json` records extracted, installed-consumer verification and native tiny runtime proof.
- Local `.audit/` artifacts retain independent reviews, shortcut controls, runtime evidence, and the final readiness report. They are not automatically published.

A native tiny smoke uses random small weights in the same Qwen class, compiler, trainer, checkpoint, and reload paths. It proves structural training behavior, not pretrained quality or CUDA fit. The real-weight CPU smoke has explicit BF16 memory admission and may refuse the 16 GB local host. It never substitutes tiny weights. Actual 4090 forward/backward, memory headroom, and exact CUDA replay belong to the Pod preflight.

Human review remains unperformed unless a person actually completes it. Automated qualification admits a private experiment, not a production or generalist claim.

## Local commands

Use Python 3.12 and the locked training environment. Local corpus verification additionally needs the pinned font, ffmpeg, and ffprobe. Tesseract is not used. Training needs ffmpeg and ffprobe, but does not rerun upstream acquisition or rendered-field recognition.

```sh
uv sync --frozen --python 3.12 --extra training
.venv/bin/hf download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 --cache-dir data/raw/processor-cache
uv run --with pytest python -m pytest tests -q
.venv/bin/python -m mdprep.verify --root .
.venv/bin/python -m mdprep.processor_check --root .
.venv/bin/python -m mdprep.shortcut_check --root . --output .audit/generated-shortcuts-current.json
.venv/bin/python -m mdtrain smoke --root . --kind tiny --output .audit/new-tiny-smoke.json
.venv/bin/python -m mdprep.package --root .
.venv/bin/python -m mdprep.package verify --root . --include-locked --smoke
```

Smoke output directories must be new. Complete the pinned base download before model or processor commands. Runtime loaders are offline-only. Regeneration or code changes require new qualification where their hashes change.

## Pod execution

Use one RTX 4090 or RTX 3090 with 24 GB VRAM. The CUDA preflight refuses any other GPU. The locked Linux stack uses CUDA 13 dependencies, so the host needs NVIDIA driver 580 or newer. A different container cannot upgrade the host driver. `RUNPOD.md` contains transfer, installation, launch, recovery, and final-evaluation commands. `scripts/runpod_pod.py` creates the network volume and pod, and `scripts/v3-run.sh` and `scripts/v3-finish.sh` run a full cycle on it.

Inference also runs on an Apple Silicon Mac with `--device mps` for `predict` and `evaluate`. Training and finalization stay on CUDA. On a 16 GB M4 the full v3 model matched 1,077 of 1,080 CUDA answers on the rule probe. `.audit/mps-run/` holds that local receipt, and the model card on Hugging Face has the measurements.

The public CLI supervises paid work in its own process group. It requests a reserve-aware graceful checkpoint and enforces a process deadline. It does not stop the Pod or provider billing. The operator must stop the Pod or configure a separate provider-side stop.

The ledger allows $8 of elapsed time, priced at the pod's own `costPerHr`: 57,600 seconds at $0.50/hour and 32,359 seconds at $0.89/hour. It counts allocation wall time, including setup and downtime. A full epoch is attempted only if measured preflight forecasts and fixed-question diagnostic/pilot gates permit it. No total-cost or quality guarantee is made.

## Outputs and scientific limits

A usable inference artifact pairs the pinned base identity, adapter, FP32 pointer, processor/template, calibration state, and checksums. An adapter alone is incomplete. Resumable checkpoints also contain optimizer, scheduler, RNG, phase, exact orders/cursor, qualified input commitments, and elapsed allowance. Resume requires matching code, data/media, processor, backend, and GPU identity.

Evaluation reports accuracy, NLL, summed multiclass Brier, 15-bin ECE, high-confidence errors, tied risk/coverage, source/origin/input/family/holdout slices, and group-bootstrap uncertainty. Infinite mathematical losses are represented as null numeric fields with explicit `nonfinite_values` JSON-pointer metadata. They are not clipped. NaN is rejected.

Finalization may fit one positive temperature on calibration only. Automation policy remains disabled. Locked tests require a finalized artifact. Ablation and real-world evaluation are still needed before claiming image use, temporal reasoning, calibration transfer, or generalist ability. The code is licensed under Apache License 2.0, the same license as the released v3 model.
