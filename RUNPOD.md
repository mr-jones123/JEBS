# RunPod execution

This runbook targets one RTX 4090 at the user's quoted $0.89/hour. It does not start a Pod. Do not use two GPUs merely because the offer says “2 max”.

## Select the host before paying

| Resource | Required setup |
|---|---|
| GPU | One RTX 4090, 24 GB VRAM |
| Host memory | The selected 31 GB offer. Runtime requires at least 28 GiB. |
| CPU | 12 vCPU |
| Python | 3.12 |
| Driver | NVIDIA 580 or newer for the locked Linux CUDA 13 stack |
| Decoder | FFmpeg 6 or newer, with matching ffprobe. Use an Ubuntu 24.04-or-newer template. |
| Container disk | Start with 30 GB for installation and temporary files |
| Persistent storage | Start with 100 GB under `/workspace`. Storage is separately billed. |
| Access | SSH key. Do not expose dashboards or inference ports publicly. |

The driver requirement follows [NVIDIA's CUDA compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). The lock contains CUDA runtime 13.0.96 and cuDNN CUDA 13 packages. Selecting a CUDA 13 container does not upgrade an older host driver. Choose a compatible 4090 host rather than silently changing the locked stack or renting a larger GPU.

Record the actual Pod allocation start from RunPod in `PAID_STARTED_AT`, as RFC3339 UTC. Do not substitute the time training starts after downloads. No Pod is connected during local preparation, so this value cannot be filled in honestly yet.

## Transfer and install

Transfer these local files over SSH into `/workspace/decision-transfer/dist/`.

- `training-code.tar.gz` contains executable code, tests, lock, and runbooks.
- `training-data.tar.gz` contains train/dev/calibration/policy, train-only subsets, media, scoped attribution, licenses, and accepted receipts.
- `bundles.json` contains archive and member inventories.
- `SHA256SUMS.training` contains the two transfer checksums.

Keep `locked-evaluation-data.tar.gz` and its checksum separate until weights and calibration are frozen. Never put Hugging Face caches, keys, upstream test CSVs, or locked-test rows into a training archive.

Run from `/workspace/decision-transfer`.

```sh
sha256sum -c dist/SHA256SUMS.training
mkdir -p /workspace/decision
tar -xzf dist/training-code.tar.gz -C /workspace/decision
tar -xzf dist/training-data.tar.gz -C /workspace/decision
```

Run from `/workspace/decision`. Install `uv` if the selected template does not already provide it. On a root Ubuntu 24.04-or-newer template, install the decoder with `apt-get update && apt-get install -y ffmpeg`. Check `ffmpeg -version` and `ffprobe -version` before continuing. Both must be version 6 or newer. The decoder uses `-fps_mode passthrough`; older Ubuntu templates can install an incompatible FFmpeg.

```sh
python -m pip install uv==0.9.27
uv sync --frozen --python 3.12 --extra training
.venv/bin/hf download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --cache-dir /workspace/decision/data/raw/processor-cache
nvidia-smi
```

Download the complete pinned snapshot. The offline runtime rejects an incomplete Hugging Face snapshot, even if only repository metadata is missing. Do not upgrade Torch, Transformers, or processor files ad hoc after qualification.

Model downloads and dependency installation consume paid allocation time. Stop the Pod if installation cannot proceed. The training guardian is not running during installation and cannot stop provider billing.

## One training command

Set `PAID_STARTED_AT` to the actual allocation start before this command. It fails immediately if that variable is absent.

```sh
.venv/bin/python -m mdtrain pipeline \
  --root /workspace/decision \
  --recipe mixed \
  --run-dir /workspace/decision/runs/mixed \
  --budget-ledger /workspace/decision/runs/budget.json \
  --paid-started-at "${PAID_STARTED_AT:?Set the actual RunPod allocation start in RFC3339 UTC}" \
  --output /workspace/decision/runs/pipeline.json
```

The command performs actual CUDA preflight, diagnostic, pilot, and one main joint pass sequentially. It stops on failed qualification, unsupported hardware, memory failure, exact-replay failure, insufficient remaining allowance, interruption, or a failed fixed-question learning gate. It does not silently quantize, shrink the corpus, remove video, or switch recipes.

Preflight exercises actual maximum-length training examples from all four input kinds and near-2,048-position cases. It checks head-only and language-adapter updates, frozen weights, measured GPU allocation/reservation headroom, exact optimizer replay, and fresh-process inference reload. These checks cannot be pre-approved on the Mac.

The `real` recipe is an explicit alternative for a separate experiment. It trains on 5,600 real-source-derived rows and has no generated-video training exposure. Do not run both recipes concurrently or assume both fit the initial budget. The default planned attempt is `mixed`.

## Baseline run: how much of the result is the frozen backbone

The trained model's locked accuracy is unattributed: nothing yet measures how much the frozen `Qwen3-VL-4B-Instruct` backbone already carried before any JEBS training. The `pointer-only` schedule answers that. It trains only the readout, extends the head phase across the whole training set, and never updates the adapter.

**Rebuild the code archive first, and preserve the accepted one.** The accepted `dist/training-code.tar.gz` was frozen before `--schedule` existed, so it cannot select the baseline. `package build` writes its output straight into `dist/`, replacing the accepted archive and rewriting `dist/bundles.json` with new checksums, so copy `dist/` aside first. Those archives are the evidence for the completed run.

```sh
cp -a dist dist-accepted-2026-10-16
.venv/bin/python -m mdprep.package build --root .
.venv/bin/python -m mdprep.package verify --root .
```

The rebuilt code archive will legitimately differ from the accepted one: it now contains `--schedule`, the new prep module, and the new tests. The data archives should rebuild byte-identically, since `data/prepared` is unchanged.

Then run the baseline as its own experiment, with its own run directory and its own budget ledger. It must not share the accepted run's ledger, and it cannot reuse that run's preflight receipt: a receipt from one schedule is rejected by the other in both directions.

```sh
.venv/bin/python -m mdtrain pipeline \
  --root /workspace/decision \
  --recipe mixed \
  --schedule pointer-only \
  --run-dir /workspace/decision/runs/pointer-only \
  --budget-ledger /workspace/decision/runs/budget-pointer-only.json \
  --paid-started-at "${PAID_STARTED_AT:?Set the actual RunPod allocation start in RFC3339 UTC}" \
  --output /workspace/decision/runs/pipeline-pointer-only.json
```

Timing. The measured head update is 4.578 s against a measured joint update of 11.695 s, a 2.55x difference. The pointer-only main stage is 1,350 head-only updates, projecting to about 6,180 s or 1.72 h against 15,077 s of measured joint update time. `[INFERENCE]` That is arithmetic on two measurements and excludes preflight, diagnostic, pilot, finalization, evaluation, and setup, so budget the run at the same order as the first one and read the realised allowance from the ledger rather than treating 1.72 h as a quote.

What to compare. Evaluate both artifacts on the same locked splits and read the result per slice, not as a single number. The informative comparison is the `test_ood` `task_family` axis and the video families, because those are where the trained model is weakest in lift over chance. If pointer-only lands near the trained model on a slice, the adapter added little there; if it lands near chance, the adapter is doing the work.

Capacity. If the corpus grows to hard-v1 or the other candidate sources, the position budget of 2,048 becomes the binding constraint, not the GPU. `long_policy` already loses most of its members to it.

## hard-v1 paired run and external probes

Build `.audit/rule-probe/` with `tools/rule_probe.py`, then rebuild and verify `dist/`.
On a manually started pod, use the two scripts in order:

```sh
./scripts/hardv1-run.sh --host HOST --port PORT --pod-id POD_ID
# after runs/hardv1-paired/main/result.json reports completed:
./scripts/hardv1-finish.sh --host HOST --port PORT
```

The launcher reads `lastStartedAt` from `GET https://rest.runpod.io/v1/pods/POD_ID`
using `~/.runpod-key`; it never substitutes the later setup or training start. It first
runs the frozen v1 trained and pointer-only artifacts on the rule probe, hard-v1 test,
and DocLayNet request files with the accepted v1 code. It then installs the new archives
and launches `--recipe mixed --schedule paired` detached in its own run directory and
ledger. The finisher fits temperature, transfers the locked archive, evaluates
`test_id,test_ood`, runs the same external requests with v2, and creates one checksummed
recovery archive. Do not stop the pod until the finisher has verified that checksum
locally. DocLayNet images are rooted at `/workspace/external/doclaynet-candidate`.

Score each external set locally after the recovery bundle is extracted. For the probe:

```sh
.venv/bin/python tools/score_predictions.py \
  --key .audit/rule-probe/rule-probe.key.jsonl \
  --predictions v1-trained=.audit/hardv1-paid-run/results/v1-trained-rule-probe.predictions.jsonl \
  --predictions v1-baseline=.audit/hardv1-paid-run/results/v1-pointer-only-rule-probe.predictions.jsonl \
  --predictions v2=.audit/hardv1-paid-run/results/v2-rule-probe.predictions.jsonl \
  --output .audit/hardv1-paid-run/rule-probe-scores.json
```

Use the corresponding hard-v1 or DocLayNet key and prediction names for those reports.

## Budget and stopping

The cumulative process allowance is 32,359 seconds, just under $8 at $0.89/hour. The initial $10 is not a guarantee that all training and evaluation finish. The remaining $2 is a margin, not a measured storage/setup quote. Main also has a five-hour stage allowance.

Do not run this fixed ledger if the actual GPU quote differs from $0.89/hour. A different rate needs a revised and verified allowance. The process deadline is not a provider spending limit.

The lightweight guardian remains outside the paid worker and does not hold its ledger lock. It requests a graceful stop before the save reserve and kills the worker process group at the hard deadline. Forced termination preserves the last published checkpoint, not unsaved updates. GPU stalls and provider shutdown behavior still require observation on the Pod.

The guardian does not stop RunPod billing. Stop the Pod manually or configure a separate provider-side stop. Confirm the provider state after a crash, disconnect, budget stop, or successful run. Persistent storage may continue billing.

## Recovery

Check the pipeline and stage result files first. Resume only the authoritative latest checkpoint, with unchanged data/media, code, processor, backend, ledger, and CUDA device. Use the interrupted stage's actual latest path.

A completed diagnostic or pilot can also use `resume` to reconcile a receipt lost after checkpoint publication. It reuses saved fixed-question evidence without another optimizer update. Conflicting receipts are rejected. Completed main runs are not resumed.

For an interrupted main stage in the default layout, run the following command.

```sh
.venv/bin/python -m mdtrain resume \
  --root /workspace/decision \
  --checkpoint /workspace/decision/runs/mixed/main/checkpoints/latest \
  --preflight /workspace/decision/runs/mixed/preflight.json \
  --budget-ledger /workspace/decision/runs/budget.json
```

Do not create a new ledger to reset spending. A changed allocation or GPU requires a fresh qualified preflight and explicit allocation accounting. Only the latest and previous durable checkpoint generations are retained. Back up important completed artifacts before terminating persistent resources.

## Freeze and evaluate

After a completed main stage, finalization can fit the single temperature using calibration only.

```sh
.venv/bin/python -m mdtrain finalize \
  --root /workspace/decision \
  --checkpoint /workspace/decision/runs/mixed/main/checkpoints/latest \
  --fit-temperature --device cuda \
  --budget-ledger /workspace/decision/runs/budget.json \
  --output /workspace/decision/runs/finalized-model
```

This writes the paired artifact directory `/workspace/decision/runs/finalized-model` and its sibling `finalized-model-freeze-receipt.json`. Use the artifact directory as `--artifact` for `predict` or `evaluate`. An adapter alone is incomplete. Finalization leaves automation policy disabled.

Only then transfer, verify, and extract the locked evaluation archive. It is separately scoped and must not overwrite the training qualification receipts. Keep it in a separate evaluation root with a copy of the same code, lock, and pinned model cache selected through `HF_HUB_CACHE`. Evaluate `test_id,test_ood` with the finalized artifact. Evaluation reports real-source and generated slices separately. The final evaluation and visual/temporal ablations consume additional allowance and may require a separately budgeted run.

After transferring the locked archive and its checksum, run these extraction commands from `/workspace/decision-transfer`.

```sh
sha256sum -c dist/SHA256SUMS.locked
mkdir -p /workspace/decision-evaluation
tar -xzf dist/training-code.tar.gz -C /workspace/decision-evaluation
tar -xzf dist/locked-evaluation-data.tar.gz -C /workspace/decision-evaluation
```

Extracting the code archive into the evaluation root is required, not optional. The locked verification contract checks its verifier source files by path relative to that root, so a root that holds only the locked data fails immediately with `Missing or unconfined artifact: src/mdprep/common.py`. The code is byte-identical to the archive used for training, so this copy changes nothing about which implementation runs. It satisfies the path-relative check.

Run locked evaluation with the original environment and ledger.

```sh
HF_HUB_CACHE=/workspace/decision/data/raw/processor-cache \
  /workspace/decision/.venv/bin/python -m mdtrain evaluate \
  --root /workspace/decision-evaluation \
  --artifact /workspace/decision/runs/finalized-model \
  --splits test_id,test_ood --device cuda \
  --budget-ledger /workspace/decision/runs/budget.json \
  --output /workspace/decision-evaluation/final-evaluation.json
```

Do not infer useful decision quality from the tiny smoke, a falling training loss, or successful temperature fitting. Human review and real source-domain generalization are not established by this preparation.
