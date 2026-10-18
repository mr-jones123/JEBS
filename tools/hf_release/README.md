---
license: apache-2.0
language: en
base_model: Qwen/Qwen3-VL-4B-Instruct
base_model_relation: adapter
library_name: peft
tags:
  - decision-model
  - calibration
  - lora
  - pointer-head
  - multiple-choice
  - qwen3-vl
  - multimodal
---

# JEBS-4B

JEBS-4B is a decision model. It reads a state (text, an image, or a short video), a question, and the options you supply, and it returns a calibrated probability for each option in one forward pass. It does not generate text.

It is a LoRA adapter and a small decision readout on a frozen `Qwen/Qwen3-VL-4B-Instruct`. This is the third trained version (v3) of a research project. It is an experiment, not a production system. Read [Limits](#limits) before you use it.

## The idea: replace the vocabulary head with a decision head

A language model ends in a vocabulary head. At each position it scores about 150,000 tokens and picks the next one. To make a decision with it, you ask it to write an answer and maybe a confidence, then parse the text. The answer arrives one token at a time, the confidence is just more generated text, and nothing ties the output to the options you meant.

JEBS keeps the pretrained transformer body and does not use the vocabulary head. The input lists each option, and the model reads two kinds of hidden state.

- The hidden state at the end of each option's text.
- The hidden state at a final "decide" position after the question.

A bias-free two-projection readout maps both into a 256-dimensional space and scores each option by a dot product (`two-projection-dot-256-v1` in `src/mdtrain/engine.py`). A softmax over only the supplied options gives the distribution. A single temperature, fitted after training on a separate calibration split, corrects overconfidence. This release uses T = 1.9185.

![JEBS-4B architecture. The request goes through the frozen Qwen3-VL-4B body with LoRA. The unused vocabulary head is greyed out. The trained decision readout scores each option-end hidden state against the decide hidden state and returns a softmax over the supplied options.](https://huggingface.co/experimentalmachines/JEBS-4B/resolve/main/architecture.svg)

The same recipe appears in other open decision models, such as [Kev](https://huggingface.co/jaredpalmer/kev-0.8b) and [circuit-vl-4b](https://huggingface.co/jbarney/circuit-vl-4b). JEBS differs in its base (a native vision-language model), its four-frame video path, and its training data.

Three question types share the readout.

| Type | Options | Use |
|---|---|---|
| `choice` | Any named options | Pick one action or category |
| `noul` | `false`, `true` | Yes or no |
| `score` | `0` to `n` in order | An ordered level, such as a severity tier |

## What we did

We trained three versions on the same architecture. Only the training data changed. Every result below comes from locked test splits that were read once per version and never used for selection or calibration.

| Version | Training rows | What we added | Why |
|---|---:|---|---|
| v1 | 20,000 | Rendered documents and charts (12,000), BANKING77 and Open Images (5,600 real labels), three rule templates (2,400) | The first full run |
| v2 | 27,876 | 7,876 hard-v1 skill records | Held-out rule families scored 70% in v1 |
| v3 | 33,276 | 5,400 `rule-variants-v1` rows: rule-text variants of the three templates and random rule structures, built as counterfactual pairs | v2 still kept the old answer after many rule edits |

We also trained a baseline with the body and the LoRA frozen and only the readout learning. It scored 75.13% on `test_id` and 71.00% on `test_ood`, against 93.67% and 88.60% for v1. The adapter adds about 18 points, so training does more than calibrate an easy readout.

The runs used one rented RTX 4090 (v1 and the baseline) or one RTX 3090 (v2 and v3). Total GPU spend for the project was 17.05 USD. The v3 run cost 3.79 USD.

## Main findings

1. **Templated rules were memorized, not read.** Each v1 template had 800 training rows and one fixed rule text. v1 scored 100% on those templates in distribution. When we edited a rule so the right answer changed, v1 still gave the old answer 31.9% of the time.
2. **Varied rules in other tasks did not fix memorized templates.** hard-v1 raised v2 from 55.3% to 76.2% on hard-v1's own test. It moved the edited-rule probe only from 65.0% to 71.7%.
3. **Counterfactual rule pairs taught rule reading.** v3 trained on pairs where the same evidence has different correct answers under different rules. On the probe, v3 answers 84.2% of label-changing edits correctly, and it keeps the old answer only 15.0% of the time. Against v2 it fixed 50 answers and broke 5 (exact McNemar test, p = 2e-10).
4. **The rule reading transfers to unseen rule structures.** On 600 held-out questions with combinations of rule logic, domain, and wording style that never appeared in training, v3 scores 92.7% against v2's 63.2%. Both answers of a pair are right 85.3% of the time against 34.7% (183 fixed, 6 broken, p = 1e-46).
5. **Three locked held-out rule families did not move.** They scored 70.0%, 70.0%, and 66.7% across v1, v2, and v3, on 90 questions. They use the same question shapes v3 now reads well, so the remaining gap is probably the arithmetic they require, such as deadlines, time windows, and balances. This is a hypothesis, not a measurement.
6. **Synthetic labels were memorized and real labels generalized.** In v1, training removed 100% of the baseline's errors on generated rules in distribution but only 49.1% out of distribution. On real human labels the order flipped, with 43.6% in distribution against 71.8% out of it.

## Results for this release (v3)

| Locked split | Questions | Accuracy | ECE | Answers with confidence of 0.95 or more |
|---|---:|---:|---:|---|
| `test_id` | 3,000 | 93.27% | 0.87% | 2,374 answers, 12 wrong (0.51%) |
| `test_ood` | 1,000 | 87.10% | 3.75% | 692 answers, 10 wrong (1.45%) |

The chance rate on `test_id` is 33.42%, because the test is balanced across two to six options per question.

| Slice | `test_id` | `test_ood` |
|---|---:|---:|
| Text | 98.0% | 86.7% |
| Image | 88.3% | 91.6% |
| Text plus rendered image | 99.1% | 97.4% |
| Video | 71.0% | 41.0% |
| BANKING77 intent routing | 96.9% | 95.2% (held-out intents) |

| External set | Questions | v1 | v2 | v3 |
|---|---:|---:|---:|---:|
| Edited-rule probe, label-changing edits | 360 | 65.0% | 71.7% | 84.2% |
| Held-out rule structures | 600 | | 63.2% | 92.7% |
| hard-v1 test | 921 | 55.3% | 76.2% | 76.3% |
| DocLayNet real document regions (majority class 49.5%) | 200 | 64.5% | 66.5% | 64.5% |

All numbers come from one training run per version.

## Strengths

- **Reading rendered documents and charts.** On receipts, request forms, dashboards, and bar charts, value lookup and visible-relation questions score 100% on both locked splits. Rubric questions on the same documents score 90% to 98%.
- **Customer-intent routing.** 95.2% on BANKING77 intents the model never saw in training.
- **Following a written rule.** All-of, any-of, unless-exception, first-match, and additive-points rules score 92.7%, including combinations the model never trained on.
- **Calibration on familiar tasks.** In distribution, ECE is 0.87%, and answers at 0.95 confidence or higher are wrong 0.51% of the time. A confidence threshold can send uncertain cases to a person.

## Limits

- **Deadlines, time windows, and balances.** The held-out dispute-deadline family scores 40% on a four-option question.
- **Video.** Four frames compile to two temporal positions. Held-out video scores 41%, close to chance.
- **Real photographs and real documents.** Counting objects in photos scores 53% in distribution. Real document regions score 64.5%.
- **Calibration out of distribution.** ECE rises from 0.87% to 3.75%, and it is worse on unfamiliar task families.
- **Not measured.** Option-order sensitivity, blank or wrong media controls, latency, peak memory, and variation across training seeds.
- **No human review.** Every label and check is automated.

Do not use JEBS-4B for automated decisions with legal, financial, medical, employment, or access consequences for people. Use it as a research model, or as a triage step with a person reviewing its output.

## Run it

You need Linux with a CUDA GPU that supports BF16 and has 24 GB of memory. We ran it on an RTX 3090 and an RTX 4090. You also need Python 3.12 and [uv](https://docs.astral.sh/uv/). The loader pins `transformers` 5.19.0 and `torch` 2.14.1 and refuses other versions, so install from the lock file.

1. Download this repository.

	```sh
	hf download experimentalmachines/JEBS-4B --local-dir JEBS-4B
	cd JEBS-4B
	```

2. Install the locked environment.

	```sh
	uv sync --frozen --python 3.12 --extra training
	```

3. Download the pinned base model into the cache that the loader reads. The loader runs offline and never downloads weights itself.

	```sh
	.venv/bin/hf download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 --cache-dir data/raw/processor-cache
	```

4. Run the bundled example requests.

	```sh
	.venv/bin/python -m mdtrain predict --root . --artifact model --input examples/requests.jsonl --media-root . --device cuda --output predictions.jsonl
	```

	On an RTX 3090 the three predictions are these. Your probabilities can differ in the last digits on other GPUs.

	| `id` | `predicted_key` | Probability |
	|---|---|---:|
	| `refund-yes-no` | `true` | 0.989 |
	| `shipment-action` | `review` | 1.000 |
	| `claim-severity` | `3` | 0.993 |

Each line of `predictions.jsonl` holds `id`, `probabilities` (one per option key), `predicted_key`, `expected_score` (for `score` questions), `artifact_id`, `processor_id`, `calibration_id`, and `policy`. The `artifact_id` is `21bcc482ada605d4b197210cf7a8a3175fd8c401883908c10bb26dbcf6ed8f60`. The loader checks the seal over `model/` and refuses a modified artifact.

To keep the base model somewhere else, set `HF_HUB_CACHE` to that cache directory.

## Request format

Write one JSON object per line.

| Field | Type | Rule |
|---|---|---|
| `id` | string | Unique per request |
| `state` | string | The policy, document text, or context. May be empty |
| `kind` | string | `choice`, `noul`, or `score` |
| `instructions` | string | The question |
| `candidates` | list of `{"key", "description"}` | At least two. `noul` keys are `false` then `true`. `score` keys are `"0"`, `"1"`, and so on in order |
| `images` | list of paths | Relative to `--media-root` |
| `videos` | list of `{"path", "fps", "duration_s", "frame_count"}` | Relative to `--media-root`. A request holds images or videos, not both |

`examples/requests.jsonl` has one request of each kind.

## Training details

| Setting | Value |
|---|---|
| Base | `Qwen/Qwen3-VL-4B-Instruct` at `ebb281ec70b05090aa6165b016eac8ec08e71b17`, frozen, BF16 |
| Adapter | LoRA, rank 16, on the language tower |
| Readout | Bias-free two projections into 256 dimensions, dot-product scoring |
| Schedule | 100 readout-only updates at learning rate 1e-3. Then 2,080 joint updates with the LoRA at 5e-5 and the readout at 1e-4. Batch 16 |
| Loss | Cross-entropy over each question's options |
| Calibration | One temperature fitted by negative log-likelihood on a 2,814-question calibration split |
| Context | At most 2,048 positions per question |

Training data sources and licenses are in `NOTICE`. Generated data uses code-computed labels. No teacher model or human produced a label.

## What this repository holds

| Path | Contents |
|---|---|
| `model/` | The sealed artifact: `adapter/`, `pointer.safetensors`, `processor/`, `calibration.json`, `model-spec.json`, `checksums.json` |
| `src/mdtrain`, `src/mdprep` | The runtime that loads the artifact and answers requests, plus the data tools that built the corpus |
| `pyproject.toml`, `uv.lock` | The locked environment |
| `examples/requests.jsonl` | Three example requests |
| `architecture.svg` | The architecture diagram at the top of this card |
| `SHA256SUMS` | Checksums for every file |

`model/adapter/adapter_config.json` names the base model by the path it had on the training machine. The runtime passes the already loaded base to PEFT and does not read that field.

## License

Apache License 2.0 for the adapter, the readout, and the code. The base model is Apache License 2.0 from the Qwen team. See `LICENSE` and `NOTICE`.

## References

These are the sources that shaped the design, the data, and the evaluation.

### Decision models

| Source | How it shaped JEBS |
|---|---|
| TypeSafe, [Introducing System One models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | The output contract: typed `choice`, `noul`, and `score` questions with probabilities returned directly, not generated |
| Archer Hume, [Jev's architecture unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/) | Black-box evidence that the candidate list and shared state change the answer |
| [Kev-0.8B](https://huggingface.co/jaredpalmer/kev-0.8b) | Frozen base, LoRA, and a pointer head. Source of the hard-v1 data. Its card reports the same in-distribution-only gain from hard-v1 that v2 measured |
| [circuit-vl-4b](https://huggingface.co/jbarney/circuit-vl-4b) | The closest multimodal precedent: frozen Qwen3-VL-4B-Instruct, language-only LoRA, and a two-projection pointer |
| [Decision-1.0-Eos-0.8B](https://huggingface.co/vllm-sr/Decision-1.0-Eos-0.8B) | A full-tuning reference with a richer head and separate selection and calibration partitions |
| [JEVision](https://huggingface.co/divyanshx11/JEVision) | A native-vision pointer design with separate text and image routes |

### Rule reading

| Paper | How it shaped JEBS |
|---|---|
| Wu et al., [Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks](https://arxiv.org/abs/2307.02477) (2023) | The method behind the edited-rule probe: keep the evidence, change the rule, and check that the answer follows |
| Kaushik et al., [Learning the Difference that Makes a Difference with Counterfactually-Augmented Data](https://arxiv.org/abs/1909.12434) (2020) | The counterfactual-pair design of `rule-variants-v1` |
| Zhang et al., [On the Paradox of Learning to Reason from Data](https://arxiv.org/abs/2205.11502) (2022) | Why `rule-variants-v1` has surface-feature and evidence-only shortcut checks |
| Clark et al., [Transformers as Soft Reasoners over Language](https://arxiv.org/abs/2002.05867) (2020) | Evidence that synthetic rules teach rule reading when the rules vary per example |
| Tafjord et al., [ProofWriter](https://arxiv.org/abs/2012.13048) (2020) | Natural-language deduction data that maps onto `noul` and `choice` |
| [DynaGuard](https://arxiv.org/abs/2509.02563) (2025) | A precedent for supplying the policy with each request |
| OpenAI, [gpt-oss-safeguard](https://openai.com/index/introducing-gpt-oss-safeguard/) (2025) | A second precedent for policy as input |

### Data, calibration, and evaluation

| Paper | How it shaped JEBS |
|---|---|
| [Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning](https://arxiv.org/abs/2505.20161) (2025) | Support for varying rules and structures instead of adding more rows of the same kind |
| Zheng et al., [Large Language Models Are Not Robust Multiple Choice Selectors](https://arxiv.org/abs/2309.03882) (2023) | Why options are scored by a pointer instead of by option-letter tokens, and why option-order sensitivity is still on the list to measure |
| [On the Limitations of Temperature Scaling for Distributions with Overlaps](https://arxiv.org/abs/2306.00740) (2023) | Why one temperature calibrates familiar tasks well and unfamiliar ones less well |
| [Breaking Down Video LLM Benchmarks: Knowledge, Spatial Perception, or True Temporal Understanding?](https://arxiv.org/abs/2505.14321) (2025) | The no-video and shuffled-frame controls that any video claim needs |
| [DocLayNet](https://arxiv.org/abs/2206.01062) (2022) | The real-document regions used as the external DocLayNet test |
