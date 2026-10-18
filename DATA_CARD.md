# Multimodal decision corpus

## Use and readiness

This private experimental fine-tuning corpus teaches candidate decisions over text, photographs, rendered images, joint text/image rules, and short temporal clips. It is not a validated generalist benchmark or a production authorization policy.

Assembly produced 44,904 questions. Training clearance depends on successful current-hash `data/verification-contract.json` and `data/processor-contract.json`, not this document or file existence. Failed verification cannot publish a new successful contract. Human review and independent photo-visible truth remain unperformed.

## Origins

| Source | Questions | Human contribution | Software construction |
|---|---:|---|---|
| BANKING77 | 5,270 | Customer utterances and intent labels | Questions, candidate pools, option order, and partitions |
| Open Images V7 | 3,000 | Photographs and explicit image-label verification | Choice, boolean, and named-class-count decisions |
| Generated text rules | 3,730 | None | Policies, evidence, and rule-derived targets |
| hard-v1 generated rules | 8,804 | None | Published varying rules converted to canonical choice, boolean, and score decisions |
| rule-variants-v1 generated rules | 6,100 | None | Counterfactual template variants and random rule structures with code-computed targets |
| Generated images and joint rules | 15,000 | None | Receipts, charts, forms, dashboards, shapes, visible policies, and targets |
| Generated video | 3,000 | None | Four-frame trajectories, colors, temporal questions, and targets |

There are 8,270 real human-label-derived questions and 36,634 generated questions. Real-source-derived does not mean a human authored each resulting question. Generated examples use procedural rules and rendering, not paid API-teacher answers.

The corpus has 503 real photographs, 3,000 generated PNGs, and 750 MP4s. Generated images are 450 by 450 pixels. Videos are 256 by 192 pixels, four frames at 2 FPS, with two-second duration. Several questions share one evidence group. Count unique content separately from repeated questions.

Video encoding preserves RGB losslessly. Contact scenes retain visible evidence for both objects. These repairs prevent codec damage and complete occlusion from making precise controlled labels unverifiable. They do not add natural-video diversity.

## Partitions

| Partition | Questions | Role |
|---|---:|---|
| train | 33,276 | Weight updates |
| dev | 2,814 | Development evaluation |
| calibration | 2,814 | Optional scalar temperature after selecting weights |
| policy | 2,000 | Reserved. No automation policy is implemented. |
| test_id | 3,000 | Locked final same-family evaluation |
| test_ood | 1,000 | Locked declared holdout axes |

The 1,000-question pilot and 64-question diagnostic files are byte-identical train-derived subsets of the original corpus. They are not additional examples or evaluation sets. The reader enforces their train-only membership, bounded size, row identity, and uniqueness.

All questions and counterfactuals derived from one source evidence unit stay together. Exact media, normalized text, source groups, photo creator/original/landing identities, and temporal rotation/reversal orbits have explicit isolation checks. These do not establish complete semantic or perceptual deduplication.

The `real` recipe selects 5,600 training questions from BANKING77 and Open Images. It has no generated-video or joint policy/image training exposure. The `mixed` recipe selects all 33,276 training questions. Report real-source and generated evaluation slices separately. A larger mixture does not by itself establish generalization.

## Meaning and verification

BANKING77 preserves the pinned upstream training utterances and human intent labels. The official upstream test file supplies no corpus training, development, calibration, policy, or final-evaluation examples. Candidate descriptions are reconciled with their intent meanings.

Open Images uses explicit human verification. Confidence 1 is an inherited verified-positive label. Confidence 0 is an inherited verified-negative label. Missing annotation is never a negative. Score questions count positive class labels among the three named classes, not object instances. Raw source fidelity does not independently adjudicate visible truth. Ontology and fine-grained labels may remain ambiguous or noisy.

Generated text verification admits complete visible policy/evidence grammars, parses variable thresholds and rubrics, and checks candidate meanings. It does not reuse generator target functions.
hard-v1 contributes all 7,876 admitted train questions. Its 928 development questions are assigned by complete `group_id` to equal 464-question development and calibration halves. The published hard-v1 test partition is excluded from the canonical corpus and remains an external evaluation set.
`rule-variants-v1` contributes 5,400 train questions plus 350 each in development and calibration. Every evidence unit has two rule siblings, with most sibling labels changed and a minority retained. Its random grammar covers all-of, any-of, exception-veto, ordered first-match, and additive-point rules over banking and non-banking domains. Natural-language wording lint rejects machine notation and ambiguous numeric comparisons, while boundary coverage requires facts exactly at, one below, and one above inclusive thresholds. Reserved combinator/domain combinations and rendering style 4 appear only in the 600-question external held-out structure set under `.audit/rule-structures-heldout/`.


Generated image verification is deliberately renderer-specific. It reads actual PNG text through complete dictionaries using the pinned font, supported field positions, and exact glyph pixels. It independently measures colored geometry and bar extents. It then interprets the full visible rule and candidate meanings. No row target, audit value, or source scene value is used to select recognized text. Unknown, damaged, incomplete, or ambiguous fields fail. This is not generic OCR, an independent rendering engine, or human readability certification.

Video verification fully decodes the four frames. Answers come from observed colors, centers, contact, motion, and order. Declared frame counts, indices, times, and audit facts must reconcile with all observations. A hidden family identifier cannot substitute for checking the visible question and candidate meanings.

## Shortcut controls and repairs

The initial BANKING77 candidate sets leaked intent information. Balanced five-intent blocks removed that construction defect. The unchanged description-only development probe fell from 317 of 360 to 72 of 360, or 20 percent. This is not a trained-model result.

Three numeric image lookup families initially made every correct answer the second-smallest candidate. New distractors share the actual value support. Spatial lookup uses three candidates because its support contains only three possible correct counts. Rubrics and action cutpoints are now sampled independently before targets are computed from actual scene values. They are not reverse-engineered from desired labels.

`python -m mdprep.shortcut_check` repeats the numeric-rank and rubric-gap controls using train/dev only. Retained before/after measurements are in `.audit/generated-shortcuts-before.json` and `.audit/generated-shortcuts-after.json`. The repaired second-smallest control is 18 of 60, or 30 percent. Candidate-count-weighted uniform chance is about 27.8 percent. The rubric-gap control fell from 72 to 40 percent, while its empirical majority baseline changed from 25 to 33.5 percent. Residual priors remain. These narrow controls do not prove the corpus is shortcut-free or that a trained model uses its images.

## Holdout axes

BANKING77 holds out intent classes within one source. Open Images holds out transport-positive images based on their complete reconciled positive label sets. Negative transport vocabulary may still occur in ID questions. This is not an unseen-word guarantee.

Generated visuals hold out layouts. Generated text and video hold out task families. Source-specific origin and axis declarations are enforced rather than accepted as arbitrary enums. None of these sets is a real source-domain holdout. Unknown pretraining exposure remains unexcluded.

## Licensing and private distribution

BANKING77 is pinned to `PolyAI-LDN/task-specific-datasets` revision `57ec275d8078af65b7731c2a98be812d844a6d6b`. Its CC BY 4.0 text, attribution, and conversion notice travel with the data.

Each Open Images photograph retains its exact original license URL, author, landing URL, metadata, and processed-file identity. Only the builder's admitted CC BY or CC0 variants are used. Annotation terms are separately CC BY 4.0. Source metadata is not a warranty of every underlying right. Resizing and rotation are recorded.

The Atkinson Hyperlegible Regular font is pinned to Google Fonts revision `1cb311624b2ddf88e9e37873999d165a8cd28b46`. Its SHA256 is `7fb917c89019896d0b52ee84b7cbb3304c18cb90b19a62f5e32712bd23e97669`. The font and SIL Open Font License accompany each scoped data archive. Generated videos contain no third-party media or fonts.

No public license is assigned here to the user's original code, generated scenes, or trained artifact. Keep the handoff private pending release authorization and license review. Base model weights, credentials, and caches are not in the archives.

## Review and archive boundary

The 200-example review pack is separate from automated qualification. Creating it does not mean a person reviewed it. Any model-assisted inspection must be labeled as such.

The training archive contains only its primary splits, train-derived subsets, referenced media, scoped provenance, licenses, and accepted receipts. Final-test rows, media, and label-bearing source entries remain in a separate locked archive. Do not expose them to model selection or calibration. Archive extraction and installed-consumer checks have separate evidence from semantic qualification.
