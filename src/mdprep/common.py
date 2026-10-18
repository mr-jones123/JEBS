"""Shared serialization and source-acquisition contract for data builders."""

import hashlib
import json
import random
import urllib.request
from pathlib import Path
from typing import Iterable

SPLITS = ("train", "dev", "calibration", "policy", "test_id", "test_ood")
PART_QUOTAS = {
    "text": dict(zip(SPLITS, (6000, 600, 600, 600, 900, 300))),
    "photos": dict(zip(SPLITS, (2000, 200, 200, 200, 300, 100))),
    "visual_image": dict(zip(SPLITS, (3000, 300, 300, 300, 450, 150))),
    "visual_mixed": dict(zip(SPLITS, (7000, 700, 700, 700, 1050, 350))),
    "videos": dict(zip(SPLITS, (2000, 200, 200, 200, 300, 100))),
    "hardv1": dict(zip(SPLITS, (7876, 464, 464, 0, 0, 0))),
    "rule_variants": dict(zip(SPLITS, (5400, 350, 350, 0, 0, 0))),
}
GENERATOR_VERSION = "mdprep-2026-10-15-v2"


def stable_id(*values: object) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def rng_for(*values: object) -> random.Random:
    return random.Random(int(stable_id(*values), 16))


def fetch(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return destination
    request = urllib.request.Request(url, headers={"User-Agent": "multimodal-decision-data/0.1"})
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def choice(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions,
            "criteria": {"false": "No", "true": "Yes"}}


def score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions,
            "criteria": {str(index): level for index, level in enumerate(levels)}}


def record(*, split: str, source: str, group_id: str, family: str,
           modality: str, state: str, question: dict, label: str,
           images: list[str] | None = None, videos: list[dict] | None = None,
           provenance: dict, audit: dict | None = None, variant: object = "") -> dict:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    criteria = question["criteria"]
    if label not in criteria:
        raise ValueError(f"Label {label!r} is not a candidate")
    question = {**question, "criteria": dict(criteria)}
    audit = dict(audit or {})
    if question["type"] == "choice":
        candidates = list(criteria.items())
        rng_for(group_id, family, variant, "option-order").shuffle(candidates)
        audit["original_candidate_keys"] = [key for key, _ in candidates]
        audit["original_target_key"] = label
        question["criteria"] = {f"option_{i}": value for i, (_, value) in enumerate(candidates)}
        label = f"option_{audit['original_candidate_keys'].index(label)}"
    real = source in {"banking77", "open_images_v7_train_human_verified"}
    generated_rule = source in {"mdprep_text_rules", "hard-v1", "rule-variants-v1"}
    origin = "real_human_label" if real else "generated_rule" if generated_rule else "generated_media"
    axis = "none"
    if split == "test_ood":
        if real:
            axis = "class_or_intent"
        elif source == "synthetic-visuals-v1":
            axis = "layout"
        elif source == "rule-variants-v1":
            axis = "template"
        else:
            axis = "task_family"
    metadata = {**provenance, "source": source, "family": family,
                "generator_version": GENERATOR_VERSION, "variant": variant,
                "origin": origin, "ood_axis": axis, "source_domain": source}
    return {
        "id": stable_id(source, group_id, family, variant),
        "group_id": group_id,
        "split": split,
        "modality": modality,
        "state": state,
        "images": images or [],
        "videos": videos or [],
        "question": question,
        "target": {"label": label},
        "provenance": metadata,
        "audit": audit,
    }
