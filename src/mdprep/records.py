"""Strict corpus and inference boundaries, independent of model frameworks."""

import math
from dataclasses import dataclass
from pathlib import PurePosixPath

from mdprep.common import SPLITS, stable_id
from mdprep.source_contracts import SOURCE_CONTRACTS


@dataclass(frozen=True)
class Candidate:
    key: str
    description: str


@dataclass(frozen=True)
class VideoRef:
    path: str
    fps: float
    duration_s: float
    frame_count: int


@dataclass(frozen=True)
class DecisionInput:
    id: str
    state: str
    kind: str
    instructions: str
    candidates: tuple[Candidate, ...]
    images: tuple[str, ...]
    videos: tuple[VideoRef, ...]


@dataclass(frozen=True)
class TrainingExample:
    input: DecisionInput
    target_index: int
    group_id: str
    origin: str
    source: str
    family: str
    ood_axis: str


def _keys(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} requires exactly {sorted(keys)}")
    return value


def _text(value: object, name: str, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"Invalid {name}")
    return value


def _path(value: object) -> str:
    value = _text(value, "media path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Media paths must be relative and confined")
    return value


def parse_request(row: dict) -> DecisionInput:
    _keys(row, {"id", "state", "kind", "instructions", "candidates", "images", "videos"}, "request")
    identifier = _text(row["id"], "id")
    state = _text(row["state"], "state", empty=True)
    kind = row["kind"]
    if kind not in {"choice", "noul", "score"}:
        raise ValueError("Unknown decision kind")
    instructions = _text(row["instructions"], "instructions")
    if not isinstance(row["candidates"], list) or len(row["candidates"]) < 2:
        raise ValueError("At least two ordered candidates are required")
    candidates = []
    for item in row["candidates"]:
        _keys(item, {"key", "description"}, "candidate")
        candidates.append(Candidate(_text(item["key"], "candidate key"), _text(item["description"], "candidate description")))
    keys = [item.key for item in candidates]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate candidate keys")
    if kind == "noul" and keys != ["false", "true"]:
        raise ValueError("Noul order must be false,true")
    if kind == "score" and keys != [str(i) for i in range(len(keys))]:
        raise ValueError("Score keys must be ordered ordinal levels")
    if not isinstance(row["images"], list) or not isinstance(row["videos"], list):
        raise ValueError("Media must be arrays")
    images = tuple(_path(path) for path in row["images"])
    videos = []
    for item in row["videos"]:
        _keys(item, {"path", "fps", "duration_s", "frame_count"}, "video")
        for key in ("fps", "duration_s"):
            if type(item[key]) not in (int, float) or not math.isfinite(item[key]) or item[key] <= 0:
                raise ValueError(f"Invalid video {key}")
        if type(item["frame_count"]) is not int or item["frame_count"] <= 0:
            raise ValueError("Invalid video frame_count")
        videos.append(VideoRef(_path(item["path"]), float(item["fps"]), float(item["duration_s"]), item["frame_count"]))
    if images and videos:
        raise ValueError("Image/video mixtures are unsupported")
    return DecisionInput(identifier, state, kind, instructions, tuple(candidates), images, tuple(videos))


def validate_record(row: dict) -> None:
    _keys(row, {"id", "group_id", "split", "modality", "state", "images", "videos", "question", "target", "provenance", "audit"}, "record")
    _text(row["group_id"], "group_id")
    if row["split"] not in SPLITS:
        raise ValueError("Unknown split")
    _keys(row["question"], {"type", "instructions", "criteria"}, "question")
    question = row["question"]
    if not isinstance(question["criteria"], dict):
        raise ValueError("Criteria must be an ordered mapping")
    request = {key: row[key] for key in ("id", "state", "images", "videos")}
    request.update(kind=question["type"], instructions=question["instructions"], candidates=[{"key": k, "description": v} for k, v in question["criteria"].items()])
    parsed = parse_request(request)
    _keys(row["target"], {"label"}, "target")
    if row["target"]["label"] not in question["criteria"]:
        raise ValueError("Unknown target")
    if row["modality"] not in {"text", "image", "mixed", "video"}:
        raise ValueError("Unknown modality")
    modality = row["modality"]
    if (modality == "text" and (parsed.images or parsed.videos)) or (modality in {"image", "mixed"} and (not parsed.images or parsed.videos)) or (modality == "video" and (not parsed.videos or parsed.images)):
        raise ValueError("Media does not match modality")
    if modality in {"text", "mixed"} and not parsed.state.strip():
        raise ValueError("Text/mixed evidence must not be empty")
    provenance = row["provenance"]
    if not isinstance(provenance, dict) or not isinstance(row["audit"], dict):
        raise ValueError("Provenance/audit must be objects")
    for key in ("source", "family", "source_domain", "license", "generator_version"):
        _text(provenance.get(key), key)
    source = provenance["source"]
    if source not in SOURCE_CONTRACTS:
        raise ValueError("Unsupported canonical data source")
    expected_origin, held_out_axis = SOURCE_CONTRACTS[source]
    if provenance.get("origin") != expected_origin or provenance["source_domain"] != source:
        raise ValueError("Origin and source domain must match the source contract")
    expected_axis = held_out_axis if row["split"] == "test_ood" else "none"
    if provenance.get("ood_axis") != expected_axis:
        raise ValueError("Holdout axis must match the source and split")
    if "variant" not in provenance or row["id"] != stable_id(provenance["source"], row["group_id"], provenance["family"], provenance["variant"]):
        raise ValueError("Record identity does not match serialized variant")
    if parsed.kind == "choice":
        if list(question["criteria"]) != [f"option_{i}" for i in range(len(parsed.candidates))]:
            raise ValueError("Choice keys must be neutralized exactly once")
        original = row["audit"].get("original_candidate_keys")
        if not isinstance(original, list) or len(original) != len(parsed.candidates) or not all(isinstance(k, str) and k for k in original) or len(set(original)) != len(original):
            raise ValueError("Missing original response keys")
        index = list(question["criteria"]).index(row["target"]["label"])
        if row["audit"].get("original_target_key") != original[index]:
            raise ValueError("Original target disagrees with canonical target")


def parse_record(row: dict) -> TrainingExample:
    validate_record(row)
    question = row["question"]
    keys = list(question["criteria"])
    target = keys.index(row["target"]["label"])
    output_keys = row["audit"]["original_candidate_keys"] if question["type"] == "choice" else keys
    request = {key: row[key] for key in ("id", "state", "images", "videos")}
    request.update(kind=question["type"], instructions=question["instructions"], candidates=[{"key": key, "description": description} for key, description in zip(output_keys, question["criteria"].values())])
    provenance = row["provenance"]
    return TrainingExample(parse_request(request), target, row["group_id"], provenance["origin"], provenance["source"], provenance["family"], provenance["ood_axis"])
