"""Strict prepared readers and explicit without-replacement question orders."""

import collections
import json
import random
from dataclasses import dataclass
from pathlib import Path

from mdprep.common import sha256_file
from mdprep.native import digest_json, validate_contract
from mdprep.records import TrainingExample, parse_record
from mdprep.verify import validate_verification_contract

EVALUATION_SPLITS = ("dev", "calibration", "policy", "test_id", "test_ood")

DEFAULT_SCHEDULE = "paired"


@dataclass(frozen=True)
class TrainingSchedule:
    name: str
    phase_components: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def phases(self) -> tuple[str, ...]:
        return tuple(phase for phase, _ in self.phase_components)

    def compose(self, components: dict[str, list[str]]) -> dict[str, list[str]]:
        orders = {}
        for phase, names in self.phase_components:
            order = []
            for name in names:
                order.extend(components[name])
            orders[phase] = order
        return orders


SCHEDULES = {
    "paired": TrainingSchedule("paired", (("head", ("warmup",)), ("joint", ("training",)))),
    "pointer-only": TrainingSchedule("pointer-only", (("head", ("warmup", "training")),)),
}
OPTIMIZER_PHASES = tuple(dict.fromkeys(
    phase for schedule in SCHEDULES.values() for phase in schedule.phases))


def training_schedule(name: str) -> TrainingSchedule:
    try:
        return SCHEDULES[name]
    except KeyError as error:
        raise ValueError(f"Unknown training schedule: {name}") from error


def bound_schedule(payload: dict) -> str:
    """Resolve historical paired identities while requiring new schedules to name themselves."""
    name = payload.get("schedule", DEFAULT_SCHEDULE)
    training_schedule(name)
    return name


def qualified_inputs(root: Path, *, processor: dict | None = None,
                     verification: dict | None = None) -> dict:
    """Commit successful qualification, processor implementation, rows, and media."""
    processor = processor if processor is not None else validate_contract(root)
    verification = verification if verification is not None else validate_verification_contract(root)
    for split in set(processor["prepared_hashes"]) & set(verification["prepared_hashes"]):
        if processor["prepared_hashes"][split] != verification["prepared_hashes"][split]:
            raise ValueError("Processor and data qualification disagree on prepared rows")
    for relative in set(processor["media_hashes"]) & set(verification["media_hashes"]):
        if processor["media_hashes"][relative] != verification["media_hashes"][relative]:
            raise ValueError("Processor and data qualification disagree on media")
    return {"processor_id": processor["processor_id"],
            "processor_contract_id": digest_json(processor),
            "qualification_id": verification["qualification_sha256"],
            "prepared_hashes": processor["prepared_hashes"],
            "media_hashes": processor["media_hashes"]}


@dataclass(frozen=True)
class DatasetView:
    examples: tuple[TrainingExample, ...]
    rows: tuple[dict, ...]
    split: str
    recipe: str
    identity: str

    @classmethod
    def open(cls, root: Path, *, split: str, recipe: str = "mixed"):
        if split not in ("train", *EVALUATION_SPLITS) or recipe not in ("real", "mixed"):
            raise ValueError("Unknown dataset split or recipe")
        processor = validate_contract(root)
        verification = validate_verification_contract(root)
        if split not in verification["scope"]:
            raise ValueError(f"Verified scope does not include {split}")
        if split not in processor["prepared_hashes"]:
            raise ValueError(f"Processor scope does not include {split}")
        path = root / "data/prepared" / f"{split}.jsonl"
        rows, examples = [], []
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["split"] != split:
                    raise ValueError("Prepared row split mismatch")
                example = parse_record(row)
                if recipe == "real" and example.origin != "real_human_label":
                    continue
                rows.append(row)
                examples.append(example)
        ids = [example.input.id for example in examples]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Dataset view is empty or contains duplicate IDs")
        prepared_hash = sha256_file(path)
        if prepared_hash != processor["prepared_hashes"][split]:
            raise ValueError("Prepared rows changed while constructing the qualified view")
        media = {relative for example in examples
                 for relative in (*example.input.images, *(video.path for video in example.input.videos))}
        commitment = qualified_inputs(root, processor=processor, verification=verification)
        identity = digest_json({"prepared_sha256": prepared_hash, "recipe": recipe, "selected_ids": ids,
                                "media_hashes": {name: processor["media_hashes"][name] for name in sorted(media)},
                                "qualified_inputs": commitment})
        return cls(tuple(examples), tuple(rows), split, recipe, identity)

    def by_id(self) -> dict[str, TrainingExample]:
        return {example.input.id: example for example in self.examples}


def ordered_ids(view: DatasetView, *, seed: int, lengths: dict[str, int]) -> list[str]:
    rng = random.Random(seed)
    strata = collections.defaultdict(lambda: collections.defaultdict(list))
    for example, row in zip(view.examples, view.rows):
        strata[(example.origin, row["modality"], example.input.kind)][lengths[example.input.id] // 256].append(example.input.id)
    queues = []
    for stratum in sorted(strata):
        buckets = strata[stratum]
        keys = sorted(buckets)
        rng.shuffle(keys)
        queue = []
        for key in keys:
            rng.shuffle(buckets[key])
            queue.extend(buckets[key])
        queues.append(collections.deque(queue))
    rng.shuffle(queues)
    result = []
    while any(queues):
        for queue in queues:
            if queue:
                result.append(queue.popleft())
    if len(result) != len(view.examples) or len(set(result)) != len(result):
        raise ValueError("Without-replacement order is not a permutation")
    return result


def plan_orders(view: DatasetView, stage: str, lengths: dict[str, int],
                schedule: str = DEFAULT_SCHEDULE) -> dict:
    limits = {"diagnostic": (64, 8), "pilot": (1000, 16), "main": (len(view.examples), 100)}
    if stage not in limits:
        raise ValueError("Unknown training stage")
    selected_schedule = training_schedule(schedule)
    size, head_updates = limits[stage]
    if len(view.examples) < size:
        raise ValueError(f"{stage} requires {size} rows without recipe fallback")
    order = ordered_ids(view, seed=1729, lengths=lengths)[:size]
    warmup = [order[index % size] for index in range(head_updates * 16)]
    training = ([order[index % size] for index in range(64 * 16)]
                if stage == "diagnostic" else order)
    phase_orders = selected_schedule.compose({"warmup": warmup, "training": training})
    orders = {phase: phase_orders.get(phase, []) for phase in OPTIMIZER_PHASES}
    updates = {phase: (len(orders[phase]) + 15) // 16 for phase in OPTIMIZER_PHASES}
    return {**orders, "selection": order,
            "head_updates": updates["head"], "joint_updates": updates["joint"],
            "without_replacement_joint": stage != "diagnostic" and "joint" in selected_schedule.phases}
