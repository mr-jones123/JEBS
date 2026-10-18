"""Pure question-level scoring and scalar temperature calibration.

``score_predictions`` consumes already aligned prediction rows. Each row needs
id, group_id, probabilities (an ordered candidate-key mapping), target, source,
origin, modality, primitive, family, and ood_axis. Optional position_count is a
positive integer. The caller owns alignment against its requested dataset view;
this module cannot infer absent questions or candidates from predictions alone.

The report contains count, group_count, accuracy, nll, brier, ece, ece_bins,
high_confidence, risk_coverage, slices, uncertainty, and definitions. NLL and
Brier are question means; Brier sums squared errors over all candidates within
each question. A zero target probability has infinite NLL, without clipping.
Mapping insertion order breaks prediction ties. Probability mass is checked
with absolute tolerance 1e-6 and is never silently renormalized.

Each of the 15 ECE bins has lower, upper, count, accuracy, mean_confidence, and
gap. Bins are left-closed and right-open except the last, which includes 1.
Empty bins have null accuracy, mean_confidence, and gap. high_confidence has
threshold, count, errors, and error_rate (null for no selected questions).
Each risk_coverage point has threshold, count, coverage, errors, and risk;
all questions tied at a confidence enter together, using confidence >= threshold.

slices maps source, origin, modality, primitive, family, ood_axis,
candidate_count, and position_count to string-valued slices. Missing position
counts use the key "missing". Each slice has the same point metrics as the
whole report, without nested slices, uncertainty, or definitions.

uncertainty has method, group_key, resamples, seed, confidence_level,
quantile_method, and intervals. Each accuracy/nll/brier interval has lower and
upper. Groups are sampled uniformly with replacement, preserving every member
question and recomputing question-weighted means. These descriptive intervals
are not coverage guarantees or an abstention policy. No model or data is loaded.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


_PROBABILITY_TOLERANCE = 1e-6
_ECE_BIN_COUNT = 15
_LOG_TEMPERATURE_BOUNDS = (-8.0, 8.0)
_SLICE_FIELDS = ("source", "origin", "modality", "primitive", "family", "ood_axis")


@dataclass(frozen=True)
class _Question:
    group_id: str
    correct: bool
    nll: float
    brier: float
    confidence: float
    dimensions: tuple[str, ...]


def _real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _logit_rows(logits: Iterable[Sequence[float]]) -> list[np.ndarray]:
    try:
        raw_rows = list(logits)
    except TypeError as exc:
        raise ValueError("logits must be a sequence of candidate sequences") from exc
    rows = []
    for index, raw in enumerate(raw_rows):
        if isinstance(raw, (str, bytes, Mapping)):
            raise ValueError(f"logits[{index}] must be a candidate sequence")
        try:
            values = list(raw)
        except TypeError as exc:
            raise ValueError(f"logits[{index}] must be a candidate sequence") from exc
        if not values:
            raise ValueError(f"logits[{index}] must contain at least one candidate")
        rows.append(np.asarray([
            _real(value, f"logits[{index}][{candidate}]")
            for candidate, value in enumerate(values)
        ], dtype=np.float64))
    return rows


def _shifted_logits(row: np.ndarray, temperature: float) -> np.ndarray:
    # Center first to preserve small differences atop large common offsets.
    with np.errstate(over="ignore", under="ignore"):
        maximum = np.max(row)
        centered = row - maximum
        scaled = centered / temperature
        if temperature > 1 and np.isneginf(centered).any():
            # Opposite-sign finite extremes can overflow subtraction but not division.
            scaled = np.where(np.isfinite(centered), scaled, row / temperature - maximum / temperature)
        return scaled


def probabilities(
    logits: Iterable[Sequence[float]], temperature: float = 1.0
) -> list[list[float]]:
    """Return stable softmax for each unpadded, variable-cardinality logit row.

    All candidate logits must be finite. Remove masked padding before calling.
    Temperature must be finite and positive. An empty batch returns an empty
    list; an empty candidate row is invalid. This function never fits calibration.
    """
    temperature = _real(temperature, "temperature")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    result = []
    for row in _logit_rows(logits):
        with np.errstate(under="ignore"):
            weights = np.exp(_shifted_logits(row, temperature))
        result.append((weights / np.sum(weights)).tolist())
    return result


def fit_temperature(
    logits: Iterable[Sequence[float]], targets: Iterable[int]
) -> float:
    """Minimize question-mean calibration NLL over log temperature in [-8, 8].

    Targets are zero-based indices into the corresponding unpadded logit row.
    Only the supplied calibration observations enter the deterministic bounded
    scalar optimization. Identity and both endpoints are also evaluated, so the
    selected temperature never has higher objective NLL than temperature 1.
    Equal objective values prefer identity. An empty calibration set is invalid.
    """
    rows = _logit_rows(logits)
    try:
        targets = list(targets)
    except TypeError as exc:
        raise ValueError("targets must contain one index per logit row") from exc
    if not rows or len(rows) != len(targets):
        raise ValueError("calibration requires nonempty matching logits and targets")
    for index, (row, target) in enumerate(zip(rows, targets)):
        if isinstance(target, (bool, np.bool_)) or not isinstance(target, Integral):
            raise ValueError(f"targets[{index}] must be an integer candidate index")
        if not 0 <= target < len(row):
            raise ValueError(f"targets[{index}] is outside its candidate row")

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        losses = []
        for row, target in zip(rows, targets):
            with np.errstate(over="ignore", under="ignore"):
                scaled = _shifted_logits(row, temperature)
                log_normalizer = math.log(float(np.exp(scaled).sum()))
            losses.append(log_normalizer - float(scaled[target]))
        return math.fsum(loss / len(rows) for loss in losses)

    lower, upper = _LOG_TEMPERATURE_BOUNDS
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_loss, right_loss = objective(left), objective(right)
    for _ in range(96):
        if upper - lower <= 1e-10:
            break
        if left_loss <= right_loss:
            upper, right, right_loss = right, left, left_loss
            left = upper - ratio * (upper - lower)
            left_loss = objective(left)
        else:
            lower, left, left_loss = left, right, right_loss
            right = lower + ratio * (upper - lower)
            right_loss = objective(right)
    candidates = [0.0, *_LOG_TEMPERATURE_BOUNDS, left, right, (lower + upper) / 2]
    best = min(candidates, key=objective)
    return math.exp(best)


def _questions(rows: Iterable[Mapping[str, Any]]) -> list[_Question]:
    questions = []
    ids = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each prediction row must be a mapping")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("each prediction needs a nonempty string id")
        if identifier in ids:
            raise ValueError(f"duplicate prediction id {identifier!r}")
        ids.add(identifier)
        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(f"{identifier}: group_id must be a nonempty string")
        dimensions = []
        for field in _SLICE_FIELDS:
            value = row.get(field)
            if not isinstance(value, str):
                raise ValueError(f"{identifier}: {field} must be a string")
            dimensions.append(value)
        distribution = row.get("probabilities")
        if not isinstance(distribution, Mapping) or not distribution:
            raise ValueError(f"{identifier}: probabilities must be a nonempty mapping")
        keys, values = [], []
        for key, value in distribution.items():
            if not isinstance(key, str) or not key.strip() or key in keys:
                raise ValueError(f"{identifier}: candidate keys must be unique nonempty strings")
            keys.append(key)
            probability = _real(value, f"{identifier}: probability for {key!r}")
            if not 0 <= probability <= 1:
                raise ValueError(f"{identifier}: probabilities must be between 0 and 1")
            values.append(probability)
        if abs(math.fsum(values) - 1.0) > _PROBABILITY_TOLERANCE:
            raise ValueError(f"{identifier}: probabilities must sum to 1 within 1e-6")
        target = row.get("target")
        if not isinstance(target, str) or target not in keys:
            raise ValueError(f"{identifier}: target must be a candidate key")
        predicted_index = max(range(len(values)), key=values.__getitem__)
        target_index = keys.index(target)
        target_probability = values[target_index]
        brier = math.fsum(
            (value - int(index == target_index)) ** 2
            for index, value in enumerate(values)
        )
        position_count = row.get("position_count", "missing")
        if "position_count" in row:
            if (isinstance(position_count, (bool, np.bool_))
                    or not isinstance(position_count, Integral) or position_count <= 0):
                raise ValueError(f"{identifier}: position_count must be a positive integer")
        dimensions.extend((str(len(keys)), str(position_count)))
        questions.append(_Question(
            group_id=group_id,
            correct=predicted_index == target_index,
            nll=-math.log(target_probability) if target_probability else math.inf,
            brier=brier,
            confidence=values[predicted_index],
            dimensions=tuple(dimensions),
        ))
    if not questions:
        raise ValueError("scoring requires at least one prediction row")
    return questions


def _means(count: int, correct: float, nll: float, brier: float) -> dict[str, float]:
    return {"accuracy": correct / count, "nll": nll / count, "brier": brier / count}


def _summary(questions: Sequence[_Question]) -> dict[str, Any]:
    count = len(questions)
    means = _means(
        count, sum(question.correct for question in questions),
        math.fsum(question.nll for question in questions),
        math.fsum(question.brier for question in questions),
    )
    bins: list[list[_Question]] = [[] for _ in range(_ECE_BIN_COUNT)]
    confidence_groups: dict[float, list[_Question]] = defaultdict(list)
    high_count, high_errors = 0, 0
    for question in questions:
        bin_index = min(int(question.confidence * _ECE_BIN_COUNT), _ECE_BIN_COUNT - 1)
        bins[bin_index].append(question)
        confidence_groups[question.confidence].append(question)
        if question.confidence >= 0.95:
            high_count += 1
            high_errors += not question.correct
    ece_bins, contributions = [], []
    for index, members in enumerate(bins):
        accuracy = sum(question.correct for question in members) / len(members) if members else None
        confidence = math.fsum(question.confidence for question in members) / len(members) if members else None
        gap = abs(accuracy - confidence) if members else None
        if members:
            contributions.append(len(members) / count * gap)
        ece_bins.append({
            "lower": index / _ECE_BIN_COUNT, "upper": (index + 1) / _ECE_BIN_COUNT,
            "count": len(members), "accuracy": accuracy,
            "mean_confidence": confidence, "gap": gap,
        })
    risk_coverage, covered, errors = [], 0, 0
    for threshold in sorted(confidence_groups, reverse=True):
        members = confidence_groups[threshold]
        covered += len(members)
        errors += sum(not question.correct for question in members)
        risk_coverage.append({
            "threshold": threshold, "count": covered, "coverage": covered / count,
            "errors": errors, "risk": errors / covered,
        })
    return {
        "count": count, "group_count": len({question.group_id for question in questions}),
        **means, "ece": math.fsum(contributions), "ece_bins": ece_bins,
        "high_confidence": {
            "threshold": 0.95, "count": high_count, "errors": high_errors,
            "error_rate": high_errors / high_count if high_count else None,
        },
        "risk_coverage": risk_coverage,
    }


def _uncertainty(questions: Sequence[_Question]) -> dict[str, Any]:
    grouped: dict[str, list[_Question]] = defaultdict(list)
    for question in questions:
        grouped[question.group_id].append(question)
    totals = np.asarray([
        [len(members), sum(question.correct for question in members),
         math.fsum(question.nll for question in members),
         math.fsum(question.brier for question in members)]
        for _, members in sorted(grouped.items())
    ], dtype=np.float64)
    rng = np.random.default_rng(1729)
    samples: dict[str, list[float]] = {field: [] for field in ("accuracy", "nll", "brier")}
    for _ in range(1000):
        selected = rng.integers(len(totals), size=len(totals))
        count, correct, nll, brier = totals[selected].sum(axis=0)
        for field, value in _means(int(count), float(correct), float(nll), float(brier)).items():
            samples[field].append(value)
    intervals = {}
    for field, values in samples.items():
        values.sort()
        intervals[field] = {"lower": values[24], "upper": values[974]}
    return {
        "method": "percentile_group_bootstrap", "group_key": "group_id",
        "resamples": 1000, "seed": 1729, "confidence_level": 0.95,
        "quantile_method": "nearest_rank", "intervals": intervals,
    }


def score_predictions(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and score aligned keyed predictions without model or file access.

    The return schema and numerical conventions are documented at module level.
    All provided candidate probabilities participate in every question score.
    Slice metrics use precisely the same question scoring as the overall result.
    """
    questions = _questions(rows)
    fields = (*_SLICE_FIELDS, "candidate_count", "position_count")
    slices = {}
    for index, field in enumerate(fields):
        grouped: dict[str, list[_Question]] = defaultdict(list)
        for question in questions:
            grouped[question.dimensions[index]].append(question)
        slices[field] = {key: _summary(members) for key, members in sorted(grouped.items())}
    return {
        **_summary(questions), "slices": slices, "uncertainty": _uncertainty(questions),
        "definitions": {
            "unit": "question", "probability_sum_absolute_tolerance": _PROBABILITY_TOLERANCE,
            "prediction_ties": "first candidate in mapping insertion order",
            "nll": "mean negative natural log target probability; zero gives positive infinity",
            "brier": "question mean of sum over candidates of (probability - one_hot_target)^2",
            "ece": "15 equal-width bins; [lower, upper), last includes 1; question-weighted absolute gaps",
            "high_confidence": "confidence >= 0.95; descriptive, not an accuracy guarantee",
            "risk_coverage": "descending confidence >= threshold; exact confidence ties enter together",
            "uncertainty": "resample whole group_id groups with replacement; question-weighted means",
        },
    }
