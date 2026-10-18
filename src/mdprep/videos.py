"""Build deterministic procedural video decision examples."""

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import permutations, product
from pathlib import Path

from .common import (
    GENERATOR_VERSION,
    PART_QUOTAS,
    choice,
    noul,
    record,
    score,
    sha256_file,
    stable_id,
    write_json,
    write_jsonl,
)
from mdprep.video_media import _render_clip, _require_binary
from mdprep.video_specs import (
    FAMILY_CYCLE, FPS, FRAME_COUNT, FRAME_SIZE, OOD_FAMILY, PALETTE, SAMPLE_TIMES_S, SOURCE,
)





def _frame_order(variant_index: int) -> tuple[int, int, int, int]:
    if variant_index == 0:
        return (0, 1, 2, 3)
    if variant_index == 1:
        return (3, 2, 1, 0)
    if variant_index == 2:
        return (1, 2, 3, 0)
    raise ValueError(f"Unsupported trajectory variant: {variant_index}")


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _direction_changes(deltas: list[float]) -> int:
    signs = [_sign(delta) for delta in deltas if delta]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def _motion_pattern(deltas: list[float]) -> str:
    signs = [_sign(delta) for delta in deltas]
    if signs == [1, 1, 1]:
        return "right_only"
    if signs == [-1, -1, -1]:
        return "left_only"
    if signs[0] == 1 and signs[-1] == -1 and _direction_changes(deltas) == 1:
        return "right_then_left"
    if signs[0] == -1 and signs[-1] == 1 and _direction_changes(deltas) == 1:
        return "left_then_right"
    if signs == [1, -1, 1]:
        return "alternating_right_first"
    if signs == [-1, 1, -1]:
        return "alternating_left_first"
    raise ValueError(f"Unrecognized horizontal trajectory: {deltas}")
def _sequence_orbit(sequence: tuple) -> tuple[tuple, ...]:
    reversed_sequence = tuple(reversed(sequence))
    variants = []
    for candidate in (sequence, reversed_sequence):
        variants.extend(
            candidate[offset:] + candidate[:offset]
            for offset in range(len(candidate))
        )
    return tuple(variants)


def _round_robin_buckets(buckets: dict[object, list[tuple]]) -> tuple[tuple, ...]:
    ordered_keys = sorted(buckets, key=str)
    positions = {key: 0 for key in ordered_keys}
    output = []
    while True:
        added = False
        for key in ordered_keys:
            position = positions[key]
            values = buckets[key]
            if position < len(values):
                output.append(values[position])
                positions[key] += 1
                added = True
        if not added:
            return tuple(output)


@lru_cache(maxsize=1)
def _bounce_trajectories() -> tuple[tuple[int, int, int, int], ...]:
    buckets: dict[str, list[tuple]] = {}
    for trajectory in permutations(range(34, 222, 17), FRAME_COUNT):
        if trajectory != min(_sequence_orbit(trajectory)):
            continue
        deltas = [right - left for left, right in zip(trajectory, trajectory[1:])]
        pattern = _motion_pattern(deltas)
        buckets.setdefault(pattern, []).append(trajectory)
    return _round_robin_buckets(buckets)


@lru_cache(maxsize=1)
def _color_trajectories() -> tuple[tuple[str, str, str, str], ...]:
    color_names = tuple(PALETTE)
    buckets: dict[tuple[int, bool], list[tuple]] = {}
    for trajectory in product(color_names, repeat=FRAME_COUNT):
        if len(set(_sequence_orbit(trajectory))) != 8 or trajectory != min(_sequence_orbit(trajectory)):
            continue
        changes = sum(left != right for left, right in zip(trajectory, trajectory[1:]))
        returned = _has_color_return(list(trajectory))
        buckets.setdefault((changes, returned), []).append(trajectory)
    return _round_robin_buckets(buckets)


def _gate_positions(event_index: int, serial: int, lane: str) -> tuple[int, int, int, int]:
    raw = bytes.fromhex(stable_id(SOURCE, "gate-position", serial, lane))
    positions = []
    for index in range(FRAME_COUNT):
        if index == event_index:
            positions.append(128)
        elif index < event_index:
            positions.append(24 + raw[index] % 88)
        else:
            positions.append(145 + raw[index] % 87)
    return tuple(positions)


@lru_cache(maxsize=1)
def _gate_trajectories() -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    trajectories = []
    seen = set()
    serial = 0
    while len(trajectories) < 512 and serial < 100_000:
        amber_event = serial % FRAME_COUNT
        violet_event = (serial // FRAME_COUNT) % FRAME_COUNT
        amber_xs = _gate_positions(amber_event, serial, "amber")
        violet_xs = _gate_positions(violet_event, serial, "violet")
        paired = tuple(zip(amber_xs, violet_xs))
        canonical = min(_sequence_orbit(paired))
        if canonical not in seen:
            seen.add(canonical)
            trajectories.append((amber_xs, violet_xs))
        serial += 1
    if len(trajectories) < 512:
        raise RuntimeError("Could not allocate enough unique center-guide trajectories")
    return tuple(trajectories)


def _transform_collision_points(
    points: tuple[tuple[int, int], ...],
    angle_degrees: int,
    translate_x: int,
    translate_y: int,
) -> tuple[tuple[int, int], ...]:
    if angle_degrees % 90:
        raise ValueError("Collision rotations must preserve integer distance geometry")
    turns = (angle_degrees // 90) % 4
    transformed = []
    for x, y in points:
        x, y = x - 128, y - 96
        for _ in range(turns):
            x, y = -y, x
        transformed.append((128 + x + translate_x, 96 + y + translate_y))
    return tuple(transformed)


@lru_cache(maxsize=1)
def _collision_trajectories() -> tuple[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]], ...]:
    patterns = (
        (((45, 70), (85, 70), (120, 70), (165, 70)), ((211, 122), (171, 122), (140, 122), (95, 122))),
        (((45, 96), (90, 96), (120, 96), (170, 96)), ((211, 96), (166, 96), (136, 96), (86, 96))),
        (((45, 96), (110, 96), (120, 96), (170, 96)), ((211, 96), (138, 96), (136, 96), (86, 96))),
        (((100, 96), (112, 96), (120, 96), (170, 96)), ((128, 96), (140, 96), (136, 96), (86, 96))),
    )
    trajectories = []
    seen = set()
    serial = 0
    while len(trajectories) < 512 and serial < 100_000:
        amber_base, violet_base = patterns[serial % len(patterns)]
        variation = serial // len(patterns)
        angle = (variation % 2) * 180
        translate_x = ((variation // 2) % 17) - 8
        translate_y = ((variation // 34) % 17) - 8
        amber_points = _transform_collision_points(amber_base, angle, translate_x, translate_y)
        violet_points = _transform_collision_points(violet_base, angle, translate_x, translate_y)
        paired = tuple(zip(amber_points, violet_points))
        canonical = min(_sequence_orbit(paired))
        if canonical not in seen:
            seen.add(canonical)
            trajectories.append((amber_points, violet_points))
        serial += 1
    if len(trajectories) < 512:
        raise RuntimeError("Could not allocate enough unique collision trajectories")
    return tuple(trajectories)


def _bounce_scene(pattern_index: int, order: tuple[int, ...]) -> dict:
    trajectories = _bounce_trajectories()
    if pattern_index >= len(trajectories):
        raise RuntimeError(f"Bounce trajectory allocation exhausted at {pattern_index}")
    base_xs = trajectories[pattern_index]
    xs = [base_xs[index] for index in order]
    deltas = [right - left for left, right in zip(xs, xs[1:])]
    changes = _direction_changes(deltas)
    pattern = _motion_pattern(deltas)
    facts = []
    for index, x in enumerate(xs):
        facts.append({
            "sample_index": index,
            "time_s": SAMPLE_TIMES_S[index],
            "focus_center": [x, 96],
            "direction_from_previous": None if index == 0 else ("right" if deltas[index - 1] > 0 else "left"),
        })
    first_direction = "right" if deltas[0] > 0 else "left"
    final_direction = "right" if deltas[-1] > 0 else "left"
    questions = [
        {
            "family": "bounce.motion_pattern",
            "question": choice(
                "Which description matches the amber circle's complete horizontal motion?",
                {
                    "right_only": "It moved right throughout.",
                    "left_only": "It moved left throughout.",
                    "right_then_left": "It moved right, then reversed left.",
                    "left_then_right": "It moved left, then reversed right.",
                    "alternating_right_first": "It moved right, then left, then right.",
                    "alternating_left_first": "It moved left, then right, then left.",
                },
            ),
            "label": pattern,
            "operation": "classify the signs of all three consecutive horizontal displacements",
            "derived": {"horizontal_deltas": deltas, "motion_pattern": pattern},
        },
        {
            "family": "bounce.any_reversal",
            "question": noul("Did the amber circle reverse horizontal direction before the clip ended?"),
            "label": str(changes > 0).lower(),
            "operation": "detect any sign change between consecutive horizontal displacements",
            "derived": {"direction_change_count": changes},
        },
        {
            "family": "bounce.reversal_count",
            "question": score(
                "How many horizontal direction changes occurred across the three intervals?",
                ["No direction changes", "One direction change", "Two direction changes"],
            ),
            "label": str(changes),
            "operation": "count sign changes in consecutive horizontal displacements",
            "derived": {"direction_change_count": changes},
        },
        {
            "family": "bounce.endpoint_intervals",
            "question": choice(
                "Which directions describe the first interval and the final interval, in that order?",
                {
                    "right_right": "Right, then right",
                    "right_left": "Right, then left",
                    "left_right": "Left, then right",
                    "left_left": "Left, then left",
                },
            ),
            "label": f"{first_direction}_{final_direction}",
            "operation": "compare horizontal displacement in the first and final intervals",
            "derived": {"first_interval": first_direction, "final_interval": final_direction},
        },
    ]
    return {
        "family": "bounce",
        "render": {"kind": "bounce", "xs": xs},
        "frame_facts": facts,
        "events": {"horizontal_direction_change_count": changes},
        "questions": questions,
    }


def _has_color_return(colors: list[str]) -> bool:
    for left in range(len(colors)):
        for right in range(left + 2, len(colors)):
            if colors[left] == colors[right] and any(
                color != colors[left] for color in colors[left + 1:right]
            ):
                return True
    return False


def _color_scene(pattern_index: int, order: tuple[int, ...]) -> dict:
    trajectories = _color_trajectories()
    if pattern_index >= len(trajectories):
        raise RuntimeError(f"Color trajectory allocation exhausted at {pattern_index}")
    base_colors = trajectories[pattern_index]
    colors = [base_colors[index] for index in order]
    changes = sum(left != right for left, right in zip(colors, colors[1:]))
    returned = _has_color_return(colors)
    final_color_first_index = colors.index(colors[-1])
    facts = [
        {"sample_index": index, "time_s": SAMPLE_TIMES_S[index], "focus_color": color}
        for index, color in enumerate(colors)
    ]
    color_candidates = {color: color.capitalize() for color in PALETTE}
    questions = [
        {
            "family": "color.penultimate",
            "question": choice("Which color was visible immediately before the final sampled color?", color_candidates),
            "label": colors[-2],
            "operation": "select the color at the penultimate sampled time",
            "derived": {"penultimate_color": colors[-2]},
        },
        {
            "family": "color.return",
            "question": noul(
                "After changing away from a color, did the circle ever return to that earlier color?"
            ),
            "label": str(returned).lower(),
            "operation": "detect a repeated color separated by at least one different color",
            "derived": {"returned_to_earlier_color": returned},
        },
        {
            "family": "color.change_count",
            "question": score(
                "How many color changes occurred between consecutive sampled frames?",
                ["No changes", "One change", "Two changes", "Three changes"],
            ),
            "label": str(changes),
            "operation": "count unequal adjacent colors",
            "derived": {"color_change_count": changes},
        },
        {
            "family": "color.final_first_seen",
            "question": choice(
                "At which sampled frame did the final color first appear?",
                {
                    "frame_1": "First sampled frame",
                    "frame_2": "Second sampled frame",
                    "frame_3": "Third sampled frame",
                    "frame_4": "Fourth sampled frame",
                },
            ),
            "label": f"frame_{final_color_first_index + 1}",
            "operation": "find the earliest occurrence of the final sampled color",
            "derived": {"final_color": colors[-1], "first_seen_index": final_color_first_index},
        },
    ]
    return {
        "family": "color_sequence",
        "render": {"kind": "color_sequence", "colors": colors},
        "frame_facts": facts,
        "events": {"color_change_count": changes, "returned_to_earlier_color": returned},
        "questions": questions,
    }




def _gate_order_scene(pattern_index: int, order: tuple[int, ...]) -> dict:
    trajectories = _gate_trajectories()
    if pattern_index >= len(trajectories):
        raise RuntimeError(f"Gate trajectory allocation exhausted at {pattern_index}")
    amber_base, violet_base = trajectories[pattern_index]
    amber_xs = [amber_base[index] for index in order]
    violet_xs = [violet_base[index] for index in order]
    amber_center = amber_xs.index(128)
    violet_center = violet_xs.index(128)
    if amber_center < violet_center:
        first = "amber_first"
    elif violet_center < amber_center:
        first = "violet_first"
    else:
        first = "same_time"
    gap = abs(amber_center - violet_center)
    amber_on_second = amber_xs[1] == 128
    violet_on_second = violet_xs[1] == 128
    if amber_on_second and violet_on_second:
        second_frame = "both"
    elif amber_on_second:
        second_frame = "amber_only"
    elif violet_on_second:
        second_frame = "violet_only"
    else:
        second_frame = "neither"
    facts = []
    for index, (amber_x, violet_x) in enumerate(zip(amber_xs, violet_xs)):
        facts.append({
            "sample_index": index,
            "time_s": SAMPLE_TIMES_S[index],
            "amber_circle_center": [amber_x, 70],
            "violet_square_center": [violet_x, 126],
            "amber_on_center_guide": amber_x == 128,
            "violet_on_center_guide": violet_x == 128,
        })
    questions = [
        {
            "family": "gate.first_arrival",
            "question": choice(
                "Which object reached the center guide first?",
                {
                    "amber_first": "The amber circle",
                    "violet_first": "The violet square",
                    "same_time": "They reached it at the same sampled time",
                },
            ),
            "label": first,
            "operation": "compare the first center-guide sample index for both objects",
            "derived": {"amber_index": amber_center, "violet_index": violet_center, "first": first},
        },
        {
            "family": "gate.amber_before_violet",
            "question": noul("Did the amber circle reach the center guide before the violet square?"),
            "label": str(amber_center < violet_center).lower(),
            "operation": "test whether amber's center-guide index is less than violet's",
            "derived": {"amber_index": amber_center, "violet_index": violet_center},
        },
        {
            "family": "gate.arrival_gap",
            "question": score(
                "How many sampled-frame steps separated the two center-guide arrivals?",
                ["Same sampled frame", "One frame step", "Two frame steps", "Three frame steps"],
            ),
            "label": str(gap),
            "operation": "take the absolute difference between center-guide sample indices",
            "derived": {"arrival_gap_frames": gap},
        },
        {
            "family": "gate.second_frame",
            "question": choice(
                "At the second sampled frame, which object was exactly on the center guide?",
                {
                    "amber_only": "Only the amber circle",
                    "violet_only": "Only the violet square",
                    "both": "Both objects",
                    "neither": "Neither object",
                },
            ),
            "label": second_frame,
            "operation": "inspect both center coordinates at the second sampled time",
            "derived": {"second_frame_center_occupancy": second_frame},
        },
    ]
    return {
        "family": "gate_order",
        "render": {"kind": "gate_order", "amber_xs": amber_xs, "violet_xs": violet_xs},
        "frame_facts": facts,
        "events": {"amber_center_index": amber_center, "violet_center_index": violet_center},
        "questions": questions,
    }


def _collision_scene(pattern_index: int, order: tuple[int, ...]) -> dict:
    trajectories = _collision_trajectories()
    if pattern_index >= len(trajectories):
        raise RuntimeError(f"Collision trajectory allocation exhausted at {pattern_index}")
    amber_base, violet_base = trajectories[pattern_index]
    amber_points = [amber_base[index] for index in order]
    violet_points = [violet_base[index] for index in order]
    distances = [
        math.dist(amber_point, violet_point)
        for amber_point, violet_point in zip(amber_points, violet_points)
    ]
    contacts = [distance <= 32.0 for distance in distances]
    contact_count = sum(contacts)
    if not any(contacts):
        after_contact = "no_contact"
    else:
        first_contact = contacts.index(True)
        after_contact = "separated" if False in contacts[first_contact + 1:] else "remained_in_contact"
    trends = []
    for left, right in zip(distances, distances[1:]):
        if right < left:
            trends.append("closer")
        elif right > left:
            trends.append("farther")
        else:
            trends.append("unchanged")
    if "unchanged" in trends[:2]:
        early_trend = "includes_no_change"
    else:
        early_trend = f"{trends[0]}_then_{trends[1]}"
    facts = []
    for index, (amber_point, violet_point, distance, contact) in enumerate(
        zip(amber_points, violet_points, distances, contacts)
    ):
        facts.append({
            "sample_index": index,
            "time_s": SAMPLE_TIMES_S[index],
            "amber_center": list(amber_point),
            "violet_center": list(violet_point),
            "center_distance_px": round(distance, 4),
            "touching_or_overlapping": contact,
        })
    questions = [
        {
            "family": "collision.after_first_contact",
            "question": choice(
                "What happened after the circles first touched or overlapped?",
                {
                    "separated": "They later separated.",
                    "remained_in_contact": "They remained touching or overlapping through the end.",
                    "no_contact": "They never touched or overlapped.",
                },
            ),
            "label": after_contact,
            "operation": "find first contact and inspect all later contact states",
            "derived": {"contact_sequence": contacts, "outcome_after_first_contact": after_contact},
        },
        {
            "family": "collision.before_final",
            "question": noul("Did the two circles touch or overlap before the final sampled frame?"),
            "label": str(any(contacts[:-1])).lower(),
            "operation": "test for contact in the first three sampled frames",
            "derived": {"pre_final_contact": any(contacts[:-1])},
        },
        {
            "family": "collision.contact_count",
            "question": score(
                "In how many sampled frames were the two circles touching or overlapping?",
                ["Zero frames", "One frame", "Two frames", "Three frames"],
            ),
            "label": str(contact_count),
            "operation": "count center distances no greater than the sum of radii",
            "derived": {"contact_threshold_px": 32, "contact_count": contact_count},
        },
        {
            "family": "collision.early_distance_trend",
            "question": choice(
                "Across the first two intervals, how did the distance between the circle centers change?",
                {
                    "closer_then_closer": "Closer, then closer again",
                    "closer_then_farther": "Closer, then farther",
                    "farther_then_closer": "Farther, then closer",
                    "farther_then_farther": "Farther, then farther again",
                    "includes_no_change": "At least one interval had no distance change",
                },
            ),
            "label": early_trend,
            "operation": "compare center distances over the first two consecutive intervals",
            "derived": {"center_distances_px": [round(value, 4) for value in distances], "early_trend": early_trend},
        },
    ]
    return {
        "family": "collision",
        "render": {"kind": "collision", "amber_points": amber_points, "violet_points": violet_points},
        "frame_facts": facts,
        "events": {"contact_sequence": contacts, "contact_count": contact_count},
        "questions": questions,
    }


@lru_cache(maxsize=1)
def _zigzag_trajectories() -> tuple[tuple[tuple[int, int], ...], ...]:
    patterns = (
        ((30, 35), (96, 55), (150, 96), (210, 145)),
        ((35, 30), (60, 96), (96, 145), (180, 160)),
        ((30, 30), (96, 96), (150, 55), (205, 130)),
        ((205, 145), (150, 96), (96, 55), (30, 35)),
        ((30, 150), (96, 130), (145, 96), (205, 135)),
        ((35, 160), (60, 96), (96, 48), (180, 120)),
    )
    required_count = len({group for group, _ in _group_layout(PART_QUOTAS["videos"]["test_ood"] // 4)})
    trajectories = []
    seen = set()
    serial = 0
    while len(trajectories) < required_count and serial < 100_000:
        base = patterns[serial % len(patterns)]
        variation = serial // len(patterns)
        candidate = []
        for index, (x, y) in enumerate(base):
            x_offset = ((variation + 1) * (index + 3) * 5) % 13 - 6
            y_offset = ((variation + 1) * (index + 5) * 7) % 13 - 6
            adjusted_x = x if x == 96 else min(230, max(26, x + x_offset))
            adjusted_y = y if y == 96 else min(166, max(26, y + y_offset))
            if adjusted_x == 96 and x != 96:
                adjusted_x += 1
            if adjusted_y == 96 and y != 96:
                adjusted_y += 1
            candidate.append((adjusted_x, adjusted_y))
        trajectory = tuple(candidate)
        cyclic_y = [point[1] for point in trajectory + trajectory[:1]]
        if all(left != right for left, right in zip(cyclic_y, cyclic_y[1:])):
            canonical = min(_sequence_orbit(trajectory))
            if canonical not in seen:
                seen.add(canonical)
                trajectories.append(trajectory)
        serial += 1
    if len(trajectories) < required_count:
        raise RuntimeError("Could not allocate enough unique OOD zigzag trajectories")
    return tuple(trajectories)


def _zigzag_scene(pattern_index: int, order: tuple[int, ...]) -> dict:
    trajectories = _zigzag_trajectories()
    if pattern_index >= len(trajectories):
        raise RuntimeError(f"Zigzag trajectory allocation exhausted at {pattern_index}")
    base_points = trajectories[pattern_index]
    points = [base_points[index] for index in order]
    cyan_indices = [index for index, (x, _) in enumerate(points) if x == 96]
    magenta_indices = [index for index, (_, y) in enumerate(points) if y == 96]
    if not cyan_indices or not magenta_indices:
        raise ValueError("OOD zigzag must intersect both guides")
    cyan_first = cyan_indices[0]
    magenta_first = magenta_indices[0]
    if cyan_first < magenta_first:
        guide_order = "cyan_first"
    elif magenta_first < cyan_first:
        guide_order = "magenta_first"
    else:
        guide_order = "same_time"
    y_deltas = [right[1] - left[1] for left, right in zip(points, points[1:])]
    vertical_reversals = _direction_changes(y_deltas)
    first_vertical = "down" if y_deltas[0] > 0 else "up"
    final_vertical = "down" if y_deltas[-1] > 0 else "up"
    facts = []
    for index, point in enumerate(points):
        facts.append({
            "sample_index": index,
            "time_s": SAMPLE_TIMES_S[index],
            "triangle_center": list(point),
            "on_cyan_vertical_guide": point[0] == 96,
            "on_magenta_horizontal_guide": point[1] == 96,
        })
    questions = [
        {
            "family": "zigzag.guide_order",
            "question": choice(
                "Which colored guide did the triangle intersect first?",
                {
                    "cyan_first": "The cyan vertical guide",
                    "magenta_first": "The magenta horizontal guide",
                    "same_time": "Both guides at the same sampled time",
                },
            ),
            "label": guide_order,
            "operation": "compare earliest exact-intersection indices for the two orthogonal guides",
            "derived": {"cyan_first_index": cyan_first, "magenta_first_index": magenta_first},
        },
        {
            "family": "zigzag.vertical_reversal",
            "question": noul("Did the triangle reverse its vertical direction during the clip?"),
            "label": str(vertical_reversals > 0).lower(),
            "operation": "detect a sign change in consecutive vertical displacements",
            "derived": {"vertical_deltas": y_deltas, "vertical_reversal_count": vertical_reversals},
        },
        {
            "family": "zigzag.vertical_reversal_count",
            "question": score(
                "How many times did the triangle reverse vertical direction?",
                ["No reversals", "One reversal", "Two reversals"],
            ),
            "label": str(vertical_reversals),
            "operation": "count sign changes in consecutive vertical displacements",
            "derived": {"vertical_reversal_count": vertical_reversals},
        },
        {
            "family": "zigzag.endpoint_vertical_motion",
            "question": choice(
                "Which vertical directions describe the first interval and final interval, in that order?",
                {
                    "down_down": "Down, then down",
                    "down_up": "Down, then up",
                    "up_down": "Up, then down",
                    "up_up": "Up, then up",
                },
            ),
            "label": f"{first_vertical}_{final_vertical}",
            "operation": "compare vertical displacement in the first and final intervals",
            "derived": {"first_interval": first_vertical, "final_interval": final_vertical},
        },
    ]
    return {
        "family": OOD_FAMILY,
        "render": {"kind": OOD_FAMILY, "points": points},
        "frame_facts": facts,
        "events": {
            "cyan_intersection_indices": cyan_indices,
            "magenta_intersection_indices": magenta_indices,
            "vertical_reversal_count": vertical_reversals,
        },
        "questions": questions,
    }


def _decorations(group_id: str) -> list[dict]:
    raw = bytes.fromhex(group_id)
    colors = ("#D6CFC1", "#C8D5D0", "#D8CAD1")
    return [
        {
            "center": [24 + raw[index * 2] % 208, 18 + raw[index * 2 + 1] % 10],
            "color": colors[raw[index * 2] % len(colors)],
        }
        for index in range(4)
    ]


def _make_scene(family: str, pattern_index: int, variant_index: int, group_id: str) -> dict:
    order = _frame_order(variant_index)
    if family == "bounce":
        scene = _bounce_scene(pattern_index, order)
    elif family == "color_sequence":
        scene = _color_scene(pattern_index, order)
    elif family == "gate_order":
        scene = _gate_order_scene(pattern_index, order)
    elif family == "collision":
        scene = _collision_scene(pattern_index, order)
    elif family == OOD_FAMILY:
        scene = _zigzag_scene(pattern_index, order)
    else:
        raise ValueError(f"Unknown scene family: {family}")
    if variant_index == 0:
        variant_transform = "forward"
    elif variant_index == 1:
        variant_transform = "time_reversed"
    else:
        variant_transform = "cyclic_sample_counterfactual"
    scene["variant_transform"] = variant_transform
    scene["render"]["decorations"] = _decorations(group_id)
    scene["render_signature"] = stable_id(SOURCE, scene["render"], "h264-lossless-rgb-v1")
    scene["trajectory_index"] = pattern_index
    return scene




def _group_layout(clip_count: int) -> list[tuple[int, int]]:
    if clip_count % 2 == 0:
        return [(clip_index // 2, clip_index % 2) for clip_index in range(clip_count)]
    if clip_count < 3:
        raise ValueError("Odd clip counts must permit one three-variant trajectory group")
    layout = [(clip_index // 2, clip_index % 2) for clip_index in range(clip_count - 3)]
    final_group = (clip_count - 3) // 2
    layout.extend((final_group, variant_index) for variant_index in range(3))
    return layout


def _clip_jobs(root: Path) -> list[dict]:
    jobs = []
    trajectory_indices: dict[tuple[str, str, int], int] = {}
    next_trajectory_index: Counter[str] = Counter()
    for split, question_quota in PART_QUOTAS["videos"].items():
        if question_quota % 4:
            raise ValueError(f"Video question quota is not divisible by four: {split}={question_quota}")
        clip_count = question_quota // 4
        for group_index, variant_index in _group_layout(clip_count):
            family = OOD_FAMILY if split == "test_ood" else FAMILY_CYCLE[group_index % len(FAMILY_CYCLE)]
            trajectory_key = (split, family, group_index)
            if trajectory_key not in trajectory_indices:
                trajectory_indices[trajectory_key] = next_trajectory_index[family]
                next_trajectory_index[family] += 1
            trajectory_index = trajectory_indices[trajectory_key]
            group_id = stable_id(SOURCE, split, family, trajectory_index)
            clip_id = stable_id(SOURCE, group_id, "clip", variant_index)
            relative_path = f"media/videos/{clip_id}.mp4"
            scene = _make_scene(family, trajectory_index, variant_index, group_id)
            jobs.append({
                "split": split,
                "group_id": group_id,
                "trajectory_index": trajectory_index,
                "variant_index": variant_index,
                "relative_path": relative_path,
                "absolute_path": root / "data" / relative_path,
                "scene": scene,
            })
    return jobs


def prepare(root: Path) -> dict:
    """Render all clips and write the canonical video JSONL part."""
    root = root.resolve()
    ffmpeg = _require_binary(Path("/opt/homebrew/bin/ffmpeg"), "ffmpeg")
    ffprobe = _require_binary(Path("/opt/homebrew/bin/ffprobe"), "ffprobe")
    jobs = _clip_jobs(root)
    worker_count = min(4, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        rendered = list(executor.map(lambda job: _render_clip(job, ffmpeg, ffprobe), jobs))

    rows = []
    source_assets = []
    primitive_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    provenance = {
        "source_url": "local://src/mdprep/videos.py",
        "revision": GENERATOR_VERSION,
        "license": "Project-created procedural data; use is governed by repository data terms",
        "license_url": None,
        "attribution": "Generated locally by the multimodal-decision video builder",
        "project_created_data_terms": "Original project-created procedural media; no third-party media assets",
        "font_terms": "No fonts, text glyphs, or third-party font assets are used in the rendered clips",
    }
    for job, render_result in zip(jobs, rendered):
        metadata = render_result["metadata"]
        video = {"path": job["relative_path"], **metadata}
        scene = job["scene"]
        source_assets.append({
            "path": job["relative_path"],
            "sha256": render_result["sha256"],
            "split": job["split"],
            "group_id": job["group_id"],
            "scene_family": scene["family"],
            "trajectory_index": scene["trajectory_index"],
            "variant_transform": scene["variant_transform"],
            **metadata,
        })
        for question_index, question_spec in enumerate(scene["questions"]):
            question = question_spec["question"]
            audit = {
                "scene_family": scene["family"],
                "trajectory_index": scene["trajectory_index"],
                "variant_transform": scene["variant_transform"],
                "render_signature": scene["render_signature"],
                "sample_times_s": list(SAMPLE_TIMES_S),
                "frame_facts": scene["frame_facts"],
                "events": scene["events"],
                "decision": {
                    "operation": question_spec["operation"],
                    "derived": question_spec["derived"],
                    "answer": question_spec["label"],
                },
                "media_sha256": render_result["sha256"],
            }
            rows.append(record(
                split=job["split"],
                source=SOURCE,
                group_id=job["group_id"],
                family=question_spec["family"],
                modality="video",
                state=(
                    "Observe the entire four-frame clip in time order. Changes between consecutive "
                    "sampled frames are evidence; do not answer from the final frame alone."
                ),
                question=question,
                label=question_spec["label"],
                videos=[video],
                provenance=provenance,
                audit=audit,
                variant=(job["variant_index"], question_index),
            ))
            primitive_counts[question["type"]] += 1
            family_counts[scene["family"]] += 1
            split_counts[job["split"]] += 1

    expected_total = sum(PART_QUOTAS["videos"].values())
    if len(rows) != expected_total or dict(split_counts) != PART_QUOTAS["videos"]:
        raise RuntimeError(
            f"Generated video quotas do not match contract: total={len(rows)}, splits={dict(split_counts)}"
        )
    write_jsonl(root / "data" / "parts" / "videos.jsonl", rows)
    source_metadata = {
        "source": SOURCE,
        "source_url": "local://src/mdprep/videos.py",
        "revision": GENERATOR_VERSION,
        "upstream_split": None,
        "license": "Project-created procedural data; use is governed by repository data terms",
        "license_url": None,
        "attribution": "Generated locally by the multimodal-decision video builder",
        "project_created_data_terms": "Original project-created procedural media; no third-party media assets",
        "font_terms": "No fonts, text glyphs, or third-party font assets are used in the rendered clips",
        "generation": {
            "generator": "mdprep.videos",
            "frame_size": list(FRAME_SIZE),
            "fps": FPS,
            "frame_count": FRAME_COUNT,
            "sample_times_s": list(SAMPLE_TIMES_S),
            "ffmpeg": ffmpeg,
            "ffprobe": ffprobe,
            "encoding": "RGB-preserving H.264, libx264rgb, CRF 0, rgb24 input, one encoder thread per clip",
            "deterministic_ids": True,
        },
        "grouping": (
            "Forward, time-reversed, and where needed cyclic-sample-counterfactual variants share "
            "a group_id and therefore a split."
        ),
        "ood_policy": {
            "held_out_family": OOD_FAMILY,
            "split": "test_ood",
            "description": "Two-dimensional triangle trajectories crossing orthogonal colored guides",
        },
        "assets": source_assets,
    }
    write_json(root / "data" / "sources" / "videos.json", source_metadata)
    return {
        "source": SOURCE,
        "revision": GENERATOR_VERSION,
        "records": len(rows),
        "generated_assets": len(source_assets),
        "split_counts": dict(split_counts),
        "primitive_counts": dict(primitive_counts),
        "scene_family_question_counts": dict(family_counts),
        "known_limitations": [
            "Clips contain four deliberately sampled frames at 2 fps rather than continuous high-frame-rate motion.",
            "Scenes are procedural geometric evidence and do not model natural-video appearance or camera motion.",
        ],
        "invocation": f"PYTHONPATH={root / 'src'} python -m mdprep.videos --root {root}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Project root containing data/")
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.root), indent=2))


if __name__ == "__main__":
    main()
