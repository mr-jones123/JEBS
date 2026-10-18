"""Temporal qualification must bind the decision shown to the model."""

import copy
import unittest

from mdprep.verify_video import temporal_answer, video_answer


_STATE = (
    "Observe the entire four-frame clip in time order. Changes between consecutive "
    "sampled frames are evidence; do not answer from the final frame alone."
)
_YES_NO = (("false", "No"), ("true", "Yes"))
_FACTS = {
    "bounce": [
        {"focus_center": (30, 96)}, {"focus_center": (80, 96)},
        {"focus_center": (140, 96)}, {"focus_center": (60, 96)},
    ],
    "color_sequence": [
        {"focus_color": "red"}, {"focus_color": "blue"},
        {"focus_color": "red"}, {"focus_color": "green"},
    ],
    "gate_order": [
        {"amber_circle_center": (30, 70), "violet_square_center": (40, 126)},
        {"amber_circle_center": (128, 70), "violet_square_center": (80, 126)},
        {"amber_circle_center": (180, 70), "violet_square_center": (128, 126)},
        {"amber_circle_center": (210, 70), "violet_square_center": (190, 126)},
    ],
    "collision": [
        {"amber_center": (40, 88), "violet_center": (140, 88)},
        {"amber_center": (80, 88), "violet_center": (100, 88)},
        {"amber_center": (140, 88), "violet_center": (80, 88)},
        {"amber_center": (200, 88), "violet_center": (40, 88)},
    ],
    "zigzag_guides": [
        {"triangle_center": (30, 35)}, {"triangle_center": (96, 55)},
        {"triangle_center": (150, 96)}, {"triangle_center": (210, 45)},
    ],
}

# Each case states a model-visible decision and its answer from literal frame facts.
_CASES = (
    ("bounce.motion_pattern", "bounce", "choice",
     "Which description matches the amber circle's complete horizontal motion?",
     (("right_only", "It moved right throughout."), ("left_only", "It moved left throughout."),
      ("right_then_left", "It moved right, then reversed left."),
      ("left_then_right", "It moved left, then reversed right."),
      ("alternating_right_first", "It moved right, then left, then right."),
      ("alternating_left_first", "It moved left, then right, then left.")), "right_then_left"),
    ("bounce.any_reversal", "bounce", "noul",
     "Did the amber circle reverse horizontal direction before the clip ended?", _YES_NO, "true"),
    ("bounce.reversal_count", "bounce", "score",
     "How many horizontal direction changes occurred across the three intervals?",
     (("0", "No direction changes"), ("1", "One direction change"), ("2", "Two direction changes")), "1"),
    ("bounce.endpoint_intervals", "bounce", "choice",
     "Which directions describe the first interval and the final interval, in that order?",
     (("right_right", "Right, then right"), ("right_left", "Right, then left"),
      ("left_right", "Left, then right"), ("left_left", "Left, then left")), "right_left"),
    ("color.penultimate", "color_sequence", "choice",
     "Which color was visible immediately before the final sampled color?",
     (("amber", "Amber"), ("blue", "Blue"), ("cyan", "Cyan"), ("green", "Green"),
      ("magenta", "Magenta"), ("red", "Red"), ("violet", "Violet"), ("yellow", "Yellow")), "red"),
    ("color.return", "color_sequence", "noul",
     "After changing away from a color, did the circle ever return to that earlier color?", _YES_NO, "true"),
    ("color.change_count", "color_sequence", "score",
     "How many color changes occurred between consecutive sampled frames?",
     (("0", "No changes"), ("1", "One change"), ("2", "Two changes"), ("3", "Three changes")), "3"),
    ("color.final_first_seen", "color_sequence", "choice",
     "At which sampled frame did the final color first appear?",
     (("frame_1", "First sampled frame"), ("frame_2", "Second sampled frame"),
      ("frame_3", "Third sampled frame"), ("frame_4", "Fourth sampled frame")), "frame_4"),
    ("gate.first_arrival", "gate_order", "choice",
     "Which object reached the center guide first?",
     (("amber_first", "The amber circle"), ("violet_first", "The violet square"),
      ("same_time", "They reached it at the same sampled time")), "amber_first"),
    ("gate.amber_before_violet", "gate_order", "noul",
     "Did the amber circle reach the center guide before the violet square?", _YES_NO, "true"),
    ("gate.arrival_gap", "gate_order", "score",
     "How many sampled-frame steps separated the two center-guide arrivals?",
     (("0", "Same sampled frame"), ("1", "One frame step"),
      ("2", "Two frame steps"), ("3", "Three frame steps")), "1"),
    ("gate.second_frame", "gate_order", "choice",
     "At the second sampled frame, which object was exactly on the center guide?",
     (("amber_only", "Only the amber circle"), ("violet_only", "Only the violet square"),
      ("both", "Both objects"), ("neither", "Neither object")), "amber_only"),
    ("collision.after_first_contact", "collision", "choice",
     "What happened after the circles first touched or overlapped?",
     (("separated", "They later separated."),
      ("remained_in_contact", "They remained touching or overlapping through the end."),
      ("no_contact", "They never touched or overlapped.")), "separated"),
    ("collision.before_final", "collision", "noul",
     "Did the two circles touch or overlap before the final sampled frame?", _YES_NO, "true"),
    ("collision.contact_count", "collision", "score",
     "In how many sampled frames were the two circles touching or overlapping?",
     (("0", "Zero frames"), ("1", "One frame"), ("2", "Two frames"), ("3", "Three frames")), "1"),
    ("collision.early_distance_trend", "collision", "choice",
     "Across the first two intervals, how did the distance between the circle centers change?",
     (("closer_then_closer", "Closer, then closer again"), ("closer_then_farther", "Closer, then farther"),
      ("farther_then_closer", "Farther, then closer"), ("farther_then_farther", "Farther, then farther again"),
      ("includes_no_change", "At least one interval had no distance change")), "closer_then_farther"),
    ("zigzag.guide_order", "zigzag_guides", "choice",
     "Which colored guide did the triangle intersect first?",
     (("cyan_first", "The cyan vertical guide"), ("magenta_first", "The magenta horizontal guide"),
      ("same_time", "Both guides at the same sampled time")), "cyan_first"),
    ("zigzag.vertical_reversal", "zigzag_guides", "noul",
     "Did the triangle reverse its vertical direction during the clip?", _YES_NO, "true"),
    ("zigzag.vertical_reversal_count", "zigzag_guides", "score",
     "How many times did the triangle reverse vertical direction?",
     (("0", "No reversals"), ("1", "One reversal"), ("2", "Two reversals")), "1"),
    ("zigzag.endpoint_vertical_motion", "zigzag_guides", "choice",
     "Which vertical directions describe the first interval and final interval, in that order?",
     (("down_down", "Down, then down"), ("down_up", "Down, then up"),
      ("up_down", "Up, then down"), ("up_up", "Up, then up")), "down_up"),
)


def decision(case):
    family, scene, kind, instructions, candidates, _ = case
    audit = {"scene_family": scene}
    if kind == "choice":
        candidates = tuple(reversed(candidates))
        audit["original_candidate_keys"] = [key for key, _ in candidates]
        criteria = {f"option_{i}": text for i, (_, text) in enumerate(candidates)}
    else:
        criteria = dict(candidates)
    return {
        "state": _STATE,
        "split": "test_ood" if scene == "zigzag_guides" else "train",
        "question": {"type": kind, "instructions": instructions, "criteria": criteria},
        "provenance": {"family": family, "ood_axis": "task_family" if scene == "zigzag_guides" else "none"},
        "audit": audit,
    }


class VideoSemanticsTests(unittest.TestCase):
    def test_all_controlled_tasks_answer_literal_observed_facts(self):
        for case in _CASES:
            with self.subTest(family=case[0]):
                self.assertEqual(video_answer(decision(case), _FACTS[case[1]]), case[-1])

    def test_reversed_observed_time_is_interpreted_in_its_supplied_order(self):
        for index, expected in ((3, "right_left"), (4, "blue"), (8, "violet_first"),
                                (9, "false"), (16, "magenta_first"), (19, "down_up")):
            case = _CASES[index]
            with self.subTest(family=case[0]):
                self.assertEqual(video_answer(decision(case), list(reversed(_FACTS[case[1]]))), expected)
        self.assertEqual(video_answer(decision(_CASES[3]), _FACTS["bounce"]), "right_left")
        self.assertEqual(video_answer(decision(_CASES[9]), _FACTS["gate_order"]), "true")

    def test_reversed_or_negated_temporal_questions_are_rejected(self):
        changes = (
            (3, "Which directions describe the final interval and the first interval, in that order?"),
            (4, "Which color was visible immediately after the first sampled color?"),
            (5, "After changing away from a color, did the circle never return to that earlier color?"),
            (8, "Which object reached the center guide last?"),
            (9, "Did the violet square reach the center guide before the amber circle?"),
            (13, "Did the two circles touch or overlap only in the final sampled frame?"),
            (15, "Across the final two intervals, how did the distance between the circle centers change?"),
            (16, "Which colored guide did the triangle intersect last?"),
            (19, "Which vertical directions describe the final interval and first interval, in that order?"),
        )
        for index, instructions in changes:
            case = _CASES[index]
            with self.subTest(family=case[0]):
                row = decision(case)
                self.assertEqual(video_answer(row, _FACTS[case[1]]), case[-1])
                row["question"]["instructions"] = instructions
                with self.assertRaisesRegex(ValueError, "question meaning"):
                    video_answer(row, _FACTS[case[1]])

    def test_swapped_choice_meanings_cannot_preserve_verification(self):
        for case in _CASES:
            if case[2] != "choice":
                continue
            with self.subTest(family=case[0]):
                row = decision(case)
                self.assertEqual(video_answer(row, _FACTS[case[1]]), case[-1])
                criteria = row["question"]["criteria"]
                criteria["option_0"], criteria["option_1"] = criteria["option_1"], criteria["option_0"]
                with self.assertRaisesRegex(ValueError, "candidate meanings"):
                    video_answer(row, _FACTS[case[1]])

    def test_ordered_noul_and_score_meanings_are_bound(self):
        for case in _CASES:
            if case[2] == "choice":
                continue
            with self.subTest(family=case[0]):
                row = decision(case)
                self.assertEqual(video_answer(row, _FACTS[case[1]]), case[-1])
                original = list(row["question"]["criteria"].items())
                first, second = original[:2]
                row["question"]["criteria"][first[0]] = second[1]
                row["question"]["criteria"][second[0]] = first[1]
                with self.assertRaisesRegex(ValueError, "ordered candidate meanings"):
                    video_answer(row, _FACTS[case[1]])
                row["question"]["criteria"] = dict(reversed(original))
                with self.assertRaisesRegex(ValueError, "ordered candidate meanings"):
                    video_answer(row, _FACTS[case[1]])

    def test_missing_or_extra_candidates_and_response_key_swaps_fail(self):
        for mutation in ("missing", "extra", "key_swap", "missing_keys", "duplicate_keys"):
            with self.subTest(mutation=mutation):
                row = decision(_CASES[8])
                self.assertEqual(video_answer(row, _FACTS["gate_order"]), "amber_first")
                if mutation == "missing":
                    row["question"]["criteria"].pop("option_0")
                elif mutation == "extra":
                    row["question"]["criteria"]["option_3"] = "Neither object arrived"
                elif mutation == "missing_keys":
                    row["audit"].pop("original_candidate_keys")
                elif mutation == "duplicate_keys":
                    row["audit"]["original_candidate_keys"][0] = row["audit"]["original_candidate_keys"][1]
                else:
                    keys = row["audit"]["original_candidate_keys"]
                    keys[0], keys[1] = keys[1], keys[0]
                with self.assertRaisesRegex(ValueError, "candidate meanings"):
                    video_answer(row, _FACTS["gate_order"])

    def test_state_and_question_evidence_cannot_be_missing_or_contradictory(self):
        for state in (None, "", "Observe the four-frame clip in reverse time order.",
                      _STATE + " Answer from the final frame alone."):
            row = decision(_CASES[0])
            self.assertEqual(video_answer(row, _FACTS["bounce"]), "right_then_left")
            row["state"] = state
            with self.assertRaisesRegex(ValueError, "state instructions"):
                video_answer(row, _FACTS["bounce"])
        for field in ("instructions", "type", "criteria"):
            row = decision(_CASES[0])
            row["question"].pop(field)
            with self.assertRaises(ValueError):
                video_answer(row, _FACTS["bounce"])

    def test_contradictory_or_unsupported_task_metadata_fails(self):
        for family in ("color.return", "bounce.vertical_reversal", "unknown.any_reversal"):
            row = decision(_CASES[1])
            self.assertEqual(video_answer(row, _FACTS["bounce"]), "true")
            row["provenance"]["family"] = family
            with self.assertRaises(ValueError):
                video_answer(row, _FACTS["bounce"])
        row = decision(_CASES[1])
        row["provenance"]["family"] = "bounce.reversal_count"
        with self.assertRaisesRegex(ValueError, "question meaning"):
            video_answer(row, _FACTS["bounce"])
        for scene in (None, "color_sequence", "gate_order", "new_scene"):
            row = decision(_CASES[1])
            row["audit"]["scene_family"] = scene
            with self.assertRaisesRegex(ValueError, "scene/task"):
                video_answer(row, _FACTS["bounce"])

    def test_only_intended_id_and_held_out_task_memberships_are_admitted(self):
        for case in _CASES:
            for split in ("train", "dev", "calibration", "policy", "test_id"):
                with self.subTest(family=case[0], split=split):
                    row = decision(case)
                    if case[1] == "zigzag_guides":
                        self.assertEqual(video_answer(row, _FACTS[case[1]]), case[-1])
                        row["split"] = split
                        row["provenance"]["ood_axis"] = "none"
                        with self.assertRaisesRegex(ValueError, "ID/OOD"):
                            video_answer(row, _FACTS[case[1]])
                    else:
                        row["split"] = split
                        self.assertEqual(video_answer(row, _FACTS[case[1]]), case[-1])
                        row["split"] = "test_ood"
                        row["provenance"]["ood_axis"] = "task_family"
                        with self.assertRaisesRegex(ValueError, "ID/OOD"):
                            video_answer(row, _FACTS[case[1]])
        for index, axis in ((1, "task_family"), (17, "none"), (17, "source"), (1, "layout")):
            case = _CASES[index]
            row = decision(case)
            row["provenance"]["ood_axis"] = axis
            with self.assertRaisesRegex(ValueError, "ID/OOD"):
                video_answer(row, _FACTS[case[1]])
        row = decision(_CASES[1])
        row["split"] = "unknown"
        with self.assertRaisesRegex(ValueError, "ID/OOD"):
            video_answer(row, _FACTS["bounce"])

    def test_missing_observed_frame_evidence_cannot_be_replaced_by_audit_answers(self):
        for case in _CASES:
            with self.subTest(family=case[0]):
                row = decision(case)
                facts = _FACTS[case[1]]
                self.assertEqual(video_answer(row, facts), case[-1])
                row["audit"]["frame_facts"] = facts
                row["audit"]["decision"] = {"answer": case[-1]}
                for incomplete in ([], facts[:3], facts + facts[:1], [{}, *facts[1:]]):
                    with self.assertRaises(ValueError):
                        video_answer(row, incomplete)
        gate = copy.deepcopy(_FACTS["gate_order"])
        gate[2]["violet_square_center"] = (129, 126)
        with self.assertRaisesRegex(ValueError, "center-guide arrival"):
            video_answer(decision(_CASES[8]), gate)
        zigzag = copy.deepcopy(_FACTS["zigzag_guides"])
        zigzag[1]["triangle_center"] = (97, 55)
        with self.assertRaisesRegex(ValueError, "colored-guide intersection"):
            video_answer(decision(_CASES[16]), zigzag)

    def test_invalid_observed_values_fail_closed(self):
        for value in (None, "30,96", (30,), (True, 96), (float("nan"), 96), (float("inf"), 96)):
            facts = copy.deepcopy(_FACTS["bounce"])
            facts[0]["focus_center"] = value
            with self.assertRaisesRegex(ValueError, "observed temporal geometry"):
                video_answer(decision(_CASES[1]), facts)
        for color in (None, "orange", ["red"]):
            facts = copy.deepcopy(_FACTS["color_sequence"])
            facts[0]["focus_color"] = color
            with self.assertRaisesRegex(ValueError, "observed color"):
                video_answer(decision(_CASES[4]), facts)
        facts = [{"amber_center": (50, 88), "violet_center": (70, 88)} for _ in range(4)]
        with self.assertRaisesRegex(ValueError, "outside the admitted candidates"):
            video_answer(decision(_CASES[14]), facts)

    def test_temporal_answers_distinguish_literal_boundary_cases(self):
        bounce = [
            {"focus_center": (30, 96)}, {"focus_center": (90, 96)},
            {"focus_center": (50, 96)}, {"focus_center": (120, 96)},
        ]
        self.assertEqual(video_answer(decision(_CASES[0]), bounce), "alternating_right_first")
        self.assertEqual(video_answer(decision(_CASES[2]), bounce), "2")
        self.assertEqual(video_answer(decision(_CASES[3]), bounce), "right_right")
        colors = [
            {"focus_color": "red"}, {"focus_color": "red"},
            {"focus_color": "blue"}, {"focus_color": "blue"},
        ]
        self.assertEqual(video_answer(decision(_CASES[5]), colors), "false")
        self.assertEqual(video_answer(decision(_CASES[6]), colors), "1")
        self.assertEqual(video_answer(decision(_CASES[7]), colors), "frame_3")
        gate = [
            {"amber_circle_center": (30, 70), "violet_square_center": (40, 126)},
            {"amber_circle_center": (128, 70), "violet_square_center": (128, 126)},
            {"amber_circle_center": (180, 70), "violet_square_center": (170, 126)},
            {"amber_circle_center": (210, 70), "violet_square_center": (190, 126)},
        ]
        self.assertEqual(video_answer(decision(_CASES[8]), gate), "same_time")
        self.assertEqual(video_answer(decision(_CASES[9]), gate), "false")
        self.assertEqual(video_answer(decision(_CASES[10]), gate), "0")
        self.assertEqual(video_answer(decision(_CASES[11]), gate), "both")
        final_contact = [
            {"amber_center": (40, 88), "violet_center": (140, 88)},
            {"amber_center": (50, 88), "violet_center": (130, 88)},
            {"amber_center": (60, 88), "violet_center": (120, 88)},
            {"amber_center": (70, 88), "violet_center": (102, 88)},
        ]
        self.assertEqual(video_answer(decision(_CASES[12]), final_contact), "remained_in_contact")
        self.assertEqual(video_answer(decision(_CASES[13]), final_contact), "false")
        self.assertEqual(video_answer(decision(_CASES[14]), final_contact), "1")
        self.assertEqual(video_answer(decision(_CASES[15]), final_contact), "closer_then_closer")
        final_contact[-1]["violet_center"] = (103, 88)
        self.assertEqual(video_answer(decision(_CASES[12]), final_contact), "no_contact")
        self.assertEqual(video_answer(decision(_CASES[14]), final_contact), "0")
        zigzag = [
            {"triangle_center": (30, 35)}, {"triangle_center": (96, 96)},
            {"triangle_center": (150, 55)}, {"triangle_center": (210, 145)},
        ]
        self.assertEqual(video_answer(decision(_CASES[16]), zigzag), "same_time")
        self.assertEqual(video_answer(decision(_CASES[18]), zigzag), "2")
        self.assertEqual(video_answer(decision(_CASES[19]), zigzag), "down_down")

    def test_answers_depend_on_observed_facts_not_declared_labels(self):
        row = decision(_CASES[9])
        row["target"] = {"label": "false"}
        row["audit"]["original_target_key"] = "false"
        row["audit"]["decision"] = {"answer": "false"}
        self.assertEqual(video_answer(row, _FACTS["gate_order"]), "true")
        self.assertEqual(video_answer(row, list(reversed(_FACTS["gate_order"]))), "false")
        self.assertEqual(temporal_answer("gate.amber_before_violet", _FACTS["gate_order"]), "true")
        self.assertEqual(temporal_answer("gate.amber_before_violet", list(reversed(_FACTS["gate_order"]))), "false")


if __name__ == "__main__":
    unittest.main()
