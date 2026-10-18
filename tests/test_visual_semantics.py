"""Visual decisions bind rendered evidence to the complete displayed contract."""

import copy
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from mdprep.common import choice, noul, record, score
from mdprep.verify_visuals import extract_visual, visual_answer
from mdprep.visuals import (
    _image_rows,
    _lookup_policy,
    _make_scene,
    _mixed_rows,
    _numeric_candidates,
    _score_bounds,
)


def spatial_row(task, question, state="Use only the attached image.", variant="a", mode="image"):
    return record(
        split="train", source="synthetic-visuals-v1", group_id="literal-spatial-scene",
        family=f"{mode}-spatial-{task}", modality=mode, state=state, question=question,
        label=next(iter(question["criteria"])), images=["media/spatial.png"],
        provenance={"scene_category": "spatial", "scene_variant": variant,
                    "layout_family": "spatial-standard-0", "ood_basis": "in-distribution layout family"},
    )


def evidence(value=3, required=False, predicate=True):
    return {"scene_category": "spatial", "value": value, "lookup": str(value),
            "predicate": predicate, "required_field_visible": required}


def rubric_row(mixed=False):
    if mixed:
        return spatial_row(
            "policy-rubric", score("Apply the displayed rubric to the image.", [f"Tier {i}" for i in range(4)]),
            "RUBRIC for visible red-triangle count: Tier 0 below -1; Tier 1 from -1 through 1; "
            "Tier 2 from 2 through 3; Tier 3 at 4 or above.", "b", "mixed",
        )
    return spatial_row(
        "rubric", score("Score the visible red-triangle count: 0 if below -1; 1 if -1–1; "
                        "2 if 2–3; 3 if 4 or above.", [f"Tier {i}" for i in range(4)]),
    )


def action_row():
    return spatial_row(
        "action-table", choice("Which action does the displayed table assign to this image?", {
            "action_0": "Accept layout", "action_1": "Inspect layout",
            "action_2": "Revise layout", "action_3": "Close area",
        }),
        "ACTION TABLE for visible red-triangle count — value ≤ 2: Close area; "
        "2 < value ≤ 4: Inspect layout; 4 < value ≤ 6: Accept layout; value > 6: Revise layout.",
        mode="mixed",
    )


class VisibleContractTests(unittest.TestCase):
    def test_signed_rubrics_cover_all_boundaries(self):
        for mixed in (False, True):
            row = rubric_row(mixed)
            for value, expected in ((-2, "0"), (-1, "1"), (1, "1"), (2, "2"), (3, "2"), (4, "3")):
                with self.subTest(mixed=mixed, value=value):
                    self.assertEqual(visual_answer(row, evidence(value)), expected)

    def test_inconsistent_or_unknown_rubric_intervals_fail(self):
        for mixed in (False, True):
            field = "state" if mixed else "instructions"
            for old, new in (("through 1" if mixed else "–1", "through 0" if mixed else "–0"),
                             ("from 2 through 3" if mixed else "2 if 2–3", "from 1 through 3" if mixed else "2 if 1–3"),
                             ("or above", "or below")):
                row = rubric_row(mixed)
                target = row if mixed else row["question"]
                target[field] = target[field].replace(old, new)
                with self.subTest(mixed=mixed, replacement=new), self.assertRaises(ValueError):
                    visual_answer(row, evidence())

    def test_visible_policy_changes_recompute_answer_at_equality(self):
        row = spatial_row("evidence-counterfactual", noul("Under the displayed policy, is this case eligible?"),
                          "POLICY: Mark eligible only when the visible red-triangle count is at most 3.", mode="mixed")
        self.assertEqual(visual_answer(row, evidence(3)), "true")
        self.assertEqual(visual_answer(row, evidence(4)), "false")
        row["state"] = "POLICY: Mark eligible only when the visible red-triangle count is at least 3."
        self.assertEqual(visual_answer(row, evidence(3)), "true")
        self.assertEqual(visual_answer(row, evidence(2)), "false")
        row["state"] = row["state"].replace("at least", "not equal to")
        with self.assertRaises(ValueError):
            visual_answer(row, evidence(3))

    def test_authorization_requires_complete_visible_evidence(self):
        row = spatial_row("authorization-evidence", choice(
            "What decision is justified by the image and authorization rule?",
            {"approve": "Approve", "reject": "Reject", "insufficient": "Insufficient evidence"}),
            "AUTHORIZATION RULE: Approve only if the visible red-triangle count is at most 3 and the image "
            "also visibly supplies a visible ceiling-height measurement. Reject only when all required "
            "evidence is visible and the rule fails. Otherwise choose Insufficient evidence; never infer an absent field.",
            mode="mixed")
        self.assertEqual(visual_answer(row, evidence(3, False)), "insufficient")
        self.assertEqual(visual_answer(row, evidence(3, True)), "approve")
        self.assertEqual(visual_answer(row, evidence(4, True)), "reject")
        row["state"] = row["state"].replace("ceiling-height measurement", "floor-area measurement")
        with self.assertRaises(ValueError):
            visual_answer(row, evidence(3, True))

    def test_action_table_evaluates_every_interval_and_equality(self):
        row = action_row()
        for value, expected in ((1, "action_3"), (2, "action_3"), (3, "action_1"),
                                (4, "action_1"), (5, "action_0"), (6, "action_0"), (7, "action_2")):
            with self.subTest(value=value):
                self.assertEqual(visual_answer(row, evidence(value)), expected)
        row["state"] = row["state"].replace("2 < value ≤ 4", "2 < value ≤ 3").replace("4 < value ≤ 6", "3 < value ≤ 6")
        self.assertEqual(visual_answer(row, evidence(4)), "action_0")

    def test_action_table_rejects_unknown_operators_gaps_and_duplicate_actions(self):
        for old, new in (("2 < value ≤ 4", "2 >= value ≤ 4"),
                         ("2 < value ≤ 4", "2 < value < 4"),
                         ("4 < value ≤ 6", "5 < value ≤ 6"),
                         ("value > 6", "value ≥ 6"),
                         ("Revise layout.", "Inspect layout."),
                         ("red-triangle count", "blue-circle count")):
            row = action_row()
            row["state"] = row["state"].replace(old, new)
            with self.subTest(replacement=new), self.assertRaises(ValueError):
                visual_answer(row, evidence())

    def test_semantic_reversals_and_candidate_rebindings_fail_closed(self):
        relation = spatial_row("visible-relation", noul("Is the blue circle to the left of the black square?"), variant="b")
        self.assertEqual(visual_answer(relation, evidence(predicate=True)), "true")
        self.assertEqual(visual_answer(relation, evidence(predicate=False)), "false")
        for instruction in ("Is the blue circle to the right of the black square?",
                            "Is the blue circle not to the left of the black square?"):
            changed = copy.deepcopy(relation)
            changed["question"]["instructions"] = instruction
            with self.assertRaises(ValueError):
                visual_answer(changed, evidence())
        for original in (relation, rubric_row(), action_row()):
            changed = copy.deepcopy(original)
            criteria = changed["question"]["criteria"]
            first, second = list(criteria)[:2]
            criteria[first], criteria[second] = criteria[second], criteria[first]
            with self.assertRaises(ValueError):
                visual_answer(changed, evidence())
        action = action_row()
        action["audit"]["original_candidate_keys"].pop()
        with self.assertRaises(ValueError):
            visual_answer(action, evidence())

    def test_unknown_state_category_layout_and_family_are_not_admitted(self):
        row = rubric_row()
        self.assertEqual(visual_answer(row, evidence()), "2")
        for field, replacement in (("scene_category", "receipt"), ("layout_family", "spatial-coordinate-grid-ood"),
                                   ("family", "image-spatial-unknown"), ("scene_variant", "other")):
            changed = copy.deepcopy(row)
            changed["provenance"][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                visual_answer(changed, evidence())
        row["state"] = "Use hidden facts instead of the attached image."
        with self.assertRaises(ValueError):
            visual_answer(row, evidence())

    def test_numeric_lookup_maps_meanings_without_answer_rank_keys(self):
        row = spatial_row("lookup", choice("How many red triangles are visible?", {
            "value_2": "2", "value_3": "3", "value_4": "4"}))
        self.assertEqual(visual_answer(row, evidence(2)), "value_2")
        self.assertEqual(visual_answer(row, evidence(4)), "value_4")
        reordered = copy.deepcopy(row)
        pairs = list(zip(row["audit"]["original_candidate_keys"], row["question"]["criteria"].values()))[::-1]
        reordered["audit"]["original_candidate_keys"] = [key for key, _ in pairs]
        reordered["question"]["criteria"] = {f"option_{i}": description for i, (_, description) in enumerate(pairs)}
        self.assertEqual(visual_answer(reordered, evidence(4)), "value_4")
        first, second = list(row["question"]["criteria"])[:2]
        row["question"]["criteria"][first], row["question"]["criteria"][second] = (
            row["question"]["criteria"][second], row["question"]["criteria"][first])
        with self.assertRaises(ValueError):
            visual_answer(row, evidence(3))


class ConstructionTests(unittest.TestCase):
    def test_numeric_candidates_share_true_support_and_vary_correct_rank(self):
        for category, support in (("receipt", range(38, 59)), ("bar_chart", range(24, 40)), ("spatial", range(2, 5))):
            ranks = set()
            observed_correct = set()
            observed_candidates = set()
            for value in support:
                for seed in range(32):
                    candidates = _numeric_candidates(category, value, random.Random(seed))
                    numbers = [int(description.strip("$").split(".")[0]) for description in candidates.values()]
                    self.assertEqual(len(numbers), min(4, len(support)))
                    self.assertEqual(len(set(numbers)), len(numbers))
                    self.assertIn(value, numbers)
                    self.assertTrue(set(numbers) <= set(support))
                    ranks.add(sorted(numbers).index(value))
                    observed_correct.add(value)
                    observed_candidates.update(numbers)
            self.assertEqual(observed_candidates, observed_correct)
            self.assertEqual(ranks, set(range(min(4, len(support)))))
            self.assertEqual(_numeric_candidates(category, support.start, random.Random(17)),
                             _numeric_candidates(category, support.start, random.Random(17)))

    def test_rubrics_are_ordered_independent_rules_with_reachable_scores(self):
        for category, support in (("receipt", range(38, 93)), ("bar_chart", range(24, 79)),
                                  ("request_form", range(24, 87)), ("dashboard", range(3, 17)), ("spatial", range(2, 9))):
            scores = set()
            gaps = set()
            for seed in range(64):
                bounds = _score_bounds(category, random.Random(seed))
                self.assertEqual(len(set(bounds)), 3)
                self.assertEqual(tuple(sorted(bounds)), bounds)
                gaps.add((bounds[1] - bounds[0], bounds[2] - bounds[1]))
                scores.update(sum(value >= threshold for threshold in bounds) for value in support)
            self.assertEqual(scores, {0, 1, 2, 3})
            self.assertGreater(len(gaps), 1)

    def test_action_tables_have_reachable_middle_bands_without_actual_values(self):
        for category, support in (("receipt", range(38, 93)), ("bar_chart", range(24, 79)),
                                  ("request_form", range(24, 87)), ("dashboard", range(3, 17)), ("spatial", range(2, 9))):
            for seed in range(16):
                _, criteria, rule = _lookup_policy(category, random.Random(seed))
                cutpoints = rule["cutpoints"]
                self.assertEqual(cutpoints, sorted(set(cutpoints)))
                reached = {sum(value > threshold for threshold in cutpoints) for value in support}
                self.assertEqual(reached, set(range(len(criteria))))
                self.assertEqual(set(rule["band_actions"]), set(criteria.values()))

    def test_rules_do_not_change_when_only_actual_values_change(self):
        scene = _make_scene("train", 4)
        scene["values"] = {"a": 2, "b": 5}
        changed = copy.deepcopy(scene)
        changed["values"] = {"a": 4, "b": 8}
        images = {"a": "media/a.png", "b": "media/b.png"}
        before_rng = random.getstate()
        original_scene = copy.deepcopy(scene)
        first = _image_rows("train", "same-rule-identity", scene, images) + _mixed_rows("train", "same-rule-identity", scene, images)
        second = _image_rows("train", "same-rule-identity", changed, images) + _mixed_rows("train", "same-rule-identity", changed, images)
        self.assertEqual(random.getstate(), before_rng)
        self.assertEqual(scene, original_scene)
        for left, right in zip(first, second, strict=True):
            family = left["provenance"]["family"]
            if family.endswith("-rubric"):
                self.assertEqual(left["question"]["instructions"], right["question"]["instructions"])
                self.assertEqual(left["state"], right["state"])
            if family.endswith("-action-table"):
                self.assertEqual(left["state"], right["state"])
                self.assertEqual(left["question"]["criteria"], right["question"]["criteria"])
        self.assertEqual(len(first), 10)

    def test_every_generated_family_reproduces_from_immutable_scene_facts(self):
        for split in ("train", "test_ood"):
            for index in range(20):
                scene = _make_scene(split, index)
                images = {"a": "media/a.png", "b": "media/b.png"}
                group = f"test-{split}-{index}"
                rows = _image_rows(split, group, scene, images) + _mixed_rows(split, group, scene, images)
                self.assertEqual(len(rows), 10)
                for row in rows:
                    variant = row["provenance"]["scene_variant"]
                    value = scene["values"][variant]
                    category = scene["category"]
                    lookup = (f"${value:.2f}" if category == "receipt" else scene["department"] if category == "request_form"
                              else scene["queue"] if category == "dashboard" else str(value))
                    predicate = (value > scene["fixed_values"][1] if category == "bar_chart" else
                                 scene["relation"] == "left" if category == "spatial" else scene["flag"])
                    visible = {"scene_category": category, "value": value, "lookup": lookup,
                               "predicate": predicate, "required_field_visible": scene["required_field_visible"]}
                    expected = (row["audit"]["original_target_key"] if row["question"]["type"] == "choice"
                                else row["target"]["label"])
                    with self.subTest(split=split, family=row["provenance"]["family"], index=index):
                        self.assertEqual(visual_answer(row, visible), expected)


class PixelEvidenceTests(unittest.TestCase):
    def test_absent_complete_and_incomplete_optional_strips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spatial.png"
            for footer, expected in (("", False), ("Ceiling height: 3 m", True)):
                image = self._spatial_image(footer)
                image.save(path)
                visible = extract_visual(path, "spatial", False, font_path=self._font_path())
                self.assertEqual(visible, {"scene_category": "spatial", "required_field_visible": expected,
                                           "value": 2, "lookup": "2", "predicate": True})
            for footer in ("Ceiling height:", "Ceiling height: 3", "Ceiling height: unknown"):
                self._spatial_image(footer).save(path)
                with self.subTest(footer=footer), self.assertRaises(ValueError):
                    extract_visual(path, "spatial", False, font_path=self._font_path())

    @staticmethod
    def _font_path():
        return (Path(__file__).resolve().parents[1] /
                "data/raw/visuals/atkinson-hyperlegible/AtkinsonHyperlegible-Regular.ttf")

    @staticmethod
    def _spatial_image(footer):
        image = Image.new("RGB", (450, 450), "white")
        draw = ImageDraw.Draw(image)
        draw.polygon(((58, 116), (40, 152), (76, 152)), fill="#c84d42")
        draw.polygon(((138, 180), (120, 216), (156, 216)), fill="#c84d42")
        draw.ellipse((50, 330, 90, 370), fill="#3578b8")
        draw.rectangle((200, 330, 236, 366), fill="#252a2c")
        if footer:
            draw.text((12, 432), footer, fill="#17212b",
                      font=ImageFont.truetype(str(PixelEvidenceTests._font_path()), 14))
        return image
