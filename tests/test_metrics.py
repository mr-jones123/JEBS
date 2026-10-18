"""Behavioral regression tests for the pure scoring and calibration interface."""

import copy
import math
import unittest

import numpy as np

from mdtrain.metrics import fit_temperature, probabilities, score_predictions


def prediction(identifier="q1", **changes):
    row = {
        "id": identifier,
        "group_id": "g1",
        "probabilities": {"a": 0.75, "b": 0.25},
        "target": "a",
        "source": "banking77",
        "origin": "real_human_label",
        "modality": "text",
        "primitive": "choice",
        "family": "intent",
        "ood_axis": "none",
    }
    row.update(changes)
    return row


class ScorePredictionsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            prediction("q1", probabilities={"a": 1.0, "b": 0.0}, position_count=16),
            prediction("q2", probabilities={"a": 0.95, "b": 0.05}, target="b",
                       position_count=16, ood_axis="layout"),
            prediction("q3", group_id="g2", probabilities={"a": 0.8, "b": 0.2},
                       source="rules", origin="generated_rule", primitive="noul", family="rule"),
            prediction("q4", group_id="g2", probabilities={"a": 0.8, "b": 0.1, "c": 0.1},
                       target="b", source="rules", origin="generated_rule", modality="mixed",
                       primitive="score", family="rule", ood_axis="task_family", position_count=32),
        ]

    def test_question_means_sum_brier_across_classes_before_averaging(self):
        result = score_predictions(self.rows)
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["group_count"], 2)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["nll"], 1.3803652294655616, places=12)
        self.assertAlmostEqual(result["brier"], 0.83625, places=12)
        self.assertAlmostEqual(result["ece"], 0.3875, places=12)

    def test_perfect_confidence_and_exact_point95_enter_last_ece_bin(self):
        result = score_predictions(self.rows)
        self.assertEqual(len(result["ece_bins"]), 15)
        last = result["ece_bins"][14]
        self.assertAlmostEqual(last["lower"], 0.9333333333333333)
        self.assertEqual(last["upper"], 1.0)
        self.assertEqual(last["count"], 2)
        self.assertEqual(last["accuracy"], 0.5)
        self.assertAlmostEqual(last["mean_confidence"], 0.975)
        self.assertAlmostEqual(last["gap"], 0.475)
        self.assertEqual(result["ece_bins"][0], {
            "lower": 0.0, "upper": 0.06666666666666667,
            "count": 0, "accuracy": None, "mean_confidence": None, "gap": None,
        })
        self.assertEqual(result["ece_bins"][12]["count"], 2)
        self.assertEqual(result["high_confidence"], {
            "threshold": 0.95, "count": 2, "errors": 1, "error_rate": 0.5,
        })

    def test_risk_coverage_never_splits_confidence_ties(self):
        result = score_predictions(self.rows)
        self.assertEqual(result["risk_coverage"], [
            {"threshold": 1.0, "count": 1, "coverage": 0.25, "errors": 0, "risk": 0.0},
            {"threshold": 0.95, "count": 2, "coverage": 0.5, "errors": 1, "risk": 0.5},
            {"threshold": 0.8, "count": 4, "coverage": 1.0, "errors": 2, "risk": 0.5},
        ])

    def test_slices_preserve_source_origin_input_primitive_and_holdout(self):
        slices = score_predictions(self.rows)["slices"]
        self.assertEqual(set(slices), {
            "source", "origin", "modality", "primitive", "family", "ood_axis",
            "candidate_count", "position_count",
        })
        self.assertEqual(slices["source"]["banking77"]["count"], 2)
        self.assertEqual(slices["source"]["banking77"]["accuracy"], 0.5)
        self.assertAlmostEqual(slices["source"]["banking77"]["nll"], 1.4978661367769954)
        self.assertAlmostEqual(slices["source"]["banking77"]["brier"], 0.9025)
        self.assertEqual(slices["origin"]["generated_rule"]["count"], 2)
        self.assertEqual(slices["modality"]["mixed"]["accuracy"], 0.0)
        self.assertAlmostEqual(slices["modality"]["mixed"]["brier"], 1.46)
        self.assertEqual(slices["primitive"]["noul"]["accuracy"], 1.0)
        self.assertAlmostEqual(slices["family"]["rule"]["nll"], 1.2628643221541276)
        self.assertEqual(slices["ood_axis"]["none"]["accuracy"], 1.0)
        self.assertEqual(slices["ood_axis"]["layout"]["accuracy"], 0.0)
        self.assertEqual(slices["ood_axis"]["task_family"]["count"], 1)
        self.assertEqual(slices["candidate_count"]["3"]["count"], 1)
        self.assertEqual(slices["position_count"]["missing"]["count"], 1)
        self.assertEqual(slices["position_count"]["16"]["count"], 2)

    def test_group_intervals_preserve_whole_reused_groups(self):
        uncertainty = score_predictions(self.rows)["uncertainty"]
        intervals = uncertainty["intervals"]
        self.assertEqual(intervals["accuracy"], {"lower": 0.5, "upper": 0.5})
        self.assertAlmostEqual(intervals["nll"]["lower"], 1.2628643221541276)
        self.assertAlmostEqual(intervals["nll"]["upper"], 1.4978661367769954)
        self.assertAlmostEqual(intervals["brier"]["lower"], 0.77)
        self.assertAlmostEqual(intervals["brier"]["upper"], 0.9025)
        reversed_intervals = score_predictions(reversed(self.rows))["uncertainty"]["intervals"]
        self.assertEqual(intervals, reversed_intervals)

    def test_question_mean_is_not_an_unweighted_mean_of_groups(self):
        rows = [prediction("a", group_id="small", probabilities={"a": 1.0})]
        rows.extend(prediction(identifier, group_id="large", probabilities={"a": 0.5, "b": 0.5},
                               target="b") for identifier in ("b", "c", "d"))
        result = score_predictions(rows)
        self.assertEqual(result["accuracy"], 0.25)
        self.assertAlmostEqual(result["nll"], 0.5198603854199589)
        self.assertEqual(result["brier"], 0.375)
        self.assertEqual(result["uncertainty"]["intervals"]["accuracy"], {"lower": 0.0, "upper": 1.0})

    def test_candidate_order_breaks_argmax_ties(self):
        first = score_predictions([prediction(probabilities={"a": 0.5, "b": 0.5}, target="b")])
        second = score_predictions([prediction(probabilities={"b": 0.5, "a": 0.5}, target="b")])
        self.assertEqual(first["accuracy"], 0.0)
        self.assertEqual(second["accuracy"], 1.0)
        self.assertAlmostEqual(first["nll"], 0.6931471805599453)
        self.assertEqual(first["brier"], 0.5)
        self.assertEqual(first["high_confidence"], {
            "threshold": 0.95, "count": 0, "errors": 0, "error_rate": None,
        })

    def test_zero_target_probability_has_infinite_nll_without_clipping(self):
        result = score_predictions([prediction(probabilities={"a": 1.0, "b": 0.0}, target="b")])
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["nll"], math.inf)
        self.assertEqual(result["brier"], 2.0)
        self.assertEqual(result["ece"], 1.0)
        self.assertEqual(result["uncertainty"]["intervals"]["nll"], {
            "lower": math.inf, "upper": math.inf,
        })

    def test_singleton_and_single_group_intervals_are_actual_point_values(self):
        result = score_predictions([prediction(probabilities={"only": 1.0}, target="only")])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["nll"], 0.0)
        self.assertEqual(result["brier"], 0.0)
        self.assertEqual(result["ece"], 0.0)
        self.assertEqual(result["uncertainty"]["intervals"], {
            "accuracy": {"lower": 1.0, "upper": 1.0},
            "nll": {"lower": 0.0, "upper": 0.0},
            "brier": {"lower": 0.0, "upper": 0.0},
        })

    def test_roundoff_tolerance_does_not_silently_renormalize(self):
        result = score_predictions([prediction(probabilities={"a": 0.75, "b": 0.2500005})])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["nll"], 0.2876820724517809, places=12)
        self.assertAlmostEqual(result["brier"], 0.12500025000025, places=13)
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            score_predictions([prediction(probabilities={"a": 0.75, "b": 0.250002})])

    def test_rejects_missing_duplicate_or_empty_ids_and_empty_rows(self):
        missing = prediction()
        del missing["id"]
        for rows in ([], [missing], [prediction(id="")], [prediction(id=" ")],
                     [prediction(), prediction()]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                score_predictions(rows)

    def test_rejects_invalid_targets_and_candidate_coverage(self):
        for changes in ({"target": "missing"}, {"target": 0}, {"probabilities": {}},
                        {"probabilities": [0.5, 0.5]}, {"probabilities": {"": 1.0}},
                        {"probabilities": {1: 1.0}}, {"probabilities": {"b": 1.0}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                score_predictions([prediction(**changes)])

    def test_rejects_invalid_probability_mass_and_nonfinite_values(self):
        distributions = [
            {"a": 0.3, "b": 0.3}, {"a": -0.1, "b": 1.1},
            {"a": math.nan, "b": 0.5}, {"a": math.inf, "b": 0.0},
            {"a": -math.inf, "b": 0.0}, {"a": "0.75", "b": 0.25},
            {"a": True, "b": False},
        ]
        for distribution in distributions:
            with self.subTest(distribution=distribution), self.assertRaises(ValueError):
                score_predictions([prediction(probabilities=distribution)])

    def test_requires_groups_and_slice_metadata_and_valid_position_count(self):
        for field in ("group_id", "source", "origin", "modality", "primitive", "family", "ood_axis"):
            missing = prediction()
            del missing[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                score_predictions([missing])
        for value in (None, 0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "position_count"):
                score_predictions([prediction(position_count=value)])

    def test_does_not_mutate_rows_and_accepts_iterators(self):
        before = copy.deepcopy(self.rows)
        result = score_predictions(iter(self.rows))
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(self.rows, before)


class ProbabilitiesTests(unittest.TestCase):
    def test_variable_cardinality_temperature_and_singleton(self):
        result = probabilities([[math.log(4.0), 0.0], [0.0, 0.0, 0.0], [12345.0]], temperature=2.0)
        self.assertAlmostEqual(result[0][0], 0.6666666666666666)
        self.assertAlmostEqual(result[0][1], 0.3333333333333333)
        self.assertEqual(result[1], [0.3333333333333333, 0.3333333333333333, 0.3333333333333333])
        self.assertEqual(result[2], [1.0])

    def test_softmax_is_stable_for_large_offsets_and_extreme_finite_values(self):
        result = probabilities([[10000.0, 10000.0], [1e308, -1e308], [-10000.0, -10001.0]])
        self.assertEqual(result[0], [0.5, 0.5])
        self.assertEqual(result[1], [1.0, 0.0])
        self.assertAlmostEqual(result[2][0], 0.7310585786300049)
        self.assertAlmostEqual(result[2][1], 0.2689414213699951)
        self.assertEqual(probabilities([[1e308, 1e308]], temperature=1e-300), [[0.5, 0.5]])

    def test_numpy_inputs_are_supported_without_accepting_padded_infinities(self):
        self.assertEqual(probabilities(np.array([[0.0, 0.0]])), [[0.5, 0.5]])
        with self.assertRaisesRegex(ValueError, "finite"):
            probabilities([[0.0, -math.inf]])

    def test_rejects_empty_candidates_nonfinite_and_nonnumeric_logits(self):
        for logits in ([[]], [[math.nan, 0.0]], [[math.inf]], [["0"]], [[True]], [0.0], "abc"):
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                probabilities(logits)
        self.assertEqual(probabilities([]), [])
        self.assertEqual(probabilities([[0.0]]), [[1.0]])

    def test_temperature_can_rescale_opposite_finite_extremes_without_losing_mass(self):
        result = probabilities([[1e308, -1e308]], temperature=1e308)
        self.assertAlmostEqual(result[0][0], 0.8807970779778823, places=14)
        self.assertAlmostEqual(result[0][1], 0.11920292202211755, places=14)

    def test_rejects_nonpositive_nonfinite_and_nonnumeric_temperature(self):
        for temperature in (0, -1, math.inf, math.nan, "1", True):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                probabilities([[0.0, 0.0]], temperature)


class FitTemperatureTests(unittest.TestCase):
    def test_fits_known_scalar_optimum_and_improves_consumer_nll(self):
        logits = [[math.log(4.0), 0.0]] * 3
        targets = [0, 0, 1]
        temperature = fit_temperature(logits, targets)
        self.assertAlmostEqual(temperature, 2.0, places=5)
        fitted_probabilities = probabilities(logits, temperature)
        self.assertAlmostEqual(fitted_probabilities[0][0], 0.6666666666666666, places=7)
        raw_rows, fitted_rows = [], []
        for index, (raw, fitted, target) in enumerate(zip(probabilities(logits), fitted_probabilities, targets)):
            raw_rows.append(prediction(str(index), probabilities=dict(zip(("a", "b"), raw)),
                                       target=("a", "b")[target]))
            fitted_rows.append(prediction(str(index), probabilities=dict(zip(("a", "b"), fitted)),
                                          target=("a", "b")[target]))
        raw_score = score_predictions(raw_rows)
        fitted_score = score_predictions(fitted_rows)
        self.assertAlmostEqual(raw_score["nll"], 0.6852416716875066, places=12)
        self.assertAlmostEqual(fitted_score["nll"], 0.6365141682948128, places=12)
        self.assertLess(fitted_score["nll"], raw_score["nll"])
        self.assertAlmostEqual(fitted_score["brier"], 0.4444444444444444, places=12)

    def test_variable_cardinality_calibration_keeps_singletons_and_uniform_rows(self):
        logits = [[math.log(4.0), 0.0], [math.log(4.0), 0.0],
                  [math.log(4.0), 0.0], [0.0, 0.0, 0.0], [8000.0]]
        temperature = fit_temperature(logits, [0, 0, 1, 2, 0])
        self.assertAlmostEqual(temperature, 2.0, places=5)
        result = probabilities(logits, temperature)
        self.assertAlmostEqual(result[0][0], 0.6666666666666666, places=7)
        self.assertEqual(result[3], [0.3333333333333333, 0.3333333333333333, 0.3333333333333333])
        self.assertEqual(result[4], [1.0])

    def test_nonidentifiable_calibration_preserves_identity_and_observable_probabilities(self):
        temperature = fit_temperature([[10000.0, 10000.0], [0.0]], [1, 0])
        self.assertEqual(probabilities([[10000.0, 10000.0], [0.0]], temperature), [[0.5, 0.5], [1.0]])
        self.assertEqual(temperature, 1.0)

    def test_calibration_handles_large_common_offsets_without_overflow(self):
        temperature = fit_temperature([[10000.0 + math.log(4.0), 10000.0]] * 3, [0, 0, 1])
        self.assertAlmostEqual(temperature, 2.0, places=5)
        self.assertAlmostEqual(probabilities([[10000.0 + math.log(4.0), 10000.0]], temperature)[0][0],
                               0.6666666666666666, places=7)

    def test_invalid_calibration_coverage_and_targets_are_rejected(self):
        cases = [([], []), ([[0.0]], []), ([[0.0]], [0, 0]),
                 ([[0.0, 1.0]], [2]), ([[0.0]], [-1]),
                 ([[0.0]], [True]), ([[0.0]], [0.0]), ([[math.nan]], [0])]
        for logits, targets in cases:
            with self.subTest(logits=logits, targets=targets), self.assertRaises(ValueError):
                fit_temperature(logits, targets)


if __name__ == "__main__":
    unittest.main()
