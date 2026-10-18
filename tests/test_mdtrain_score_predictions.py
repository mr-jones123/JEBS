"""Offline scoring must distinguish rule reading from original-label repetition."""

import pytest

from tools.score_predictions import (
    _synthetic_key,
    _synthetic_predictions,
    score_rows,
    self_test,
)


def test_scorer_self_test_covers_all_correct_and_always_original():
    result = self_test()
    assert result["status"] == "SELF_TEST_PASSED"
    correct = result["models"]["all_correct"]
    ignored = result["models"]["always_original_label"]
    assert correct["accuracy"] == 1.0
    assert correct["paired_edit_changes_label"] == {
        "questions": 2,
        "rule_ignoring_rate_original_label": 0.0,
        "new_correct_label_rate": 1.0,
    }
    assert ignored["accuracy"] == pytest.approx(2 / 3)
    assert ignored["paired_edit_changes_label"] == {
        "questions": 2,
        "rule_ignoring_rate_original_label": 1.0,
        "new_correct_label_rate": 0.0,
    }


def test_scorer_rejects_candidate_order_mismatch():
    keys = _synthetic_key()
    predictions = _synthetic_predictions(keys, always_original=False)
    predictions[0]["probabilities"] = dict(reversed(predictions[0]["probabilities"].items()))
    with pytest.raises(ValueError, match="candidate alignment"):
        score_rows(predictions, keys)
