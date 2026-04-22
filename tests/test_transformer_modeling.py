import numpy as np

from banking77_intent_classifier.transformer_modeling import (
    _select_oos_aware_threshold_candidates,
    apply_probability_threshold,
    evaluate_oos_threshold_candidates,
    top_k_from_probabilities,
)


def test_apply_probability_threshold_forces_low_confidence_to_oos() -> None:
    probabilities = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.2, 0.5, 0.3],
            [0.2, 0.25, 0.55],
        ]
    )
    predictions = apply_probability_threshold(
        probabilities=probabilities,
        label_names=["intent_a", "intent_b", "oos"],
        oos_confidence_threshold=0.6,
    )

    assert predictions.tolist() == [1, 2, 2]


def test_top_k_from_probabilities_returns_ranked_labels() -> None:
    probabilities = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
        ]
    )

    ranked = top_k_from_probabilities(probabilities=probabilities, k=2)

    assert ranked.tolist() == [[1, 2], [2, 1]]


def test_evaluate_oos_threshold_candidates_ranks_by_macro_f1() -> None:
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.45, 0.4, 0.15],
            [0.34, 0.33, 0.33],
        ]
    )
    y_true = [0, 1, 2, 2]

    rows, metadata = evaluate_oos_threshold_candidates(
        probabilities=probabilities,
        y_true=y_true,
        label_names=["intent_a", "intent_b", "oos"],
        threshold_candidates=[0.3, 0.5],
        analysis_top_k_confusions=10,
        analysis_top_k_features_per_class=5,
        selection_metric="macro_f1",
    )

    assert len(rows) == 2
    assert rows[0]["threshold"] == 0.5
    assert rows[0]["macro_f1"] >= rows[1]["macro_f1"]
    assert metadata["selected_threshold"] == 0.5
    assert metadata["strategy"] == "metric"


def test_evaluate_oos_threshold_candidates_selects_constrained_oos_threshold() -> None:
    rows, metadata = _select_oos_aware_threshold_candidates(
        rows=[
            {"threshold": 0.02, "macro_f1": 0.94, "oos_f1": 0.41, "in_scope_false_oos_rate": 0.001, "oos_metrics": {"f1": 0.41}, "accuracy": 0.93, "top_5_accuracy": 0.96},
            {"threshold": 0.05, "macro_f1": 0.93, "oos_f1": 0.44, "in_scope_false_oos_rate": 0.02, "oos_metrics": {"f1": 0.44}, "accuracy": 0.91, "top_5_accuracy": 0.96},
            {"threshold": 0.08, "macro_f1": 0.90, "oos_f1": 0.50, "in_scope_false_oos_rate": 0.05, "oos_metrics": {"f1": 0.50}, "accuracy": 0.89, "top_5_accuracy": 0.96},
        ],
        selection_metric="macro_f1",
        max_in_scope_false_oos_rate=0.03,
        macro_f1_tolerance_ladder=[0.01],
        fallback_strategy="best_macro_f1",
    )

    assert metadata["selected_threshold"] == 0.05
    assert metadata["successful_tolerance"] == 0.01
    assert metadata["fallback_used"] is False
    assert rows[0]["threshold"] == 0.05
    assert rows[0]["selection_eligible"] is True
    assert rows[0]["eligibility_reason"] == "selected_by_oos_aware_constraints"


def test_evaluate_oos_threshold_candidates_falls_back_to_best_macro_f1() -> None:
    rows, metadata = _select_oos_aware_threshold_candidates(
        rows=[
            {"threshold": 0.02, "macro_f1": 0.94, "oos_f1": 0.20, "in_scope_false_oos_rate": 0.04, "oos_metrics": {"f1": 0.20}, "accuracy": 0.93, "top_5_accuracy": 0.96},
            {"threshold": 0.05, "macro_f1": 0.90, "oos_f1": 0.45, "in_scope_false_oos_rate": 0.06, "oos_metrics": {"f1": 0.45}, "accuracy": 0.89, "top_5_accuracy": 0.96},
        ],
        selection_metric="macro_f1",
        max_in_scope_false_oos_rate=0.03,
        macro_f1_tolerance_ladder=[0.01, 0.02],
        fallback_strategy="best_macro_f1",
    )

    assert metadata["fallback_used"] is True
    assert metadata["selected_threshold"] == 0.02
    assert rows[0]["threshold"] == 0.02
    assert rows[0]["eligibility_reason"] == "fallback_best_macro_f1"
