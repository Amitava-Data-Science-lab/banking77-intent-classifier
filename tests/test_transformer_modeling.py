import numpy as np

from banking77_intent_classifier.transformer_modeling import (
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

    rows = evaluate_oos_threshold_candidates(
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
