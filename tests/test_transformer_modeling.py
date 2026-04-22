import numpy as np

from banking77_intent_classifier.transformer_modeling import (
    _select_oos_aware_threshold_candidates,
    build_energy_threshold_candidates,
    build_distance_threshold_candidates,
    build_known_intent_centroids,
    apply_probability_threshold,
    compute_energy_scores,
    compute_nearest_known_intent_distances,
    evaluate_energy_threshold_candidates,
    evaluate_oos_threshold_candidates,
    evaluate_distance_threshold_candidates,
    fit_temperature,
    negative_log_likelihood,
    _softmax,
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


def test_build_known_intent_centroids_excludes_oos_and_normalizes() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    labels = [0, 0, 1, 2]
    centroids, label_ids = build_known_intent_centroids(
        embeddings=embeddings,
        labels=labels,
        label_names=["intent_a", "intent_b", "oos"],
        distance_metric="cosine",
    )

    assert label_ids == [0, 1]
    assert centroids.shape == (2, 2)
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0)


def test_build_distance_threshold_candidates_uses_validation_distances_deterministically() -> None:
    distances = np.array([0.4, 0.2, 0.2, 0.9, 0.6], dtype=np.float32)
    candidates = build_distance_threshold_candidates(distances=distances, max_candidates=10)

    assert candidates == [0.20000000298023224, 0.4000000059604645, 0.6000000238418579, 0.8999999761581421]


def test_apply_probability_threshold_supports_distance_or_probability_rule() -> None:
    probabilities = np.array(
        [
            [0.95, 0.03, 0.02],
            [0.70, 0.20, 0.10],
            [0.85, 0.10, 0.05],
        ]
    )
    distances = np.array([0.1, 0.8, 0.2], dtype=np.float32)
    predictions = apply_probability_threshold(
        probabilities=probabilities,
        label_names=["intent_a", "intent_b", "oos"],
        oos_confidence_threshold=0.2,
        nearest_known_intent_distances=distances,
        oos_distance_threshold=0.5,
    )

    assert predictions.tolist() == [0, 2, 0]


def test_apply_probability_threshold_supports_energy_or_probability_rule() -> None:
    probabilities = np.array(
        [
            [0.95, 0.03, 0.02],
            [0.70, 0.20, 0.10],
            [0.85, 0.10, 0.05],
        ]
    )
    energies = np.array([-5.0, -1.0, -4.0], dtype=np.float32)
    predictions = apply_probability_threshold(
        probabilities=probabilities,
        label_names=["intent_a", "intent_b", "oos"],
        oos_confidence_threshold=0.2,
        energy_scores=energies,
        oos_energy_threshold=-2.0,
    )

    assert predictions.tolist() == [0, 2, 0]


def test_evaluate_distance_threshold_candidates_uses_joint_rule() -> None:
    probabilities = np.array(
        [
            [0.95, 0.03, 0.02],
            [0.60, 0.30, 0.10],
            [0.75, 0.10, 0.15],
            [0.40, 0.20, 0.40],
        ]
    )
    distances = np.array([0.1, 0.7, 0.2, 0.9], dtype=np.float32)
    y_true = [0, 2, 1, 2]

    rows, metadata = evaluate_distance_threshold_candidates(
        probabilities=probabilities,
        nearest_known_intent_distances=distances,
        y_true=y_true,
        label_names=["intent_a", "intent_b", "oos"],
        fixed_oos_confidence_threshold=0.05,
        distance_threshold_candidates=[0.3, 0.8],
        analysis_top_k_confusions=10,
        analysis_top_k_features_per_class=5,
        selection_metric="macro_f1",
        selection_strategy="oos_aware_constrained",
        max_in_scope_false_oos_rate=0.5,
        macro_f1_tolerance_ladder=[0.5],
        fallback_strategy="best_macro_f1",
    )

    assert metadata["selected_distance_threshold"] == rows[0]["distance_threshold"]
    assert metadata["fixed_oos_confidence_threshold"] == 0.05
    assert len(rows) == 2


def test_compute_energy_scores_matches_expected_ordering() -> None:
    logits = np.array(
        [
            [4.0, 1.0, -1.0],
            [0.1, 0.0, -0.1],
        ],
        dtype=np.float32,
    )

    energies = compute_energy_scores(logits=logits, temperature=1.0)

    assert energies.shape == (2,)
    assert energies[0] < energies[1]


def test_temperature_scaling_softens_probabilities() -> None:
    logits = np.array([[4.0, 0.0]], dtype=np.float32)

    sharp = _softmax(logits, temperature=1.0)
    soft = _softmax(logits, temperature=2.0)

    assert sharp[0, 0] > soft[0, 0]


def test_negative_log_likelihood_and_fit_temperature_select_best_grid_value() -> None:
    logits = np.array(
        [
            [4.0, 0.0],
            [3.0, 0.0],
            [0.1, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1], dtype=np.int64)

    grid = [0.5, 1.0, 2.0, 5.0]
    temperature, best_nll = fit_temperature(logits=logits, labels=labels, temperature_grid=grid)

    nlls = {candidate: negative_log_likelihood(_softmax(logits, candidate), labels) for candidate in grid}
    assert temperature == min(nlls, key=nlls.get)
    assert best_nll == nlls[temperature]


def test_build_energy_threshold_candidates_uses_validation_energies_deterministically() -> None:
    energies = np.array([-4.0, -2.0, -2.0, -1.0], dtype=np.float32)
    candidates = build_energy_threshold_candidates(energies=energies, max_candidates=10)

    assert candidates == [-4.0, -2.0, -1.0]


def test_evaluate_energy_threshold_candidates_uses_joint_rule() -> None:
    probabilities = np.array(
        [
            [0.95, 0.03, 0.02],
            [0.60, 0.30, 0.10],
            [0.75, 0.10, 0.15],
            [0.40, 0.20, 0.40],
        ]
    )
    energies = np.array([-5.0, -1.0, -4.0, -0.5], dtype=np.float32)
    y_true = [0, 2, 1, 2]

    rows, metadata = evaluate_energy_threshold_candidates(
        probabilities=probabilities,
        energy_scores=energies,
        y_true=y_true,
        label_names=["intent_a", "intent_b", "oos"],
        fixed_oos_confidence_threshold=0.05,
        energy_threshold_candidates=[-2.0, -0.75],
        analysis_top_k_confusions=10,
        analysis_top_k_features_per_class=5,
        selection_metric="macro_f1",
        selection_strategy="oos_aware_constrained",
        max_in_scope_false_oos_rate=0.5,
        macro_f1_tolerance_ladder=[0.5],
        fallback_strategy="best_macro_f1",
    )

    assert metadata["selected_energy_threshold"] == rows[0]["energy_threshold"]
    assert metadata["fixed_oos_confidence_threshold"] == 0.05
    assert len(rows) == 2
