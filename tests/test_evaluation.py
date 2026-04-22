import numpy as np

from banking77_intent_classifier.evaluation import evaluate_predictions
from banking77_intent_classifier.reranking import build_label_text_mapping


def test_evaluation_returns_confusions_and_top_features() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]
    top_k_predicted_labels = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1],
        ]
    )
    label_names = ["balance", "card_arrival"]
    coefficients = np.array(
        [
            [0.9, 0.2, -0.1],
            [-0.2, 0.4, 0.8],
        ]
    )
    feature_names = np.array(["balance", "account", "card"])

    result = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        top_k_predicted_labels=top_k_predicted_labels,
        label_names=label_names,
        coefficients=coefficients,
        feature_names=feature_names,
        top_k_confusions=10,
        top_k_features_per_class=2,
    )

    assert result.accuracy == 0.5
    assert result.macro_f1 == 0.5
    assert result.top_5_accuracy == 1.0
    assert result.classification_report["summary_metrics"]["top_5_accuracy"] == 1.0
    assert result.oos_metrics["available"] is False
    assert not result.top_confusions_df.empty
    assert set(result.top_features_by_class_df["label"]) == set(label_names)


def test_evaluation_builds_oos_metrics_when_oos_label_exists() -> None:
    y_true = [0, 1, 2, 2]
    y_pred = [0, 2, 2, 1]
    top_k_predicted_labels = np.array(
        [
            [0, 2],
            [2, 1],
            [2, 0],
            [1, 2],
        ]
    )
    label_names = ["balance", "card_arrival", "oos"]
    coefficients = np.array(
        [
            [0.9, 0.2, -0.1],
            [-0.2, 0.4, 0.8],
            [0.1, -0.3, 0.6],
        ]
    )
    feature_names = np.array(["balance", "account", "card"])

    result = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        top_k_predicted_labels=top_k_predicted_labels,
        label_names=label_names,
        coefficients=coefficients,
        feature_names=feature_names,
        top_k_confusions=10,
        top_k_features_per_class=2,
    )

    assert result.oos_metrics["available"] is True
    assert result.oos_metrics["label"] == "oos"
    assert result.oos_metrics["support"] == 2
    assert result.oos_metrics["true_oos_predicted_oos"] == 1
    assert result.oos_metrics["true_oos_predicted_in_scope"] == 1
    assert result.oos_metrics["true_in_scope_predicted_oos"] == 1
    assert result.oos_metrics["true_in_scope_predicted_in_scope"] == 1
    assert result.oos_metrics["recall"] == 0.5
    assert result.oos_metrics["miss_rate"] == 0.5
    assert result.oos_metrics["in_scope_false_oos_rate"] == 0.5


def test_build_label_text_mapping_humanizes_labels() -> None:
    mapping = build_label_text_mapping(["verify_my_identity", "unable_to_verify_identity"])

    assert mapping[0] == "verify my identity"
    assert mapping[1] == "unable to verify identity"
