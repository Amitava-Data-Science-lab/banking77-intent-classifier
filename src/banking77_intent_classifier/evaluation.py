"""Evaluation utilities for metrics and model diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass(slots=True)
class EvaluationArtifacts:
    """Structured evaluation outputs for export and downstream analysis."""

    accuracy: float
    macro_f1: float
    top_5_accuracy: float
    oos_metrics: dict
    classification_report: dict
    confusion_matrix_df: pd.DataFrame
    normalized_confusion_matrix_df: pd.DataFrame
    top_confusions_df: pd.DataFrame
    top_features_by_class_df: pd.DataFrame


def evaluate_predictions(
    y_true: list[int],
    y_pred: list[int],
    top_k_predicted_labels: np.ndarray,
    label_names: list[str],
    coefficients: np.ndarray | None,
    feature_names: np.ndarray,
    top_k_confusions: int,
    top_k_features_per_class: int,
) -> EvaluationArtifacts:
    """Generate metrics, confusion analysis, and feature importance tables."""

    labels = list(range(len(label_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = float(report["macro avg"]["f1-score"])
    top_5_accuracy = _top_k_accuracy(
        y_true=y_true,
        top_k_predicted_labels=top_k_predicted_labels,
    )
    oos_metrics = _build_oos_metrics(
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
        report=report,
    )
    report["summary_metrics"] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "top_5_accuracy": top_5_accuracy,
    }

    confusion_df = pd.DataFrame(matrix, index=label_names, columns=label_names)
    row_sums = confusion_df.sum(axis=1).replace(0, 1)
    normalized_df = confusion_df.div(row_sums, axis=0)

    top_confusions_df = _top_confusions_from_matrix(
        matrix=matrix,
        normalized_matrix=normalized_df.to_numpy(),
        label_names=label_names,
        top_k=top_k_confusions,
    )
    top_features_df = _top_features_by_class(
        coefficients=coefficients,
        feature_names=feature_names,
        label_names=label_names,
        top_k=top_k_features_per_class,
    )

    return EvaluationArtifacts(
        accuracy=accuracy,
        macro_f1=macro_f1,
        top_5_accuracy=top_5_accuracy,
        oos_metrics=oos_metrics,
        classification_report=report,
        confusion_matrix_df=confusion_df,
        normalized_confusion_matrix_df=normalized_df,
        top_confusions_df=top_confusions_df,
        top_features_by_class_df=top_features_df,
    )


def save_confusion_matrix_figure(
    confusion_matrix_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Banking77 Confusion Matrix",
) -> None:
    """Persist a confusion matrix heatmap figure."""

    output_path = Path(output_path)
    fig, axis = plt.subplots(figsize=(24, 20))
    image = axis.imshow(confusion_matrix_df.to_numpy(), interpolation="nearest", cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(np.arange(len(confusion_matrix_df.columns)))
    axis.set_yticks(np.arange(len(confusion_matrix_df.index)))
    axis.set_xticklabels(confusion_matrix_df.columns, rotation=90, fontsize=7)
    axis.set_yticklabels(confusion_matrix_df.index, fontsize=7)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _top_k_accuracy(y_true: list[int], top_k_predicted_labels: np.ndarray) -> float:
    matches = [
        int(true_label) in predicted_labels.tolist()
        for true_label, predicted_labels in zip(y_true, top_k_predicted_labels, strict=True)
    ]
    return float(np.mean(matches))


def _top_confusions_from_matrix(
    matrix: np.ndarray,
    normalized_matrix: np.ndarray,
    label_names: list[str],
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for true_index, true_label in enumerate(label_names):
        for predicted_index, predicted_label in enumerate(label_names):
            if true_index == predicted_index:
                continue

            error_count = int(matrix[true_index, predicted_index])
            if error_count == 0:
                continue

            rows.append(
                {
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": error_count,
                    "row_normalized_rate": float(normalized_matrix[true_index, predicted_index]),
                }
            )

    top_rows = sorted(
        rows,
        key=lambda item: (item["count"], item["row_normalized_rate"]),
        reverse=True,
    )[:top_k]
    return pd.DataFrame(top_rows)


def _top_features_by_class(
    coefficients: np.ndarray | None,
    feature_names: np.ndarray,
    label_names: list[str],
    top_k: int,
) -> pd.DataFrame:
    if coefficients is None:
        return pd.DataFrame(columns=["label", "rank", "feature", "weight"])

    rows: list[dict] = []
    for class_index, label_name in enumerate(label_names):
        class_weights = coefficients[class_index]
        top_indices = np.argsort(class_weights)[-top_k:][::-1]
        for rank, feature_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "label": label_name,
                    "rank": rank,
                    "feature": str(feature_names[feature_index]),
                    "weight": float(class_weights[feature_index]),
                }
            )

    return pd.DataFrame(rows)


def _build_oos_metrics(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
    report: dict,
) -> dict:
    if "oos" not in label_names:
        return {
            "available": False,
            "label": None,
        }

    oos_label_id = label_names.index("oos")
    true_oos_predicted_oos = sum(
        true_label == oos_label_id and predicted_label == oos_label_id
        for true_label, predicted_label in zip(y_true, y_pred, strict=True)
    )
    true_oos_predicted_in_scope = sum(
        true_label == oos_label_id and predicted_label != oos_label_id
        for true_label, predicted_label in zip(y_true, y_pred, strict=True)
    )
    true_in_scope_predicted_oos = sum(
        true_label != oos_label_id and predicted_label == oos_label_id
        for true_label, predicted_label in zip(y_true, y_pred, strict=True)
    )
    true_in_scope_predicted_in_scope = sum(
        true_label != oos_label_id and predicted_label != oos_label_id
        for true_label, predicted_label in zip(y_true, y_pred, strict=True)
    )

    oos_support = int(report["oos"]["support"])
    in_scope_support = len(y_true) - oos_support

    return {
        "available": True,
        "label": "oos",
        "precision": float(report["oos"]["precision"]),
        "recall": float(report["oos"]["recall"]),
        "f1": float(report["oos"]["f1-score"]),
        "support": oos_support,
        "miss_rate": float(1.0 - report["oos"]["recall"]),
        "in_scope_false_oos_rate": (
            float(true_in_scope_predicted_oos / in_scope_support) if in_scope_support else 0.0
        ),
        "true_oos_predicted_oos": true_oos_predicted_oos,
        "true_oos_predicted_in_scope": true_oos_predicted_in_scope,
        "true_in_scope_predicted_oos": true_in_scope_predicted_oos,
        "true_in_scope_predicted_in_scope": true_in_scope_predicted_in_scope,
    }
