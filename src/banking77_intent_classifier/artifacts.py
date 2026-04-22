"""Persistence helpers for model artifacts and reports."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline

from banking77_intent_classifier.config import ExperimentConfig
from banking77_intent_classifier.evaluation import EvaluationArtifacts
from banking77_intent_classifier.modeling import WeightExport


def ensure_output_directories(config: ExperimentConfig) -> None:
    """Create configured output directories."""

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)


def save_model_artifacts(
    pipeline: Pipeline,
    config: ExperimentConfig,
    label_names: list[str],
    weight_export: WeightExport,
    label_text_mapping: dict[int, str] | None = None,
) -> None:
    """Persist model pipeline and reusable metadata."""

    joblib.dump(pipeline, config.artifacts_dir / "model.joblib")
    _write_json(config.artifacts_dir / "label_mapping.json", {str(i): label for i, label in enumerate(label_names)})
    _write_json(
        config.artifacts_dir / "model_metadata.json",
        {
            "model_family": config.model_family,
            "dataset_type": config.dataset_type,
            "dataset_name": config.dataset_name,
            "dataset_task": config.dataset_task,
            "dataset_source": str(config.dataset_source) if config.dataset_source is not None else None,
            "train_split": config.train_split,
            "validation_split": config.validation_split,
            "test_split": config.test_split,
            "include_oos": config.include_oos,
            "oos_confidence_threshold": config.oos_confidence_threshold,
            "oos_margin_threshold": config.oos_margin_threshold,
            "text_column": config.text_column,
            "label_column": config.label_column,
            "random_seed": config.random_seed,
            "encoder_model_name": config.encoder.model_name if config.model_family.startswith("sentence_transformer_") else None,
            "reranker_model_name": config.reranker.model_name if config.reranker.enabled else None,
            "reranker_top_k": config.reranker.top_k if config.reranker.enabled else None,
            "probability_calibration_method": config.classifier.probability_calibration_method,
            "probability_calibration_cv": config.classifier.probability_calibration_cv,
            "num_labels": len(label_names),
            "num_features": int(
                weight_export.coefficients.shape[1]
                if weight_export.coefficients is not None
                else len(weight_export.feature_names)
            ),
        },
    )
    feature_mapping = {feature: index for index, feature in enumerate(weight_export.feature_names.tolist())}
    mapping_filename = (
        "feature_mapping.json"
        if config.model_family.startswith("sentence_transformer_")
        else "vectorizer_vocabulary.json"
    )
    _write_json(config.artifacts_dir / mapping_filename, feature_mapping)
    if label_text_mapping is not None:
        _write_json(
            config.artifacts_dir / "label_text_mapping.json",
            {str(label_id): label_text for label_id, label_text in label_text_mapping.items()},
        )
    if weight_export.coefficients is not None:
        sparse.save_npz(config.artifacts_dir / "weights.npz", weight_export.coefficients)


def save_evaluation_reports(
    evaluation: EvaluationArtifacts,
    config: ExperimentConfig,
    run_summary: dict | None = None,
) -> None:
    """Persist evaluation outputs for offline inspection."""

    _write_json(config.reports_dir / "classification_report.json", evaluation.classification_report)
    evaluation.confusion_matrix_df.to_csv(config.reports_dir / "confusion_matrix.csv", index=True)
    evaluation.normalized_confusion_matrix_df.to_csv(
        config.reports_dir / "confusion_matrix_normalized.csv",
        index=True,
    )
    evaluation.top_confusions_df.to_csv(config.reports_dir / "top_confusions.csv", index=False)
    evaluation.top_features_by_class_df.to_csv(
        config.reports_dir / "top_features_by_class.csv",
        index=False,
    )
    if run_summary is not None:
        _write_json(config.reports_dir / "run_summary.json", run_summary)
    _write_json(
        config.reports_dir / "metrics_summary.json",
        {
            "accuracy": evaluation.accuracy,
            "macro_f1": evaluation.macro_f1,
            "top_5_accuracy": evaluation.top_5_accuracy,
        },
    )
    _write_json(config.reports_dir / "oos_metrics.json", evaluation.oos_metrics)


def save_tuning_reports(
    config: ExperimentConfig,
    best_params: dict,
    best_score: float,
    cv_results_df: pd.DataFrame,
    search_settings: dict,
) -> None:
    """Persist hyperparameter search outputs for offline analysis."""

    _write_json(
        config.reports_dir / "best_params.json",
        {
            "best_params": best_params,
            "best_cv_score": best_score,
        },
    )
    _write_json(config.reports_dir / "search_settings.json", search_settings)
    cv_results_df.to_csv(config.reports_dir / "cv_results.csv", index=False)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")


def load_top_confusions(path: str | Path) -> pd.DataFrame:
    """Helper for consumers who want to load the persisted confusion summary."""

    return pd.read_csv(path)
