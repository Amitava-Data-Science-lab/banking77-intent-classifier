"""Transformer fine-tuning workflow for intent classification."""

from __future__ import annotations

import inspect
import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report

from banking77_intent_classifier.config import ExperimentConfig, TransformerConfig
from banking77_intent_classifier.data import DatasetBundle
from banking77_intent_classifier.evaluation import EvaluationArtifacts, evaluate_predictions


@dataclass(slots=True)
class TransformerTrainingArtifacts:
    """Trained transformer components plus evaluation-time metadata."""

    model: Any
    tokenizer: Any
    trainer: Any
    selected_oos_threshold: float | None
    selected_distance_threshold: float | None
    selected_energy_threshold: float | None
    temperature: float
    validation_metrics_by_threshold: list[dict[str, Any]]
    threshold_selection_metadata: dict[str, Any] = field(default_factory=dict)
    validation_distance_metrics_by_threshold: list[dict[str, Any]] = field(default_factory=list)
    distance_selection_metadata: dict[str, Any] = field(default_factory=dict)
    validation_energy_metrics_by_threshold: list[dict[str, Any]] = field(default_factory=list)
    energy_selection_metadata: dict[str, Any] = field(default_factory=dict)
    known_intent_centroids: np.ndarray | None = None
    known_intent_centroid_label_ids: list[int] = field(default_factory=list)
    known_intent_centroid_label_names: list[str] = field(default_factory=list)


def train_transformer_classifier(
    dataset: DatasetBundle,
    config: ExperimentConfig,
) -> TransformerTrainingArtifacts:
    """Fine-tune a transformer classifier on the provided dataset."""

    tokenizer, model, trainer = _build_trainer(dataset=dataset, config=config)
    trainer.train()

    selected_oos_threshold = config.oos_confidence_threshold
    selected_distance_threshold: float | None = None
    selected_energy_threshold: float | None = None
    calibration_temperature = 1.0
    validation_metrics_by_threshold: list[dict[str, Any]] = []
    validation_distance_metrics_by_threshold: list[dict[str, Any]] = []
    validation_energy_metrics_by_threshold: list[dict[str, Any]] = []
    threshold_selection_metadata: dict[str, Any] = {
        "strategy": "manual_override" if selected_oos_threshold is not None else None,
        "selected_threshold": selected_oos_threshold,
    }
    distance_selection_metadata: dict[str, Any] = {
        "strategy": "disabled" if not config.transformer.oos_distance_enabled else None,
        "selected_distance_threshold": None,
    }
    energy_selection_metadata: dict[str, Any] = {
        "strategy": "disabled" if not config.transformer.oos_energy_enabled else None,
        "selected_energy_threshold": None,
    }
    known_intent_centroids: np.ndarray | None = None
    known_intent_centroid_label_ids: list[int] = []
    known_intent_centroid_label_names: list[str] = []
    validation_logits: np.ndarray | None = None

    if (
        selected_oos_threshold is None
        and (
            config.transformer.threshold_candidates
            or config.transformer.oos_energy_enabled
            or config.transformer.temperature_scaling_enabled
        )
    ):
        validation_logits = predict_logits(
            trainer=trainer,
            texts=dataset.validation_texts,
            labels=dataset.validation_labels,
            tokenizer=tokenizer,
            transformer_config=config.transformer,
        )
        if config.transformer.temperature_scaling_enabled:
            calibration_temperature, best_nll = fit_temperature(
                logits=validation_logits,
                labels=np.asarray(dataset.validation_labels, dtype=np.int64),
                temperature_grid=config.transformer.temperature_scaling_grid,
            )
            threshold_selection_metadata["temperature_scaling"] = {
                "enabled": True,
                "temperature": calibration_temperature,
                "objective": "validation_nll",
                "grid": config.transformer.temperature_scaling_grid,
                "best_nll": best_nll,
            }

    if selected_oos_threshold is None and config.transformer.threshold_candidates:
        if validation_logits is None:
            validation_logits = predict_logits(
                trainer=trainer,
                texts=dataset.validation_texts,
                labels=dataset.validation_labels,
                tokenizer=tokenizer,
                transformer_config=config.transformer,
            )
        validation_probabilities = _softmax(validation_logits, temperature=calibration_temperature)
        validation_metrics_by_threshold, threshold_selection_metadata = evaluate_oos_threshold_candidates(
            probabilities=validation_probabilities,
            y_true=dataset.validation_labels,
            label_names=dataset.label_names,
            threshold_candidates=config.transformer.threshold_candidates,
            analysis_top_k_confusions=config.analysis.top_k_confusions,
            analysis_top_k_features_per_class=config.analysis.top_k_features_per_class,
            selection_metric=config.transformer.threshold_selection_metric,
            selection_strategy=config.transformer.threshold_selection_strategy,
            max_in_scope_false_oos_rate=config.transformer.threshold_max_in_scope_false_oos_rate,
            macro_f1_tolerance_ladder=config.transformer.threshold_macro_f1_tolerance_ladder,
            fallback_strategy=config.transformer.threshold_fallback_strategy,
        )
        if config.transformer.temperature_scaling_enabled:
            threshold_selection_metadata["temperature_scaling"] = {
                "enabled": True,
                "temperature": calibration_temperature,
                "objective": "validation_nll",
                "grid": config.transformer.temperature_scaling_grid,
            }
        selected_oos_threshold = threshold_selection_metadata["selected_threshold"]

    if config.transformer.oos_distance_enabled:
        if config.transformer.oos_fixed_probability_source != "selected_validation_threshold":
            raise ValueError(
                "Unsupported oos_fixed_probability_source: "
                f"{config.transformer.oos_fixed_probability_source}"
            )
        if selected_oos_threshold is None:
            raise ValueError(
                "Distance-based OOS scoring requires a fixed probability threshold. "
                "Enable threshold selection or set oos_confidence_threshold explicitly."
            )
        if config.transformer.oos_distance_metric != "cosine":
            raise ValueError(
                f"Unsupported oos_distance_metric: {config.transformer.oos_distance_metric}"
            )
        if config.transformer.oos_distance_candidate_source != "validation_distances":
            raise ValueError(
                "Unsupported oos_distance_candidate_source: "
                f"{config.transformer.oos_distance_candidate_source}"
            )

        train_embeddings = predict_embeddings(
            model=model,
            tokenizer=tokenizer,
            texts=dataset.train_texts,
            transformer_config=config.transformer,
        )
        known_intent_centroids, known_intent_centroid_label_ids = build_known_intent_centroids(
            embeddings=train_embeddings,
            labels=dataset.train_labels,
            label_names=dataset.label_names,
            distance_metric=config.transformer.oos_distance_metric,
        )
        known_intent_centroid_label_names = [
            dataset.label_names[label_id] for label_id in known_intent_centroid_label_ids
        ]

        validation_embeddings = predict_embeddings(
            model=model,
            tokenizer=tokenizer,
            texts=dataset.validation_texts,
            transformer_config=config.transformer,
        )
        validation_nearest_distances = compute_nearest_known_intent_distances(
            embeddings=validation_embeddings,
            known_intent_centroids=known_intent_centroids,
            distance_metric=config.transformer.oos_distance_metric,
        )
        distance_threshold_candidates = build_distance_threshold_candidates(
            distances=validation_nearest_distances
        )
        validation_probabilities = predict_probabilities(
            trainer=trainer,
            texts=dataset.validation_texts,
            labels=dataset.validation_labels,
            tokenizer=tokenizer,
            transformer_config=config.transformer,
        )
        (
            validation_distance_metrics_by_threshold,
            distance_selection_metadata,
        ) = evaluate_distance_threshold_candidates(
            probabilities=validation_probabilities,
            nearest_known_intent_distances=validation_nearest_distances,
            y_true=dataset.validation_labels,
            label_names=dataset.label_names,
            fixed_oos_confidence_threshold=selected_oos_threshold,
            distance_threshold_candidates=distance_threshold_candidates,
            analysis_top_k_confusions=config.analysis.top_k_confusions,
            analysis_top_k_features_per_class=config.analysis.top_k_features_per_class,
            selection_metric=config.transformer.threshold_selection_metric,
            selection_strategy=config.transformer.oos_distance_selection_strategy,
            max_in_scope_false_oos_rate=config.transformer.threshold_max_in_scope_false_oos_rate,
            macro_f1_tolerance_ladder=config.transformer.threshold_macro_f1_tolerance_ladder,
            fallback_strategy=config.transformer.threshold_fallback_strategy,
        )
        selected_distance_threshold = distance_selection_metadata["selected_distance_threshold"]

    if config.transformer.oos_energy_enabled:
        if config.transformer.oos_energy_fixed_probability_source != "selected_validation_threshold":
            raise ValueError(
                "Unsupported oos_energy_fixed_probability_source: "
                f"{config.transformer.oos_energy_fixed_probability_source}"
            )
        if selected_oos_threshold is None:
            raise ValueError(
                "Energy-based OOS scoring requires a fixed probability threshold. "
                "Enable threshold selection or set oos_confidence_threshold explicitly."
            )
        if config.transformer.oos_energy_candidate_source != "validation_energies":
            raise ValueError(
                "Unsupported oos_energy_candidate_source: "
                f"{config.transformer.oos_energy_candidate_source}"
            )

        if validation_logits is None:
            validation_logits = predict_logits(
                trainer=trainer,
                texts=dataset.validation_texts,
                labels=dataset.validation_labels,
                tokenizer=tokenizer,
                transformer_config=config.transformer,
            )
        validation_probabilities = _softmax(validation_logits, temperature=calibration_temperature)
        validation_energies = compute_energy_scores(
            logits=validation_logits,
            temperature=config.transformer.oos_energy_temperature,
        )
        energy_threshold_candidates = build_energy_threshold_candidates(
            energies=validation_energies
        )
        (
            validation_energy_metrics_by_threshold,
            energy_selection_metadata,
        ) = evaluate_energy_threshold_candidates(
            probabilities=validation_probabilities,
            energy_scores=validation_energies,
            y_true=dataset.validation_labels,
            label_names=dataset.label_names,
            fixed_oos_confidence_threshold=selected_oos_threshold,
            energy_threshold_candidates=energy_threshold_candidates,
            analysis_top_k_confusions=config.analysis.top_k_confusions,
            analysis_top_k_features_per_class=config.analysis.top_k_features_per_class,
            selection_metric=config.transformer.threshold_selection_metric,
            selection_strategy=config.transformer.oos_energy_selection_strategy,
            max_in_scope_false_oos_rate=config.transformer.threshold_max_in_scope_false_oos_rate,
            macro_f1_tolerance_ladder=config.transformer.threshold_macro_f1_tolerance_ladder,
            fallback_strategy=config.transformer.threshold_fallback_strategy,
        )
        selected_energy_threshold = energy_selection_metadata["selected_energy_threshold"]

    return TransformerTrainingArtifacts(
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        selected_oos_threshold=selected_oos_threshold,
        selected_distance_threshold=selected_distance_threshold,
        selected_energy_threshold=selected_energy_threshold,
        temperature=calibration_temperature,
        validation_metrics_by_threshold=validation_metrics_by_threshold,
        threshold_selection_metadata=threshold_selection_metadata,
        validation_distance_metrics_by_threshold=validation_distance_metrics_by_threshold,
        distance_selection_metadata=distance_selection_metadata,
        validation_energy_metrics_by_threshold=validation_energy_metrics_by_threshold,
        energy_selection_metadata=energy_selection_metadata,
        known_intent_centroids=known_intent_centroids,
        known_intent_centroid_label_ids=known_intent_centroid_label_ids,
        known_intent_centroid_label_names=known_intent_centroid_label_names,
    )


def predict_probabilities(
    trainer,
    texts: list[str],
    labels: list[int],
    tokenizer,
    transformer_config: TransformerConfig,
    temperature: float = 1.0,
) -> np.ndarray:
    """Predict class probabilities from a fine-tuned transformer."""

    prediction_dataset = _build_hf_dataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=transformer_config.max_length,
    )
    predictions = trainer.predict(prediction_dataset)
    return _softmax(predictions.predictions, temperature=temperature)


def predict_logits(
    trainer,
    texts: list[str],
    labels: list[int],
    tokenizer,
    transformer_config: TransformerConfig,
) -> np.ndarray:
    """Predict raw logits from a fine-tuned transformer."""

    prediction_dataset = _build_hf_dataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=transformer_config.max_length,
    )
    predictions = trainer.predict(prediction_dataset)
    return predictions.predictions


def predict_embeddings(
    model,
    tokenizer,
    texts: list[str],
    transformer_config: TransformerConfig,
) -> np.ndarray:
    """Extract a shared pooled hidden-state representation for distance scoring."""

    try:
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as error:
        raise ImportError(
            "torch is required for transformer embedding extraction. "
            "Install it with `pip install torch` or `pip install -e \".[dev]\"`."
        ) from error

    device = next(model.parameters()).device
    model.eval()
    embeddings: list[np.ndarray] = []

    for start_index in range(0, len(texts), transformer_config.eval_batch_size):
        batch_texts = texts[start_index : start_index + transformer_config.eval_batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=transformer_config.max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            outputs = model.mpnet(**encoded, return_dict=True)
            pooled = outputs.last_hidden_state[:, 0, :]
            if transformer_config.oos_distance_metric == "cosine":
                pooled = torch_functional.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.detach().cpu().numpy())

    if not embeddings:
        hidden_size = getattr(model.config, "hidden_size", 0)
        return np.empty((0, hidden_size), dtype=np.float32)

    return np.vstack(embeddings)


def evaluate_transformer_predictions(
    probabilities: np.ndarray,
    y_true: list[int],
    label_names: list[str],
    oos_confidence_threshold: float | None,
    analysis_top_k_confusions: int,
    analysis_top_k_features_per_class: int,
    nearest_known_intent_distances: np.ndarray | None = None,
    oos_distance_threshold: float | None = None,
    energy_scores: np.ndarray | None = None,
    oos_energy_threshold: float | None = None,
) -> EvaluationArtifacts:
    """Evaluate transformer probabilities using the shared reporting format."""

    predictions = apply_probability_threshold(
        probabilities=probabilities,
        label_names=label_names,
        oos_confidence_threshold=oos_confidence_threshold,
        nearest_known_intent_distances=nearest_known_intent_distances,
        oos_distance_threshold=oos_distance_threshold,
        energy_scores=energy_scores,
        oos_energy_threshold=oos_energy_threshold,
    )
    top_k_predictions = top_k_from_probabilities(probabilities=probabilities, k=5)
    return evaluate_predictions(
        y_true=y_true,
        y_pred=predictions.tolist(),
        top_k_predicted_labels=top_k_predictions,
        label_names=label_names,
        coefficients=None,
        feature_names=np.array([], dtype=str),
        top_k_confusions=analysis_top_k_confusions,
        top_k_features_per_class=analysis_top_k_features_per_class,
    )


def apply_probability_threshold(
    probabilities: np.ndarray,
    label_names: list[str],
    oos_confidence_threshold: float | None,
    nearest_known_intent_distances: np.ndarray | None = None,
    oos_distance_threshold: float | None = None,
    energy_scores: np.ndarray | None = None,
    oos_energy_threshold: float | None = None,
) -> np.ndarray:
    """Apply an inference-time OOS threshold to class probabilities."""

    predicted_labels = np.argmax(probabilities, axis=1)
    if (
        oos_confidence_threshold is None
        and oos_distance_threshold is None
        and oos_energy_threshold is None
    ) or "oos" not in label_names:
        return predicted_labels

    oos_label_id = label_names.index("oos")
    max_probabilities = probabilities.max(axis=1)
    predicted_labels = predicted_labels.copy()
    oos_mask = np.zeros_like(max_probabilities, dtype=bool)
    if oos_confidence_threshold is not None:
        oos_mask |= max_probabilities < oos_confidence_threshold
    if oos_distance_threshold is not None:
        if nearest_known_intent_distances is None:
            raise ValueError(
                "nearest_known_intent_distances are required when using oos_distance_threshold."
            )
        oos_mask |= nearest_known_intent_distances > oos_distance_threshold
    if oos_energy_threshold is not None:
        if energy_scores is None:
            raise ValueError("energy_scores are required when using oos_energy_threshold.")
        oos_mask |= energy_scores > oos_energy_threshold
    predicted_labels[oos_mask] = oos_label_id
    return predicted_labels


def top_k_from_probabilities(probabilities: np.ndarray, k: int) -> np.ndarray:
    """Return top-k label ids from a probability matrix."""

    top_k = min(k, probabilities.shape[1])
    return np.argsort(probabilities, axis=1)[:, -top_k:][:, ::-1]


def build_known_intent_centroids(
    embeddings: np.ndarray,
    labels: list[int],
    label_names: list[str],
    distance_metric: str,
) -> tuple[np.ndarray, list[int]]:
    """Build one centroid per known intent, excluding the OOS label."""

    if distance_metric != "cosine":
        raise ValueError(f"Unsupported oos_distance_metric: {distance_metric}")

    oos_label_id = label_names.index("oos") if "oos" in label_names else None
    centroid_vectors: list[np.ndarray] = []
    centroid_label_ids: list[int] = []

    for label_id, label_name in enumerate(label_names):
        if label_id == oos_label_id or label_name == "oos":
            continue
        label_mask = np.asarray(labels) == label_id
        if not np.any(label_mask):
            continue
        centroid = embeddings[label_mask].mean(axis=0)
        centroid = _normalize_embeddings(np.asarray([centroid], dtype=np.float32))[0]
        centroid_vectors.append(centroid)
        centroid_label_ids.append(label_id)

    if not centroid_vectors:
        raise ValueError("No known-intent centroids could be built from the training split.")

    return np.vstack(centroid_vectors), centroid_label_ids


def compute_nearest_known_intent_distances(
    embeddings: np.ndarray,
    known_intent_centroids: np.ndarray,
    distance_metric: str,
) -> np.ndarray:
    """Compute nearest known-intent centroid distance for each embedding."""

    if distance_metric != "cosine":
        raise ValueError(f"Unsupported oos_distance_metric: {distance_metric}")

    similarities = embeddings @ known_intent_centroids.T
    nearest_similarities = np.max(similarities, axis=1)
    return 1.0 - nearest_similarities


def build_distance_threshold_candidates(
    distances: np.ndarray,
    max_candidates: int = 200,
) -> list[float]:
    """Build deterministic distance-threshold candidates from validation distances."""

    unique_distances = np.unique(np.asarray(distances, dtype=np.float64))
    if unique_distances.size <= max_candidates:
        return unique_distances.tolist()

    quantiles = np.linspace(0.0, 1.0, num=max_candidates)
    sampled = np.quantile(unique_distances, quantiles)
    return np.unique(sampled).tolist()


def compute_energy_scores(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute energy scores from model logits."""

    scaled_logits = logits / temperature
    max_logits = np.max(scaled_logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(scaled_logits - max_logits), axis=1)) + max_logits.squeeze(axis=1)
    return -temperature * logsumexp


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature_grid: list[float],
) -> tuple[float, float]:
    """Choose a temperature that minimizes validation negative log-likelihood."""

    if logits.size == 0:
        return 1.0, 0.0

    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in temperature_grid:
        probabilities = _softmax(logits, temperature=temperature)
        nll = negative_log_likelihood(probabilities=probabilities, labels=labels)
        if nll < best_nll:
            best_nll = nll
            best_temperature = temperature
    return best_temperature, best_nll


def negative_log_likelihood(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute mean negative log-likelihood for a probability matrix."""

    clipped = np.clip(probabilities[np.arange(labels.shape[0]), labels], a_min=1e-12, a_max=1.0)
    return float(-np.mean(np.log(clipped)))


def build_energy_threshold_candidates(
    energies: np.ndarray,
    max_candidates: int = 200,
) -> list[float]:
    """Build deterministic energy-threshold candidates from validation energies."""

    unique_energies = np.unique(np.asarray(energies, dtype=np.float64))
    if unique_energies.size <= max_candidates:
        return unique_energies.tolist()

    quantiles = np.linspace(0.0, 1.0, num=max_candidates)
    sampled = np.quantile(unique_energies, quantiles)
    return np.unique(sampled).tolist()


def evaluate_oos_threshold_candidates(
    probabilities: np.ndarray,
    y_true: list[int],
    label_names: list[str],
    threshold_candidates: list[float],
    analysis_top_k_confusions: int,
    analysis_top_k_features_per_class: int,
    selection_metric: str,
    selection_strategy: str = "metric",
    max_in_scope_false_oos_rate: float = 0.03,
    macro_f1_tolerance_ladder: list[float] | None = None,
    fallback_strategy: str = "best_macro_f1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate threshold candidates on a validation split and rank them."""

    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates:
        evaluation = evaluate_transformer_predictions(
            probabilities=probabilities,
            y_true=y_true,
            label_names=label_names,
            oos_confidence_threshold=threshold,
            analysis_top_k_confusions=analysis_top_k_confusions,
            analysis_top_k_features_per_class=analysis_top_k_features_per_class,
        )
        rows.append(
            {
                "threshold": threshold,
                "accuracy": evaluation.accuracy,
                "macro_f1": evaluation.macro_f1,
                "top_5_accuracy": evaluation.top_5_accuracy,
                "oos_f1": evaluation.oos_metrics.get("f1", 0.0),
                "in_scope_false_oos_rate": evaluation.oos_metrics.get("in_scope_false_oos_rate", 0.0),
                "oos_metrics": evaluation.oos_metrics,
            }
        )

    if selection_strategy == "oos_aware_constrained":
        return _select_oos_aware_threshold_candidates(
            rows=rows,
            selection_metric=selection_metric,
            max_in_scope_false_oos_rate=max_in_scope_false_oos_rate,
            macro_f1_tolerance_ladder=macro_f1_tolerance_ladder or [0.01, 0.02, 0.03, 0.04, 0.05],
            fallback_strategy=fallback_strategy,
        )

    ranked_rows = sorted(
        rows,
        key=lambda row: _threshold_sort_key(row=row, selection_metric=selection_metric),
        reverse=True,
    )
    selected_threshold = ranked_rows[0]["threshold"] if ranked_rows else None
    for index, row in enumerate(ranked_rows):
        row["selection_eligible"] = index == 0
        row["eligibility_reason"] = "selected_by_metric" if index == 0 else "ranked_below_selected_threshold"

    return ranked_rows, {
        "strategy": "metric",
        "selection_metric": selection_metric,
        "selected_threshold": selected_threshold,
        "fallback_used": False,
    }


def evaluate_distance_threshold_candidates(
    probabilities: np.ndarray,
    nearest_known_intent_distances: np.ndarray,
    y_true: list[int],
    label_names: list[str],
    fixed_oos_confidence_threshold: float,
    distance_threshold_candidates: list[float],
    analysis_top_k_confusions: int,
    analysis_top_k_features_per_class: int,
    selection_metric: str,
    selection_strategy: str = "oos_aware_constrained",
    max_in_scope_false_oos_rate: float = 0.03,
    macro_f1_tolerance_ladder: list[float] | None = None,
    fallback_strategy: str = "best_macro_f1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate centroid-distance thresholds with a fixed probability threshold."""

    rows: list[dict[str, Any]] = []
    for distance_threshold in distance_threshold_candidates:
        evaluation = evaluate_transformer_predictions(
            probabilities=probabilities,
            y_true=y_true,
            label_names=label_names,
            oos_confidence_threshold=fixed_oos_confidence_threshold,
            nearest_known_intent_distances=nearest_known_intent_distances,
            oos_distance_threshold=distance_threshold,
            analysis_top_k_confusions=analysis_top_k_confusions,
            analysis_top_k_features_per_class=analysis_top_k_features_per_class,
        )
        rows.append(
            {
                "threshold": distance_threshold,
                "distance_threshold": distance_threshold,
                "fixed_oos_confidence_threshold": fixed_oos_confidence_threshold,
                "accuracy": evaluation.accuracy,
                "macro_f1": evaluation.macro_f1,
                "top_5_accuracy": evaluation.top_5_accuracy,
                "oos_f1": evaluation.oos_metrics.get("f1", 0.0),
                "in_scope_false_oos_rate": evaluation.oos_metrics.get(
                    "in_scope_false_oos_rate", 0.0
                ),
                "oos_metrics": evaluation.oos_metrics,
            }
        )

    if selection_strategy == "oos_aware_constrained":
        ranked_rows, metadata = _select_oos_aware_threshold_candidates(
            rows=rows,
            selection_metric=selection_metric,
            max_in_scope_false_oos_rate=max_in_scope_false_oos_rate,
            macro_f1_tolerance_ladder=macro_f1_tolerance_ladder or [0.01, 0.02, 0.03, 0.04, 0.05],
            fallback_strategy=fallback_strategy,
        )
    else:
        ranked_rows = sorted(
            rows,
            key=lambda row: _threshold_sort_key(row=row, selection_metric=selection_metric),
            reverse=True,
        )
        selected_threshold = ranked_rows[0]["distance_threshold"] if ranked_rows else None
        for index, row in enumerate(ranked_rows):
            row["selection_eligible"] = index == 0
            row["eligibility_reason"] = (
                "selected_by_metric" if index == 0 else "ranked_below_selected_threshold"
            )
        metadata = {
            "strategy": "metric",
            "selection_metric": selection_metric,
            "selected_threshold": selected_threshold,
            "fallback_used": False,
        }

    metadata["selected_distance_threshold"] = metadata.pop("selected_threshold", None)
    metadata["fixed_oos_confidence_threshold"] = fixed_oos_confidence_threshold
    metadata["distance_threshold_candidates"] = distance_threshold_candidates
    return ranked_rows, metadata


def evaluate_energy_threshold_candidates(
    probabilities: np.ndarray,
    energy_scores: np.ndarray,
    y_true: list[int],
    label_names: list[str],
    fixed_oos_confidence_threshold: float,
    energy_threshold_candidates: list[float],
    analysis_top_k_confusions: int,
    analysis_top_k_features_per_class: int,
    selection_metric: str,
    selection_strategy: str = "oos_aware_constrained",
    max_in_scope_false_oos_rate: float = 0.03,
    macro_f1_tolerance_ladder: list[float] | None = None,
    fallback_strategy: str = "best_macro_f1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate energy thresholds with a fixed probability threshold."""

    rows: list[dict[str, Any]] = []
    for energy_threshold in energy_threshold_candidates:
        evaluation = evaluate_transformer_predictions(
            probabilities=probabilities,
            y_true=y_true,
            label_names=label_names,
            oos_confidence_threshold=fixed_oos_confidence_threshold,
            energy_scores=energy_scores,
            oos_energy_threshold=energy_threshold,
            analysis_top_k_confusions=analysis_top_k_confusions,
            analysis_top_k_features_per_class=analysis_top_k_features_per_class,
        )
        rows.append(
            {
                "threshold": energy_threshold,
                "energy_threshold": energy_threshold,
                "fixed_oos_confidence_threshold": fixed_oos_confidence_threshold,
                "accuracy": evaluation.accuracy,
                "macro_f1": evaluation.macro_f1,
                "top_5_accuracy": evaluation.top_5_accuracy,
                "oos_f1": evaluation.oos_metrics.get("f1", 0.0),
                "in_scope_false_oos_rate": evaluation.oos_metrics.get(
                    "in_scope_false_oos_rate", 0.0
                ),
                "oos_metrics": evaluation.oos_metrics,
            }
        )

    if selection_strategy == "oos_aware_constrained":
        ranked_rows, metadata = _select_oos_aware_threshold_candidates(
            rows=rows,
            selection_metric=selection_metric,
            max_in_scope_false_oos_rate=max_in_scope_false_oos_rate,
            macro_f1_tolerance_ladder=macro_f1_tolerance_ladder or [0.01, 0.02, 0.03, 0.04, 0.05],
            fallback_strategy=fallback_strategy,
        )
    else:
        ranked_rows = sorted(
            rows,
            key=lambda row: _threshold_sort_key(row=row, selection_metric=selection_metric),
            reverse=True,
        )
        selected_threshold = ranked_rows[0]["energy_threshold"] if ranked_rows else None
        for index, row in enumerate(ranked_rows):
            row["selection_eligible"] = index == 0
            row["eligibility_reason"] = (
                "selected_by_metric" if index == 0 else "ranked_below_selected_threshold"
            )
        metadata = {
            "strategy": "metric",
            "selection_metric": selection_metric,
            "selected_threshold": selected_threshold,
            "fallback_used": False,
        }

    metadata["selected_energy_threshold"] = metadata.pop("selected_threshold", None)
    metadata["fixed_oos_confidence_threshold"] = fixed_oos_confidence_threshold
    metadata["energy_threshold_candidates"] = energy_threshold_candidates
    return ranked_rows, metadata


def save_transformer_artifacts(
    transformer_artifacts: TransformerTrainingArtifacts,
    dataset: DatasetBundle,
    config: ExperimentConfig,
    label_names: list[str],
) -> None:
    """Persist a fine-tuned transformer model, tokenizer, and metadata."""

    model_dir = config.artifacts_dir / "transformer_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    transformer_artifacts.model.save_pretrained(model_dir)
    transformer_artifacts.tokenizer.save_pretrained(model_dir)
    if transformer_artifacts.known_intent_centroids is not None:
        np.savez(
            config.artifacts_dir / "known_intent_centroids.npz",
            centroids=transformer_artifacts.known_intent_centroids,
            label_ids=np.asarray(transformer_artifacts.known_intent_centroid_label_ids, dtype=np.int64),
            label_names=np.asarray(transformer_artifacts.known_intent_centroid_label_names, dtype=str),
        )
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
            "oos_confidence_threshold": transformer_artifacts.selected_oos_threshold,
            "oos_distance_threshold": transformer_artifacts.selected_distance_threshold,
            "oos_energy_threshold": transformer_artifacts.selected_energy_threshold,
            "temperature_scaling_enabled": config.transformer.temperature_scaling_enabled,
            "temperature_scaling_grid": config.transformer.temperature_scaling_grid,
            "temperature": transformer_artifacts.temperature,
            "text_column": config.text_column,
            "label_column": config.label_column,
            "random_seed": config.random_seed,
            "transformer_model_name": config.transformer.model_name,
            "max_length": config.transformer.max_length,
            "learning_rate": config.transformer.learning_rate,
            "num_train_epochs": config.transformer.num_train_epochs,
            "train_batch_size": config.transformer.train_batch_size,
            "eval_batch_size": config.transformer.eval_batch_size,
            "weight_decay": config.transformer.weight_decay,
            "warmup_steps": _resolve_warmup_steps(dataset=dataset, config=config),
            "save_strategy": config.transformer.save_strategy,
            "evaluation_strategy": config.transformer.evaluation_strategy,
            "metric_for_best_model": config.transformer.metric_for_best_model,
            "threshold_candidates": config.transformer.threshold_candidates,
            "threshold_selection_metric": config.transformer.threshold_selection_metric,
            "threshold_selection_strategy": config.transformer.threshold_selection_strategy,
            "threshold_max_in_scope_false_oos_rate": config.transformer.threshold_max_in_scope_false_oos_rate,
            "threshold_macro_f1_tolerance_ladder": config.transformer.threshold_macro_f1_tolerance_ladder,
            "threshold_fallback_strategy": config.transformer.threshold_fallback_strategy,
            "threshold_selection_metadata": transformer_artifacts.threshold_selection_metadata,
            "oos_distance_enabled": config.transformer.oos_distance_enabled,
            "oos_distance_metric": config.transformer.oos_distance_metric,
            "oos_distance_candidate_source": config.transformer.oos_distance_candidate_source,
            "oos_distance_selection_strategy": config.transformer.oos_distance_selection_strategy,
            "oos_fixed_probability_source": config.transformer.oos_fixed_probability_source,
            "distance_selection_metadata": transformer_artifacts.distance_selection_metadata,
            "oos_energy_enabled": config.transformer.oos_energy_enabled,
            "oos_energy_temperature": config.transformer.oos_energy_temperature,
            "oos_energy_candidate_source": config.transformer.oos_energy_candidate_source,
            "oos_energy_selection_strategy": config.transformer.oos_energy_selection_strategy,
            "oos_energy_fixed_probability_source": config.transformer.oos_energy_fixed_probability_source,
            "energy_selection_metadata": transformer_artifacts.energy_selection_metadata,
            "pooled_representation": "cls_last_hidden_state",
            "distance_embedding_normalized": config.transformer.oos_distance_metric == "cosine",
            "known_intent_centroid_label_ids": transformer_artifacts.known_intent_centroid_label_ids,
            "known_intent_centroid_label_names": transformer_artifacts.known_intent_centroid_label_names,
            "num_labels": len(label_names),
        },
    )
    _write_json(
        config.artifacts_dir / "threshold_config.json",
        {
            "selected_oos_threshold": transformer_artifacts.selected_oos_threshold,
            "selected_distance_threshold": transformer_artifacts.selected_distance_threshold,
            "selected_energy_threshold": transformer_artifacts.selected_energy_threshold,
            "temperature": transformer_artifacts.temperature,
            "validation_metrics_by_threshold": transformer_artifacts.validation_metrics_by_threshold,
            "selection_metadata": transformer_artifacts.threshold_selection_metadata,
            "validation_metrics_by_distance_threshold": transformer_artifacts.validation_distance_metrics_by_threshold,
            "distance_selection_metadata": transformer_artifacts.distance_selection_metadata,
            "validation_metrics_by_energy_threshold": transformer_artifacts.validation_energy_metrics_by_threshold,
            "energy_selection_metadata": transformer_artifacts.energy_selection_metadata,
        },
    )


def _resolve_warmup_steps(dataset: DatasetBundle, config: ExperimentConfig) -> int:
    """Resolve warmup steps while remaining backward-compatible with ratio-based configs."""

    if config.transformer.warmup_steps > 0:
        return config.transformer.warmup_steps

    if config.transformer.warmup_ratio and config.transformer.warmup_ratio > 0:
        steps_per_epoch = math.ceil(len(dataset.train_texts) / config.transformer.train_batch_size)
        total_steps = max(1, math.ceil(steps_per_epoch * config.transformer.num_train_epochs))
        return max(1, math.ceil(total_steps * config.transformer.warmup_ratio))

    return 0


def _build_trainer(dataset: DatasetBundle, config: ExperimentConfig):
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise ImportError(
            "transformers is required for true fine-tuning experiments. "
            "Install it with `pip install transformers accelerate` or `pip install -e \".[dev]\"`."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(config.transformer.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.transformer.model_name,
        num_labels=len(dataset.label_names),
        id2label={index: label for index, label in enumerate(dataset.label_names)},
        label2id={label: index for index, label in enumerate(dataset.label_names)},
    )

    train_dataset = _build_hf_dataset(
        texts=dataset.train_texts,
        labels=dataset.train_labels,
        tokenizer=tokenizer,
        max_length=config.transformer.max_length,
    )
    eval_dataset = _build_hf_dataset(
        texts=dataset.validation_texts,
        labels=dataset.validation_labels,
        tokenizer=tokenizer,
        max_length=config.transformer.max_length,
    )

    training_args_kwargs = {
        "output_dir": str(config.artifacts_dir / "training"),
        "learning_rate": config.transformer.learning_rate,
        "num_train_epochs": config.transformer.num_train_epochs,
        "per_device_train_batch_size": config.transformer.train_batch_size,
        "per_device_eval_batch_size": config.transformer.eval_batch_size,
        "weight_decay": config.transformer.weight_decay,
        "warmup_steps": _resolve_warmup_steps(dataset=dataset, config=config),
        "save_strategy": config.transformer.save_strategy,
        "load_best_model_at_end": config.transformer.load_best_model_at_end,
        "metric_for_best_model": config.transformer.metric_for_best_model,
        "greater_is_better": config.transformer.greater_is_better,
        "fp16": config.transformer.fp16,
        "bf16": config.transformer.bf16,
        "report_to": [],
        "logging_strategy": "epoch",
    }
    training_arguments_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_arguments_signature.parameters:
        training_args_kwargs["evaluation_strategy"] = config.transformer.evaluation_strategy
    elif "eval_strategy" in training_arguments_signature.parameters:
        training_args_kwargs["eval_strategy"] = config.transformer.evaluation_strategy
    else:
        raise TypeError(
            "Unsupported transformers TrainingArguments signature: neither "
            "`evaluation_strategy` nor `eval_strategy` is available."
        )

    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset if len(dataset.validation_texts) > 0 else None,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": _build_compute_metrics(dataset.label_names),
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise TypeError(
            "Unsupported transformers Trainer signature: neither `processing_class` "
            "nor `tokenizer` is available."
        )

    trainer = Trainer(**trainer_kwargs)
    return tokenizer, model, trainer


def _build_hf_dataset(texts: list[str], labels: list[int], tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_dict({"text": texts, "label": labels})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized


def _build_compute_metrics(label_names: list[str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        report = classification_report(
            labels,
            predictions,
            labels=list(range(len(label_names))),
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": float(report["macro avg"]["f1-score"]),
        }

    return compute_metrics


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled_logits = logits / temperature
    shifted = scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def _select_oos_aware_threshold_candidates(
    rows: list[dict[str, Any]],
    selection_metric: str,
    max_in_scope_false_oos_rate: float,
    macro_f1_tolerance_ladder: list[float],
    fallback_strategy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    best_macro_f1 = max((row["macro_f1"] for row in rows), default=0.0)
    selected_row: dict[str, Any] | None = None
    successful_tolerance: float | None = None

    for tolerance in macro_f1_tolerance_ladder:
        eligible_rows = [
            row
            for row in rows
            if row["in_scope_false_oos_rate"] <= max_in_scope_false_oos_rate
            and row["macro_f1"] >= best_macro_f1 - tolerance
        ]
        if eligible_rows:
            selected_row = max(eligible_rows, key=_oos_constrained_sort_key)
            successful_tolerance = tolerance
            break

    fallback_used = False
    if selected_row is None:
        fallback_used = True
        if fallback_strategy != "best_macro_f1":
            raise ValueError(f"Unsupported threshold fallback strategy: {fallback_strategy}")
        selected_row = max(rows, key=lambda row: _threshold_sort_key(row=row, selection_metric=selection_metric))

    annotated_rows: list[dict[str, Any]] = []
    comparison_tolerance = successful_tolerance if successful_tolerance is not None else max(macro_f1_tolerance_ladder, default=0.05)
    for row in rows:
        annotated_row = dict(row)
        reasons: list[str] = []
        if row["in_scope_false_oos_rate"] > max_in_scope_false_oos_rate:
            reasons.append("in_scope_false_oos_rate_exceeds_limit")
        if row["macro_f1"] < best_macro_f1 - comparison_tolerance:
            reasons.append("macro_f1_below_tolerance")
        if row["threshold"] == selected_row["threshold"]:
            annotated_row["selection_eligible"] = not fallback_used or not reasons
            annotated_row["eligibility_reason"] = "selected_by_oos_aware_constraints" if not fallback_used else "fallback_best_macro_f1"
        else:
            annotated_row["selection_eligible"] = len(reasons) == 0 and not fallback_used
            annotated_row["eligibility_reason"] = "eligible_not_selected" if annotated_row["selection_eligible"] else ",".join(reasons) if reasons else "ranked_below_selected_threshold"
        annotated_rows.append(annotated_row)

    selected_threshold = selected_row["threshold"]
    annotated_rows = sorted(
        annotated_rows,
        key=lambda row: (row["threshold"] != selected_threshold, -row["oos_f1"], -row["macro_f1"], row["in_scope_false_oos_rate"], row["threshold"]),
    )

    return annotated_rows, {
        "strategy": "oos_aware_constrained",
        "selection_metric": selection_metric,
        "selected_threshold": selected_threshold,
        "best_macro_f1": best_macro_f1,
        "max_in_scope_false_oos_rate": max_in_scope_false_oos_rate,
        "macro_f1_tolerance_ladder": macro_f1_tolerance_ladder,
        "successful_tolerance": successful_tolerance,
        "fallback_strategy": fallback_strategy,
        "fallback_used": fallback_used,
    }


def _oos_constrained_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        row["oos_f1"],
        row["macro_f1"],
        -row["in_scope_false_oos_rate"],
        -row["threshold"],
    )


def _threshold_sort_key(row: dict[str, Any], selection_metric: str) -> tuple:
    if selection_metric == "oos_recall":
        return (
            row["oos_metrics"].get("recall", 0.0),
            row["macro_f1"],
            row["accuracy"],
        )
    if selection_metric == "oos_f1":
        return (
            row["oos_metrics"].get("f1", 0.0),
            row["macro_f1"],
            row["accuracy"],
        )
    return (
        row["macro_f1"],
        row["oos_metrics"].get("recall", 0.0),
        row["accuracy"],
    )


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")
