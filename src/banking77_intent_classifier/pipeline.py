"""End-to-end orchestration for dataset-driven model training."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from banking77_intent_classifier.artifacts import (
    ensure_output_directories,
    save_evaluation_reports,
    save_model_artifacts,
)
from banking77_intent_classifier.config import load_config
from banking77_intent_classifier.data import load_dataset_bundle
from banking77_intent_classifier.evaluation import evaluate_predictions, save_confusion_matrix_figure
from banking77_intent_classifier.modeling import (
    build_pipeline,
    extract_weight_export,
    predict_labels,
    predict_top_k_labels,
    train_pipeline,
)
from banking77_intent_classifier.reranking import (
    CrossEncoderReranker,
    build_label_text_mapping,
    rerank_top_k_predictions,
)
from banking77_intent_classifier.transformer_modeling import (
    compute_energy_scores,
    compute_nearest_known_intent_distances,
    evaluate_transformer_predictions,
    predict_embeddings,
    predict_logits,
    predict_probabilities,
    save_transformer_artifacts,
    train_transformer_classifier,
)


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure consistent application logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_training_pipeline(
    config_path: str | Path,
    oos_threshold_override: float | None = None,
    oos_margin_threshold_override: float | None = None,
    output_suffix: str | None = None,
) -> dict:
    """Run a complete experiment and return a compact summary."""

    configure_logging()
    config = load_config(config_path)
    if oos_threshold_override is not None:
        config.oos_confidence_threshold = oos_threshold_override
    if oos_margin_threshold_override is not None:
        config.oos_margin_threshold = oos_margin_threshold_override
    if output_suffix is not None:
        config.artifacts_dir = config.artifacts_dir.parent / f"{config.artifacts_dir.name}_{output_suffix}"
        config.reports_dir = config.reports_dir.parent / f"{config.reports_dir.name}_{output_suffix}"
    LOGGER.info("Loaded config from %s", config_path)
    LOGGER.info("Resolved configuration: %s", asdict(config))

    ensure_output_directories(config)

    dataset = load_dataset_bundle(
        dataset_type=config.dataset_type,
        dataset_name=config.dataset_name,
        dataset_task=config.dataset_task,
        dataset_source=str(config.dataset_source) if config.dataset_source is not None else None,
        train_split=config.train_split,
        validation_split=config.validation_split,
        test_split=config.test_split,
        text_column=config.text_column,
        label_column=config.label_column,
        include_oos=config.include_oos,
    )
    LOGGER.info(
        "Loaded %s dataset with %d train samples, %d validation samples, %d test samples, and %d labels",
        config.dataset_type,
        len(dataset.train_texts),
        len(dataset.validation_texts),
        len(dataset.test_texts),
        len(dataset.label_names),
    )

    if config.model_family == "transformer_sequence_classifier":
        transformer_artifacts = train_transformer_classifier(dataset=dataset, config=config)
        # The threshold remains an evaluation-time rule even for fine-tuned models.
        config.oos_confidence_threshold = transformer_artifacts.selected_oos_threshold
        test_nearest_distances = None
        test_logits = predict_logits(
            trainer=transformer_artifacts.trainer,
            texts=dataset.test_texts,
            labels=dataset.test_labels,
            tokenizer=transformer_artifacts.tokenizer,
            transformer_config=config.transformer,
        )
        test_probabilities = predict_probabilities(
            trainer=transformer_artifacts.trainer,
            texts=dataset.test_texts,
            labels=dataset.test_labels,
            tokenizer=transformer_artifacts.tokenizer,
            transformer_config=config.transformer,
        )
        test_energy_scores = None
        if (
            config.transformer.oos_distance_enabled
            and transformer_artifacts.known_intent_centroids is not None
        ):
            test_embeddings = predict_embeddings(
                model=transformer_artifacts.model,
                tokenizer=transformer_artifacts.tokenizer,
                texts=dataset.test_texts,
                transformer_config=config.transformer,
            )
            test_nearest_distances = compute_nearest_known_intent_distances(
                embeddings=test_embeddings,
                known_intent_centroids=transformer_artifacts.known_intent_centroids,
                distance_metric=config.transformer.oos_distance_metric,
            )
        if config.transformer.oos_energy_enabled:
            test_energy_scores = compute_energy_scores(
                logits=test_logits,
                temperature=config.transformer.oos_energy_temperature,
            )
        evaluation = evaluate_transformer_predictions(
            probabilities=test_probabilities,
            y_true=dataset.test_labels,
            label_names=dataset.label_names,
            oos_confidence_threshold=config.oos_confidence_threshold,
            nearest_known_intent_distances=test_nearest_distances,
            oos_distance_threshold=transformer_artifacts.selected_distance_threshold,
            energy_scores=test_energy_scores,
            oos_energy_threshold=transformer_artifacts.selected_energy_threshold,
            analysis_top_k_confusions=config.analysis.top_k_confusions,
            analysis_top_k_features_per_class=config.analysis.top_k_features_per_class,
        )
        run_summary = {
            "accuracy": evaluation.accuracy,
            "macro_f1": evaluation.macro_f1,
            "top_5_accuracy": evaluation.top_5_accuracy,
            "oos_metrics": evaluation.oos_metrics,
            "model_family": config.model_family,
            "dataset_type": config.dataset_type,
            "dataset_task": config.dataset_task,
            "oos_confidence_threshold": config.oos_confidence_threshold,
            "oos_distance_threshold": transformer_artifacts.selected_distance_threshold,
            "oos_energy_threshold": transformer_artifacts.selected_energy_threshold,
            "oos_margin_threshold": None,
            "encoder_model_name": None,
            "transformer_model_name": config.transformer.model_name,
            "distance_metric": config.transformer.oos_distance_metric if config.transformer.oos_distance_enabled else None,
            "distance_candidate_source": config.transformer.oos_distance_candidate_source if config.transformer.oos_distance_enabled else None,
            "energy_temperature": config.transformer.oos_energy_temperature if config.transformer.oos_energy_enabled else None,
            "energy_candidate_source": config.transformer.oos_energy_candidate_source if config.transformer.oos_energy_enabled else None,
            "reranker_model_name": None,
            "artifacts_dir": str(config.artifacts_dir),
            "reports_dir": str(config.reports_dir),
            "top_confusions_rows": len(evaluation.top_confusions_df),
            "train_samples": len(dataset.train_texts),
            "validation_samples": len(dataset.validation_texts),
            "test_samples": len(dataset.test_texts),
            "label_count": len(dataset.label_names),
            "validation_threshold_candidates": transformer_artifacts.validation_metrics_by_threshold,
            "validation_distance_threshold_candidates": transformer_artifacts.validation_distance_metrics_by_threshold,
            "validation_energy_threshold_candidates": transformer_artifacts.validation_energy_metrics_by_threshold,
        }
        save_transformer_artifacts(
            transformer_artifacts=transformer_artifacts,
            dataset=dataset,
            config=config,
            label_names=dataset.label_names,
        )
        save_evaluation_reports(evaluation=evaluation, config=config, run_summary=run_summary)
        save_confusion_matrix_figure(
            confusion_matrix_df=evaluation.confusion_matrix_df,
            output_path=config.reports_dir / "confusion_matrix.png",
            title="CLINC150 Transformer Confusion Matrix",
        )
        LOGGER.info("Saved transformer artifacts to %s", config.artifacts_dir)
        LOGGER.info("Saved transformer reports to %s", config.reports_dir)
        return run_summary

    pipeline = build_pipeline(
        model_family=config.model_family,
        tfidf_config=config.tfidf,
        encoder_config=config.encoder,
        classifier_config=config.classifier,
        require_probabilities=(
            config.oos_confidence_threshold is not None or config.oos_margin_threshold is not None
        ),
    )
    trained_pipeline = train_pipeline(pipeline, dataset.train_texts, dataset.train_labels)
    oos_label_id = dataset.label_names.index("oos") if "oos" in dataset.label_names else None
    predictions = predict_labels(
        trained_pipeline,
        dataset.test_texts,
        oos_label_id=oos_label_id,
        oos_confidence_threshold=config.oos_confidence_threshold,
        oos_margin_threshold=config.oos_margin_threshold,
    )
    top_k = config.reranker.top_k if config.reranker.enabled else 5
    top_5_predictions = predict_top_k_labels(trained_pipeline, dataset.test_texts, k=top_k)

    label_text_mapping = build_label_text_mapping(dataset.label_names)
    if config.model_family == "sentence_transformer_linear_reranked" and config.reranker.enabled:
        reranker = CrossEncoderReranker(config.reranker.model_name)
        top_5_predictions = rerank_top_k_predictions(
            texts=dataset.test_texts,
            top_k_predicted_labels=top_5_predictions,
            label_text_mapping=label_text_mapping,
            reranker=reranker,
        )
        predictions = top_5_predictions[:, 0]

    weight_export = extract_weight_export(trained_pipeline)
    evaluation = evaluate_predictions(
        y_true=dataset.test_labels,
        y_pred=predictions.tolist(),
        top_k_predicted_labels=top_5_predictions,
        label_names=dataset.label_names,
        coefficients=weight_export.coefficients.toarray() if weight_export.coefficients is not None else None,
        feature_names=weight_export.feature_names,
        top_k_confusions=config.analysis.top_k_confusions,
        top_k_features_per_class=config.analysis.top_k_features_per_class,
    )
    run_summary = {
        "accuracy": evaluation.accuracy,
        "macro_f1": evaluation.macro_f1,
        "top_5_accuracy": evaluation.top_5_accuracy,
        "oos_metrics": evaluation.oos_metrics,
        "model_family": config.model_family,
        "dataset_type": config.dataset_type,
        "dataset_task": config.dataset_task,
        "oos_confidence_threshold": config.oos_confidence_threshold,
        "oos_margin_threshold": config.oos_margin_threshold,
        "encoder_model_name": config.encoder.model_name if config.model_family.startswith("sentence_transformer_") else None,
        "reranker_model_name": config.reranker.model_name if config.reranker.enabled else None,
        "artifacts_dir": str(config.artifacts_dir),
        "reports_dir": str(config.reports_dir),
        "top_confusions_rows": len(evaluation.top_confusions_df),
        "train_samples": len(dataset.train_texts),
        "validation_samples": len(dataset.validation_texts),
        "test_samples": len(dataset.test_texts),
        "label_count": len(dataset.label_names),
    }

    save_model_artifacts(
        pipeline=trained_pipeline,
        config=config,
        label_names=dataset.label_names,
        weight_export=weight_export,
        label_text_mapping=label_text_mapping,
    )
    save_evaluation_reports(evaluation=evaluation, config=config, run_summary=run_summary)
    save_confusion_matrix_figure(
        confusion_matrix_df=evaluation.confusion_matrix_df,
        output_path=config.reports_dir / "confusion_matrix.png",
    )

    LOGGER.info("Saved artifacts to %s", config.artifacts_dir)
    LOGGER.info("Saved reports to %s", config.reports_dir)

    return run_summary
