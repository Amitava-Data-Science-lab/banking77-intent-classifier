"""End-to-end orchestration for Banking77 model training."""

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
from banking77_intent_classifier.data import load_banking77_dataset
from banking77_intent_classifier.evaluation import evaluate_predictions, save_confusion_matrix_figure
from banking77_intent_classifier.modeling import (
    build_pipeline,
    extract_weight_export,
    predict_top_k_labels,
    train_pipeline,
)
from banking77_intent_classifier.reranking import (
    CrossEncoderReranker,
    build_label_text_mapping,
    rerank_top_k_predictions,
)


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure consistent application logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_training_pipeline(config_path: str | Path) -> dict:
    """Run a complete experiment and return a compact summary."""

    configure_logging()
    config = load_config(config_path)
    LOGGER.info("Loaded config from %s", config_path)
    LOGGER.info("Resolved configuration: %s", asdict(config))

    ensure_output_directories(config)

    dataset = load_banking77_dataset(
        dataset_name=config.dataset_name,
        train_split=config.train_split,
        test_split=config.test_split,
        text_column=config.text_column,
        label_column=config.label_column,
    )
    LOGGER.info(
        "Loaded dataset with %d train samples, %d test samples, and %d labels",
        len(dataset.train_texts),
        len(dataset.test_texts),
        len(dataset.label_names),
    )

    pipeline = build_pipeline(
        model_family=config.model_family,
        tfidf_config=config.tfidf,
        encoder_config=config.encoder,
        classifier_config=config.classifier,
    )
    trained_pipeline = train_pipeline(pipeline, dataset.train_texts, dataset.train_labels)
    predictions = trained_pipeline.predict(dataset.test_texts)
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
        "model_family": config.model_family,
        "encoder_model_name": config.encoder.model_name if config.model_family.startswith("sentence_transformer_") else None,
        "reranker_model_name": config.reranker.model_name if config.reranker.enabled else None,
        "artifacts_dir": str(config.artifacts_dir),
        "reports_dir": str(config.reports_dir),
        "top_confusions_rows": len(evaluation.top_confusions_df),
        "train_samples": len(dataset.train_texts),
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
