"""Hyperparameter search workflow for Banking77 experiments."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from banking77_intent_classifier.artifacts import (
    ensure_output_directories,
    save_evaluation_reports,
    save_model_artifacts,
    save_tuning_reports,
)
from banking77_intent_classifier.config import load_tuning_config
from banking77_intent_classifier.data import load_banking77_dataset
from banking77_intent_classifier.evaluation import evaluate_predictions, save_confusion_matrix_figure
from banking77_intent_classifier.modeling import (
    build_pipeline,
    extract_weight_export,
    predict_top_k_labels,
)
from banking77_intent_classifier.pipeline import configure_logging


LOGGER = logging.getLogger(__name__)


def run_search_pipeline(config_path: str | Path) -> dict:
    """Run hyperparameter search on the train split and evaluate the best model on the test split."""

    configure_logging()
    tuning_config = load_tuning_config(config_path)
    experiment_config = tuning_config.experiment
    LOGGER.info("Loaded tuning config from %s", config_path)
    LOGGER.info("Resolved experiment configuration: %s", asdict(experiment_config))
    LOGGER.info("Resolved search configuration: %s", asdict(tuning_config.search))

    ensure_output_directories(experiment_config)

    dataset = load_banking77_dataset(
        dataset_name=experiment_config.dataset_name,
        train_split=experiment_config.train_split,
        test_split=experiment_config.test_split,
        text_column=experiment_config.text_column,
        label_column=experiment_config.label_column,
    )
    LOGGER.info(
        "Loaded dataset with %d train samples, %d test samples, and %d labels",
        len(dataset.train_texts),
        len(dataset.test_texts),
        len(dataset.label_names),
    )

    pipeline = build_pipeline(
        model_family=experiment_config.model_family,
        tfidf_config=experiment_config.tfidf,
        encoder_config=experiment_config.encoder,
        classifier_config=experiment_config.classifier,
    )
    search = _build_search(pipeline=pipeline, tuning_config=tuning_config)
    search.fit(dataset.train_texts, dataset.train_labels)
    LOGGER.info("Hyperparameter search complete. Best score: %.6f", search.best_score_)
    LOGGER.info("Best parameters: %s", search.best_params_)

    best_pipeline = search.best_estimator_
    predictions = best_pipeline.predict(dataset.test_texts)
    top_5_predictions = predict_top_k_labels(best_pipeline, dataset.test_texts, k=5)
    weight_export = extract_weight_export(best_pipeline)
    evaluation = evaluate_predictions(
        y_true=dataset.test_labels,
        y_pred=predictions.tolist(),
        top_k_predicted_labels=top_5_predictions,
        label_names=dataset.label_names,
        coefficients=weight_export.coefficients.toarray() if weight_export.coefficients is not None else None,
        feature_names=weight_export.feature_names,
        top_k_confusions=experiment_config.analysis.top_k_confusions,
        top_k_features_per_class=experiment_config.analysis.top_k_features_per_class,
    )

    cv_results_df = pd.DataFrame(search.cv_results_).sort_values(
        by="rank_test_score",
        ascending=True,
    )
    run_summary = {
        "accuracy": evaluation.accuracy,
        "macro_f1": evaluation.macro_f1,
        "top_5_accuracy": evaluation.top_5_accuracy,
        "best_cv_score": float(search.best_score_),
        "best_params": search.best_params_,
        "artifacts_dir": str(experiment_config.artifacts_dir),
        "reports_dir": str(experiment_config.reports_dir),
        "top_confusions_rows": len(evaluation.top_confusions_df),
        "train_samples": len(dataset.train_texts),
        "test_samples": len(dataset.test_texts),
        "label_count": len(dataset.label_names),
    }

    save_model_artifacts(
        pipeline=best_pipeline,
        config=experiment_config,
        label_names=dataset.label_names,
        weight_export=weight_export,
    )
    save_evaluation_reports(
        evaluation=evaluation,
        config=experiment_config,
        run_summary=run_summary,
    )
    save_confusion_matrix_figure(
        confusion_matrix_df=evaluation.confusion_matrix_df,
        output_path=experiment_config.reports_dir / "confusion_matrix.png",
    )
    save_tuning_reports(
        config=experiment_config,
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        cv_results_df=cv_results_df,
        search_settings=asdict(tuning_config.search),
    )

    LOGGER.info("Saved tuned artifacts to %s", experiment_config.artifacts_dir)
    LOGGER.info("Saved tuned reports to %s", experiment_config.reports_dir)

    return run_summary


def _build_search(pipeline, tuning_config):
    if tuning_config.search.search_type != "randomized":
        raise ValueError(
            f"Unsupported search_type: {tuning_config.search.search_type}. "
            "This workflow currently supports only 'randomized'."
        )

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=tuning_config.search.param_distributions,
        n_iter=tuning_config.search.n_iter,
        cv=tuning_config.search.cv,
        scoring=tuning_config.search.scoring,
        n_jobs=tuning_config.search.n_jobs,
        verbose=tuning_config.search.verbose,
        refit=tuning_config.search.refit,
        random_state=tuning_config.search.random_state,
    )
