"""Model factory and training helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from banking77_intent_classifier.config import ClassifierConfig, EncoderConfig, TfidfConfig
from banking77_intent_classifier.encoders import SentenceTransformerEncoder
from banking77_intent_classifier.preprocessing import SpacyLemmatizerTokenizer


@dataclass(slots=True)
class WeightExport:
    """Container for linear model coefficients and feature names."""

    coefficients: sparse.csr_matrix | None
    feature_names: np.ndarray


def build_pipeline(
    model_family: str,
    tfidf_config: TfidfConfig,
    encoder_config: EncoderConfig,
    classifier_config: ClassifierConfig,
) -> Pipeline:
    """Build a scikit-learn pipeline for the requested experiment family."""

    classifier = LinearSVC(
        C=classifier_config.c,
        class_weight=classifier_config.class_weight,
        max_iter=classifier_config.max_iter,
        dual=classifier_config.dual,
        random_state=classifier_config.random_state,
    )

    if model_family == "tfidf_svc":
        vectorizer_kwargs = {
            "lowercase": tfidf_config.lowercase,
            "strip_accents": tfidf_config.strip_accents,
            "ngram_range": tfidf_config.ngram_range,
            "min_df": tfidf_config.min_df,
            "max_df": tfidf_config.max_df,
            "sublinear_tf": tfidf_config.sublinear_tf,
            "max_features": tfidf_config.max_features,
        }
        if tfidf_config.normalization == "lemma":
            vectorizer_kwargs["tokenizer"] = SpacyLemmatizerTokenizer()
            vectorizer_kwargs["token_pattern"] = None
        elif tfidf_config.normalization != "none":
            raise ValueError(f"Unsupported TF-IDF normalization mode: {tfidf_config.normalization}")

        vectorizer = TfidfVectorizer(
            **vectorizer_kwargs,
        )
        return Pipeline(
            steps=[
                ("vectorizer", vectorizer),
                ("classifier", classifier),
            ]
        )

    if model_family == "sentence_transformer_linear":
        encoder = SentenceTransformerEncoder(
            model_name=encoder_config.model_name,
            batch_size=encoder_config.batch_size,
            normalize_embeddings=encoder_config.normalize_embeddings,
            device=encoder_config.device,
        )
        return Pipeline(
            steps=[
                ("encoder", encoder),
                ("classifier", classifier),
            ]
        )

    if model_family == "sentence_transformer_linear_reranked":
        encoder = SentenceTransformerEncoder(
            model_name=encoder_config.model_name,
            batch_size=encoder_config.batch_size,
            normalize_embeddings=encoder_config.normalize_embeddings,
            device=encoder_config.device,
        )
        return Pipeline(
            steps=[
                ("encoder", encoder),
                ("classifier", classifier),
            ]
        )

    if model_family == "sentence_transformer_knn":
        encoder = SentenceTransformerEncoder(
            model_name=encoder_config.model_name,
            batch_size=encoder_config.batch_size,
            normalize_embeddings=encoder_config.normalize_embeddings,
            device=encoder_config.device,
        )
        knn_classifier = KNeighborsClassifier(
            n_neighbors=classifier_config.knn_neighbors,
            weights=classifier_config.knn_weights,
            metric=classifier_config.knn_metric,
        )
        return Pipeline(
            steps=[
                ("encoder", encoder),
                ("classifier", knn_classifier),
            ]
        )

    raise ValueError(f"Unsupported model family: {model_family}")


def train_pipeline(pipeline: Pipeline, texts: list[str], labels: list[int]) -> Pipeline:
    """Fit the pipeline on the provided dataset."""

    return pipeline.fit(texts, labels)


def extract_weight_export(pipeline: Pipeline) -> WeightExport:
    """Extract feature names and linear SVM coefficients for persistence."""

    if "vectorizer" in pipeline.named_steps:
        classifier: LinearSVC = pipeline.named_steps["classifier"]
        coefficients = sparse.csr_matrix(classifier.coef_)
        vectorizer: TfidfVectorizer = pipeline.named_steps["vectorizer"]
        feature_names = vectorizer.get_feature_names_out()
    elif "encoder" in pipeline.named_steps:
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "coef_"):
            coefficients = sparse.csr_matrix(classifier.coef_)
            embedding_dim = coefficients.shape[1]
        elif hasattr(classifier, "_fit_X"):
            embedding_dim = classifier._fit_X.shape[1]
            coefficients = None
        else:
            raise ValueError("Unsupported classifier for embedding pipeline.")
        feature_names = np.array([f"embedding_{index}" for index in range(embedding_dim)])
    else:
        raise ValueError("Unsupported pipeline shape for weight export.")

    return WeightExport(coefficients=coefficients, feature_names=feature_names)


def predict_top_k_labels(pipeline: Pipeline, texts: list[str], k: int) -> np.ndarray:
    """Return the top-k ranked label ids for each input text."""

    classifier = pipeline.named_steps["classifier"]
    class_labels = classifier.classes_

    if hasattr(pipeline, "decision_function"):
        decision_scores = pipeline.decision_function(texts)

        if decision_scores.ndim == 1:
            positive_class = int(class_labels[1])
            negative_class = int(class_labels[0])
            predicted_positive = decision_scores >= 0
            ranked = np.where(
                predicted_positive[:, None],
                np.array([positive_class, negative_class]),
                np.array([negative_class, positive_class]),
            )
            return ranked[:, : min(k, ranked.shape[1])]

        top_k = min(k, decision_scores.shape[1])
        top_indices = np.argsort(decision_scores, axis=1)[:, -top_k:][:, ::-1]
        return class_labels[top_indices]

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(texts)
        top_k = min(k, probabilities.shape[1])
        top_indices = np.argsort(probabilities, axis=1)[:, -top_k:][:, ::-1]
        return class_labels[top_indices]

    raise ValueError("Pipeline does not support ranked prediction outputs.")
