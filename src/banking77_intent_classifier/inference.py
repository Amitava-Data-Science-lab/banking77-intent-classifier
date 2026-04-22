"""Inference helpers for loading and serving the trained classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from banking77_intent_classifier.modeling import predict_top_k_labels
from banking77_intent_classifier.reranking import CrossEncoderReranker, rerank_top_k_predictions
from banking77_intent_classifier.transformer_modeling import apply_probability_threshold


@dataclass(slots=True)
class IntentPrediction:
    """A single intent prediction."""

    label_id: int
    label: str


class Predictor:
    """Thin reusable wrapper around the serialized scikit-learn pipeline."""

    def __init__(self, pipeline: Pipeline, label_mapping: dict[int, str]) -> None:
        self._pipeline = pipeline
        self._label_mapping = label_mapping

    def predict_one(self, text: str) -> IntentPrediction:
        """Predict a single intent."""

        label_id = int(self._pipeline.predict([text])[0])
        return IntentPrediction(label_id=label_id, label=self._label_mapping[label_id])

    def predict_many(self, texts: list[str]) -> list[IntentPrediction]:
        """Predict intents for multiple texts."""

        predictions = self._pipeline.predict(texts)
        return [
            IntentPrediction(label_id=int(label_id), label=self._label_mapping[int(label_id)])
            for label_id in predictions
        ]


class RerankingPredictor(Predictor):
    """Predictor that reranks top-k base model candidates with a cross-encoder."""

    def __init__(
        self,
        pipeline: Pipeline,
        label_mapping: dict[int, str],
        label_text_mapping: dict[int, str],
        reranker_model_name: str,
        top_k: int,
    ) -> None:
        super().__init__(pipeline=pipeline, label_mapping=label_mapping)
        self._label_text_mapping = label_text_mapping
        self._reranker_model_name = reranker_model_name
        self._reranker: CrossEncoderReranker | None = None
        self._top_k = top_k

    def predict_one(self, text: str) -> IntentPrediction:
        reranked = rerank_top_k_predictions(
            texts=[text],
            top_k_predicted_labels=predict_top_k_labels(self._pipeline, [text], k=self._top_k),
            label_text_mapping=self._label_text_mapping,
            reranker=self._get_reranker(),
        )
        label_id = int(reranked[0][0])
        return IntentPrediction(label_id=label_id, label=self._label_mapping[label_id])

    def predict_many(self, texts: list[str]) -> list[IntentPrediction]:
        reranked = rerank_top_k_predictions(
            texts=texts,
            top_k_predicted_labels=predict_top_k_labels(self._pipeline, texts, k=self._top_k),
            label_text_mapping=self._label_text_mapping,
            reranker=self._get_reranker(),
        )
        return [
            IntentPrediction(label_id=int(label_id), label=self._label_mapping[int(label_id)])
            for label_id in reranked[:, 0]
        ]

    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self._reranker_model_name)
        return self._reranker


class TransformerPredictor:
    """Predictor for fine-tuned transformer sequence classifiers."""

    def __init__(
        self,
        model,
        tokenizer,
        label_mapping: dict[int, str],
        oos_confidence_threshold: float | None = None,
        max_length: int = 128,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._label_mapping = label_mapping
        self._oos_confidence_threshold = oos_confidence_threshold
        self._max_length = max_length

    def predict_one(self, text: str) -> IntentPrediction:
        return self.predict_many([text])[0]

    def predict_many(self, texts: list[str]) -> list[IntentPrediction]:
        encoded = self._tokenizer(
            texts,
            truncation=True,
            max_length=self._max_length,
            padding=True,
            return_tensors="pt",
        )
        outputs = self._model(**encoded)
        logits = outputs.logits.detach().cpu().numpy()
        probabilities = _softmax(logits)
        label_names = [self._label_mapping[index] for index in sorted(self._label_mapping)]
        predicted_ids = apply_probability_threshold(
            probabilities=probabilities,
            label_names=label_names,
            oos_confidence_threshold=self._oos_confidence_threshold,
        )
        return [
            IntentPrediction(label_id=int(label_id), label=self._label_mapping[int(label_id)])
            for label_id in predicted_ids
        ]


def load_predictor(model_path: str | Path, label_mapping_path: str | Path) -> Predictor:
    """Load a persisted model pipeline and label mapping."""

    pipeline = joblib.load(model_path)
    with Path(label_mapping_path).open("r", encoding="utf-8") as file_handle:
        raw_mapping = json.load(file_handle)
    label_mapping = {int(label_id): label for label_id, label in raw_mapping.items()}
    return Predictor(pipeline=pipeline, label_mapping=label_mapping)


def load_reranking_predictor(
    model_path: str | Path,
    label_mapping_path: str | Path,
    label_text_mapping_path: str | Path,
    reranker_model_name: str,
    top_k: int,
) -> RerankingPredictor:
    """Load a reranking predictor composed of a base model and cross-encoder."""

    pipeline = joblib.load(model_path)
    with Path(label_mapping_path).open("r", encoding="utf-8") as file_handle:
        raw_mapping = json.load(file_handle)
    with Path(label_text_mapping_path).open("r", encoding="utf-8") as file_handle:
        raw_label_text_mapping = json.load(file_handle)

    label_mapping = {int(label_id): label for label_id, label in raw_mapping.items()}
    label_text_mapping = {
        int(label_id): label_text for label_id, label_text in raw_label_text_mapping.items()
    }
    return RerankingPredictor(
        pipeline=pipeline,
        label_mapping=label_mapping,
        label_text_mapping=label_text_mapping,
        reranker_model_name=reranker_model_name,
        top_k=top_k,
    )


def load_transformer_predictor(
    model_dir: str | Path,
    label_mapping_path: str | Path,
    oos_confidence_threshold: float | None = None,
    max_length: int = 128,
) -> TransformerPredictor:
    """Load a persisted transformer classifier and label mapping."""

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "transformers is required to load a fine-tuned transformer predictor. "
            "Install it with `pip install transformers accelerate`."
        ) from error

    with Path(label_mapping_path).open("r", encoding="utf-8") as file_handle:
        raw_mapping = json.load(file_handle)
    label_mapping = {int(label_id): label for label_id, label in raw_mapping.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return TransformerPredictor(
        model=model,
        tokenizer=tokenizer,
        label_mapping=label_mapping,
        oos_confidence_threshold=oos_confidence_threshold,
        max_length=max_length,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)
