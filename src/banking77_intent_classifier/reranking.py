"""Cross-encoder reranking helpers for top-k intent candidates."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_label_text_mapping(label_names: list[str]) -> dict[int, str]:
    """Convert intent labels into readable text for reranking."""

    return {index: label_name.replace("_", " ") for index, label_name in enumerate(label_names)}


class CrossEncoderReranker:
    """Wrapper around a sentence-transformers cross-encoder for candidate reranking."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def rerank(
        self,
        text: str,
        candidate_label_ids: np.ndarray,
        label_text_mapping: dict[int, str],
    ) -> np.ndarray:
        model = self._get_model()
        pairs = [(text, label_text_mapping[int(label_id)]) for label_id in candidate_label_ids.tolist()]
        scores = model.predict(pairs)
        top_indices = np.argsort(scores)[::-1]
        return candidate_label_ids[top_indices]

    def __getstate__(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "_model": None}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.model_name = state["model_name"]
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise ImportError(
                    "sentence-transformers is required for reranking experiments. "
                    "Install it with `pip install sentence-transformers` or "
                    "`pip install -e \".[dev]\"`."
                ) from error
            self._model = CrossEncoder(self.model_name)
        return self._model


def rerank_top_k_predictions(
    texts: list[str],
    top_k_predicted_labels: np.ndarray,
    label_text_mapping: dict[int, str],
    reranker: CrossEncoderReranker,
) -> np.ndarray:
    """Rerank top-k predictions for each input text using a cross-encoder."""

    reranked_rows = [
        reranker.rerank(
            text=text,
            candidate_label_ids=candidate_labels,
            label_text_mapping=label_text_mapping,
        )
        for text, candidate_labels in zip(texts, top_k_predicted_labels, strict=True)
    ]
    return np.vstack(reranked_rows)
