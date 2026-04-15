"""Encoder components for embedding-based model families."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin


class SentenceTransformerEncoder(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper around a sentence-transformers encoder."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self._model = None

    def fit(self, texts: list[str], labels: list[int] | None = None) -> SentenceTransformerEncoder:
        self._get_model()
        return self

    def transform(self, texts: list[str]):
        model = self._get_model()
        return model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "device": self.device,
            "_model": None,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.model_name = state["model_name"]
        self.batch_size = state["batch_size"]
        self.normalize_embeddings = state["normalize_embeddings"]
        self.device = state["device"]
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise ImportError(
                    "sentence-transformers is required for sentence embedding experiments. "
                    "Install it with `pip install sentence-transformers` or "
                    "`pip install -e \".[dev]\"` after updating dependencies."
                ) from error

            if self.device is None:
                self._model = SentenceTransformer(self.model_name)
            else:
                self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model
