"""Text preprocessing helpers for optional normalization experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SpacyLemmatizerTokenizer:
    """spaCy-backed tokenizer that emits lemmatized tokens for sklearn."""

    language: str = "en"
    _nlp: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._nlp = self._build_nlp()

    def __call__(self, text: str) -> list[str]:
        document = self._nlp(text)
        return [
            self._normalize_lemma(token)
            for token in document
            if not token.is_space and not token.is_punct
        ]

    def __getstate__(self) -> dict[str, str]:
        return {"language": self.language}

    def __setstate__(self, state: dict[str, str]) -> None:
        self.language = state["language"]
        self._nlp = self._build_nlp()

    def _build_nlp(self):
        try:
            from spacy.lang.en import English
        except ImportError as error:
            raise ImportError(
                "spaCy is required for lemmatized experiments. Install it with "
                "`pip install spacy spacy-lookups-data` or `pip install -e \".[dev]\"` "
                "after adding the dependency."
            ) from error

        if self.language != "en":
            raise ValueError(f"Unsupported lemmatizer language: {self.language}")

        nlp = English()
        nlp.add_pipe("lemmatizer", config={"mode": "rule"})
        nlp.initialize()
        return nlp

    @staticmethod
    def _normalize_lemma(token) -> str:
        lemma = token.lemma_.strip()
        if not lemma:
            return token.lower_
        if lemma == "-PRON-":
            return token.lower_
        return lemma.lower()
