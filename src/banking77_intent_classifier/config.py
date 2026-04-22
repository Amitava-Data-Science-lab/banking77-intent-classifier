"""Configuration models and loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TfidfConfig:
    """Hyperparameters for TF-IDF feature generation."""

    lowercase: bool = True
    strip_accents: str | None = "unicode"
    normalization: str = "none"
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 2
    max_df: int | float = 0.98
    sublinear_tf: bool = True
    max_features: int | None = 50_000


@dataclass(slots=True)
class EncoderConfig:
    """Configuration for embedding-based encoder models."""

    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: str | None = None


@dataclass(slots=True)
class RerankerConfig:
    """Configuration for optional cross-encoder reranking."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    enabled: bool = False


@dataclass(slots=True)
class ClassifierConfig:
    """Hyperparameters for the linear SVM classifier."""

    c: float = 1.0
    class_weight: str | None = "balanced"
    max_iter: int = 5_000
    dual: bool = True
    random_state: int = 42
    knn_neighbors: int = 5
    knn_weights: str = "distance"
    knn_metric: str = "cosine"
    probability_calibration_method: str = "sigmoid"
    probability_calibration_cv: int = 5


@dataclass(slots=True)
class AnalysisConfig:
    """Controls how many diagnostics are exported."""

    top_k_confusions: int = 25
    top_k_features_per_class: int = 20


@dataclass(slots=True)
class TransformerConfig:
    """Configuration for true transformer fine-tuning experiments."""

    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    max_length: int = 128
    learning_rate: float = 2e-5
    num_train_epochs: float = 3.0
    train_batch_size: int = 16
    eval_batch_size: int = 32
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float | None = None
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "macro_f1"
    greater_is_better: bool = True
    fp16: bool = False
    bf16: bool = False
    threshold_candidates: list[float] = field(default_factory=list)
    threshold_selection_metric: str = "macro_f1"


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration for a full training run."""

    model_family: str
    dataset_type: str
    dataset_name: str
    dataset_source: Path | None
    train_split: str
    validation_split: str | None
    test_split: str
    include_oos: bool
    oos_confidence_threshold: float | None
    oos_margin_threshold: float | None
    artifacts_dir: Path
    reports_dir: Path
    text_column: str
    label_column: str
    random_seed: int
    tfidf: TfidfConfig
    encoder: EncoderConfig
    transformer: TransformerConfig
    reranker: RerankerConfig
    classifier: ClassifierConfig
    analysis: AnalysisConfig


@dataclass(slots=True)
class SearchConfig:
    """Configuration for sklearn hyperparameter search."""

    param_distributions: dict[str, list]
    search_type: str = "randomized"
    n_iter: int = 20
    cv: int = 5
    scoring: str = "f1_macro"
    n_jobs: int | None = None
    verbose: int = 0
    refit: bool = True
    random_state: int = 42


@dataclass(slots=True)
class TuningConfig:
    """Configuration for a tuning run built on top of the training config."""

    experiment: ExperimentConfig
    search: SearchConfig


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from JSON."""

    config_path = Path(path)
    raw = _read_json(config_path)
    base_dir = config_path.parent

    return _parse_experiment_config(raw=raw, base_dir=base_dir)


def load_tuning_config(path: str | Path) -> TuningConfig:
    """Load a tuning configuration from JSON."""

    config_path = Path(path)
    raw = _read_json(config_path)
    base_dir = config_path.parent

    return TuningConfig(
        experiment=_parse_experiment_config(raw=raw, base_dir=base_dir),
        search=SearchConfig(
            param_distributions=_normalize_param_grid(raw["search"]["param_distributions"]),
            search_type=raw["search"].get("search_type", "randomized"),
            n_iter=raw["search"].get("n_iter", 20),
            cv=raw["search"].get("cv", 5),
            scoring=raw["search"].get("scoring", "f1_macro"),
            n_jobs=raw["search"].get("n_jobs"),
            verbose=raw["search"].get("verbose", 0),
            refit=raw["search"].get("refit", True),
            random_state=raw["search"].get("random_state", 42),
        ),
    )


def _parse_experiment_config(raw: dict, base_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        model_family=raw.get("model_family", "tfidf_svc"),
        dataset_type=raw.get("dataset_type", "banking77"),
        dataset_name=raw["dataset_name"],
        dataset_source=((base_dir / raw["dataset_source"]).resolve() if raw.get("dataset_source") else None),
        train_split=raw.get("train_split", "train"),
        validation_split=raw.get("validation_split"),
        test_split=raw.get("test_split", "test"),
        include_oos=raw.get("include_oos", True),
        oos_confidence_threshold=raw.get("oos_confidence_threshold"),
        oos_margin_threshold=raw.get("oos_margin_threshold"),
        artifacts_dir=(base_dir / raw["artifacts_dir"]).resolve(),
        reports_dir=(base_dir / raw["reports_dir"]).resolve(),
        text_column=raw.get("text_column", "text"),
        label_column=raw.get("label_column", "label"),
        random_seed=raw.get("random_seed", 42),
        tfidf=TfidfConfig(
            lowercase=raw["tfidf"].get("lowercase", True),
            strip_accents=raw["tfidf"].get("strip_accents", "unicode"),
            normalization=raw["tfidf"].get("normalization", "none"),
            ngram_range=tuple(raw["tfidf"].get("ngram_range", [1, 2])),
            min_df=raw["tfidf"].get("min_df", 2),
            max_df=raw["tfidf"].get("max_df", 0.98),
            sublinear_tf=raw["tfidf"].get("sublinear_tf", True),
            max_features=raw["tfidf"].get("max_features", 50_000),
        ),
        encoder=EncoderConfig(
            model_name=raw.get("encoder", {}).get("model_name", "all-MiniLM-L6-v2"),
            batch_size=raw.get("encoder", {}).get("batch_size", 32),
            normalize_embeddings=raw.get("encoder", {}).get("normalize_embeddings", True),
            device=raw.get("encoder", {}).get("device"),
        ),
        transformer=TransformerConfig(
            model_name=raw.get("transformer", {}).get("model_name", "sentence-transformers/all-mpnet-base-v2"),
            max_length=raw.get("transformer", {}).get("max_length", 128),
            learning_rate=raw.get("transformer", {}).get("learning_rate", 2e-5),
            num_train_epochs=raw.get("transformer", {}).get("num_train_epochs", 3.0),
            train_batch_size=raw.get("transformer", {}).get("train_batch_size", 16),
            eval_batch_size=raw.get("transformer", {}).get("eval_batch_size", 32),
            weight_decay=raw.get("transformer", {}).get("weight_decay", 0.01),
            warmup_steps=raw.get("transformer", {}).get("warmup_steps", 0),
            warmup_ratio=raw.get("transformer", {}).get("warmup_ratio"),
            save_strategy=raw.get("transformer", {}).get("save_strategy", "epoch"),
            evaluation_strategy=raw.get("transformer", {}).get("evaluation_strategy", "epoch"),
            load_best_model_at_end=raw.get("transformer", {}).get("load_best_model_at_end", True),
            metric_for_best_model=raw.get("transformer", {}).get("metric_for_best_model", "macro_f1"),
            greater_is_better=raw.get("transformer", {}).get("greater_is_better", True),
            fp16=raw.get("transformer", {}).get("fp16", False),
            bf16=raw.get("transformer", {}).get("bf16", False),
            threshold_candidates=raw.get("transformer", {}).get("threshold_candidates", []),
            threshold_selection_metric=raw.get("transformer", {}).get("threshold_selection_metric", "macro_f1"),
        ),
        reranker=RerankerConfig(
            model_name=raw.get("reranker", {}).get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            top_k=raw.get("reranker", {}).get("top_k", 5),
            enabled=raw.get("reranker", {}).get("enabled", False),
        ),
        classifier=ClassifierConfig(
            c=raw["classifier"].get("c", 1.0),
            class_weight=raw["classifier"].get("class_weight", "balanced"),
            max_iter=raw["classifier"].get("max_iter", 5_000),
            dual=raw["classifier"].get("dual", True),
            random_state=raw["classifier"].get("random_state", 42),
            knn_neighbors=raw["classifier"].get("knn_neighbors", 5),
            knn_weights=raw["classifier"].get("knn_weights", "distance"),
            knn_metric=raw["classifier"].get("knn_metric", "cosine"),
            probability_calibration_method=raw["classifier"].get("probability_calibration_method", "sigmoid"),
            probability_calibration_cv=raw["classifier"].get("probability_calibration_cv", 5),
        ),
        analysis=AnalysisConfig(
            top_k_confusions=raw["analysis"].get("top_k_confusions", 25),
            top_k_features_per_class=raw["analysis"].get("top_k_features_per_class", 20),
        ),
    )


def _normalize_param_grid(param_grid: dict[str, list]) -> dict[str, list]:
    normalized: dict[str, list] = {}
    for key, values in param_grid.items():
        if key.endswith("ngram_range"):
            normalized[key] = [tuple(value) for value in values]
        else:
            normalized[key] = values
    return normalized
