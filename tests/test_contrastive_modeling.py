from pathlib import Path

import numpy as np

from banking77_intent_classifier.config import ContrastiveConfig
from banking77_intent_classifier.contrastive_modeling import (
    ContrastiveRetrievalArtifacts,
    ContrastiveTrainingArtifacts,
    build_triplet_training_data,
    retrieve_contrastive_predictions,
)
from banking77_intent_classifier.data import DatasetBundle
from banking77_intent_classifier.inference import ContrastivePredictor
from banking77_intent_classifier.pipeline import run_training_pipeline


class DummyEncoder:
    def __init__(self, embeddings_by_text: dict[str, np.ndarray]) -> None:
        self._embeddings_by_text = embeddings_by_text

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        rows = np.vstack([self._embeddings_by_text[text] for text in texts]).astype(np.float64)
        if normalize_embeddings:
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            rows = rows / norms
        return rows


def _build_dataset() -> DatasetBundle:
    label_names = ["intent_a", "intent_b", "intent_c", "oos"]
    train_texts = [
        "a0",
        "a1",
        "b0",
        "b1",
        "c0",
        "c1",
        "o0",
        "o1",
    ]
    train_labels = [0, 0, 1, 1, 2, 2, 3, 3]
    return DatasetBundle(
        train_texts=train_texts,
        train_labels=train_labels,
        validation_texts=["va", "voos"],
        validation_labels=[0, 3],
        test_texts=["ta", "toos"],
        test_labels=[0, 3],
        label_names=label_names,
        metadata={"dataset_type": "clinc150"},
    )


def test_build_triplet_training_data_uses_requested_negative_mix() -> None:
    dataset = _build_dataset()
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.95, 0.05],
            [0.94, 0.06],
            [0.0, 1.0],
            [0.01, 0.99],
            [-1.0, 0.0],
            [-0.99, 0.01],
        ]
    )
    config = ContrastiveConfig(
        triplets_per_anchor=10,
        hard_negative_ratio=0.5,
        random_negative_ratio=0.3,
        oos_negative_ratio=0.2,
        hard_negative_top_k_labels=1,
    )

    training_data = build_triplet_training_data(
        dataset=dataset,
        contrastive_config=config,
        random_seed=42,
        bootstrap_embeddings=embeddings,
    )

    counts = training_data.metadata["negative_source_counts"]
    assert counts["hard_in_scope"] == 30
    assert counts["random_in_scope"] == 18
    assert counts["oos"] == 12
    assert all(triplet.anchor_label_id != 3 for triplet in training_data.triplets)
    assert all(triplet.positive_label_id != 3 for triplet in training_data.triplets)
    assert training_data.metadata["anchor_counts_by_label"] == {
        "intent_a": 2,
        "intent_b": 2,
        "intent_c": 2,
    }


def test_build_triplet_training_data_resamples_oos_examples_when_pool_is_small() -> None:
    dataset = _build_dataset()
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.95, 0.05],
            [0.94, 0.06],
            [0.0, 1.0],
            [0.01, 0.99],
            [-1.0, 0.0],
            [-0.99, 0.01],
        ]
    )
    config = ContrastiveConfig(
        triplets_per_anchor=20,
        hard_negative_ratio=0.5,
        random_negative_ratio=0.3,
        oos_negative_ratio=0.2,
        hard_negative_top_k_labels=1,
        oos_batch_reuse_limit=1,
    )

    training_data = build_triplet_training_data(
        dataset=dataset,
        contrastive_config=config,
        random_seed=7,
        bootstrap_embeddings=embeddings,
    )

    usage = training_data.metadata["oos_negative_usage"]
    assert usage["pool_size"] == 2
    assert usage["max_reuse"] > 1


def test_contrastive_predictor_returns_oos_for_low_similarity_queries() -> None:
    encoder = DummyEncoder(
        {
            "known": np.array([1.0, 0.0]),
            "unknown": np.array([-1.0, 0.0]),
        }
    )
    predictor = ContrastivePredictor(
        encoder=encoder,
        label_mapping={0: "intent_a", 1: "oos"},
        exemplar_embeddings=np.asarray([[1.0, 0.0], [0.95, 0.05]]),
        exemplar_label_ids=np.asarray([0, 0]),
        contrastive_config=ContrastiveConfig(neighbor_count=2, threshold_candidates=[0.4]),
        oos_threshold=0.4,
    )

    known_prediction = predictor.predict_one("known")
    unknown_prediction = predictor.predict_one("unknown")

    assert known_prediction.label == "intent_a"
    assert unknown_prediction.label == "oos"


def test_run_training_pipeline_routes_contrastive_family(monkeypatch) -> None:
    dataset = _build_dataset()
    encoder = DummyEncoder({"ta": np.array([1.0, 0.0]), "toos": np.array([-1.0, 0.0])})

    monkeypatch.setattr(
        "banking77_intent_classifier.pipeline.load_dataset_bundle",
        lambda **kwargs: dataset,
    )
    monkeypatch.setattr(
        "banking77_intent_classifier.pipeline.train_contrastive_model",
        lambda dataset, config: ContrastiveTrainingArtifacts(
            encoder=encoder,
            exemplar_embeddings=np.asarray([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]]),
            exemplar_label_ids=np.asarray([0, 0, 2]),
            selected_oos_threshold=0.4,
            validation_metrics_by_threshold=[{"threshold": 0.4, "macro_f1": 0.9}],
            threshold_selection_metadata={"selected_threshold": 0.4},
            training_data_metadata={"triplet_count": 60},
        ),
    )
    monkeypatch.setattr(
        "banking77_intent_classifier.pipeline.retrieve_contrastive_predictions",
        lambda **kwargs: ContrastiveRetrievalArtifacts(
            predictions=np.asarray([0, 3]),
            top_k_predicted_labels=np.asarray([[0, 1, 2, 3], [3, 0, 1, 2]]),
            best_scores=np.asarray([0.9, 0.1]),
            nearest_neighbor_similarities=np.asarray([0.95, -0.2]),
        ),
    )
    monkeypatch.setattr(
        "banking77_intent_classifier.pipeline.save_contrastive_artifacts",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "banking77_intent_classifier.pipeline.save_evaluation_reports",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "banking77_intent_classifier.pipeline.save_confusion_matrix_figure",
        lambda **kwargs: None,
    )

    summary = run_training_pipeline(
        Path("configs/clinc150_sentence_transformer_contrastive_mpnet.json")
    )

    assert summary["model_family"] == "sentence_transformer_contrastive_knn"
    assert summary["oos_confidence_threshold"] == 0.4
    assert summary["training_data_metadata"]["triplet_count"] == 60


def test_retrieve_contrastive_predictions_ranks_in_scope_and_oos() -> None:
    label_names = ["intent_a", "intent_b", "oos"]
    encoder = DummyEncoder(
        {
            "known": np.array([1.0, 0.0]),
            "other": np.array([0.0, 1.0]),
            "unknown": np.array([-1.0, 0.0]),
        }
    )
    retrieval = retrieve_contrastive_predictions(
        encoder=encoder,
        texts=["known", "unknown"],
        exemplar_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        exemplar_label_ids=np.asarray([0, 1]),
        label_names=label_names,
        contrastive_config=ContrastiveConfig(neighbor_count=2, vote_strategy="max_similarity"),
        oos_threshold=0.2,
    )

    assert retrieval.predictions.tolist() == [0, 2]
    assert retrieval.top_k_predicted_labels[1, 0] == 2
