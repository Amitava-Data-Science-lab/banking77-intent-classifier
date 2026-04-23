"""Contrastive training and retrieval utilities for intent classification."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from banking77_intent_classifier.config import ContrastiveConfig, EncoderConfig, ExperimentConfig
from banking77_intent_classifier.data import DatasetBundle
from banking77_intent_classifier.evaluation import evaluate_predictions


@dataclass(slots=True)
class ContrastiveTriplet:
    """One contrastive training triplet."""

    anchor_text: str
    positive_text: str
    negative_text: str
    anchor_label_id: int
    positive_label_id: int
    negative_label_id: int
    negative_source: str


@dataclass(slots=True)
class ContrastiveTrainingData:
    """Triplet examples and metadata for training."""

    triplets: list[ContrastiveTriplet]
    metadata: dict[str, Any]


@dataclass(slots=True)
class ContrastiveRetrievalArtifacts:
    """Structured outputs from contrastive retrieval."""

    predictions: np.ndarray
    top_k_predicted_labels: np.ndarray
    best_scores: np.ndarray
    nearest_neighbor_similarities: np.ndarray


@dataclass(slots=True)
class ContrastiveTrainingArtifacts:
    """Saved outputs from contrastive training."""

    encoder: Any
    exemplar_embeddings: np.ndarray
    exemplar_label_ids: np.ndarray
    selected_oos_threshold: float | None
    validation_metrics_by_threshold: list[dict[str, Any]] = field(default_factory=list)
    threshold_selection_metadata: dict[str, Any] = field(default_factory=dict)
    training_data_metadata: dict[str, Any] = field(default_factory=dict)


def build_triplet_training_data(
    dataset: DatasetBundle,
    contrastive_config: ContrastiveConfig,
    random_seed: int,
    bootstrap_embeddings: np.ndarray,
) -> ContrastiveTrainingData:
    """Build balanced triplets for contrastive training."""

    rng = np.random.default_rng(random_seed)
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, label_id in enumerate(dataset.train_labels):
        label_to_indices[int(label_id)].append(index)

    if "oos" not in dataset.label_names:
        raise ValueError("Contrastive training requires the dataset to include an 'oos' label.")

    oos_label_id = dataset.label_names.index("oos")
    in_scope_label_ids = [label_id for label_id in sorted(label_to_indices) if label_id != oos_label_id]
    if not in_scope_label_ids:
        raise ValueError("Contrastive training requires at least one in-scope label.")

    for label_id in in_scope_label_ids:
        if len(label_to_indices[label_id]) < 2:
            raise ValueError(
                "Each in-scope class needs at least two examples to create anchor/positive pairs."
            )

    hard_label_candidates = _build_hard_negative_label_candidates(
        bootstrap_embeddings=bootstrap_embeddings,
        train_labels=np.asarray(dataset.train_labels, dtype=np.int64),
        in_scope_label_ids=in_scope_label_ids,
        contrastive_config=contrastive_config,
    )
    negative_plan = _build_negative_source_plan(contrastive_config)
    oos_sampler = _OOSSampler(
        indices=label_to_indices[oos_label_id],
        rng=rng,
        reuse_limit=contrastive_config.oos_batch_reuse_limit,
    )
    negative_source_counts: Counter[str] = Counter()
    anchor_counts: Counter[int] = Counter()
    triplets: list[ContrastiveTriplet] = []

    for label_id in in_scope_label_ids:
        anchor_indices = label_to_indices[label_id]
        for anchor_index in anchor_indices:
            anchor_counts[label_id] += 1
            positive_candidates = [index for index in anchor_indices if index != anchor_index]
            positive_cycle = _cycled_sample(
                values=positive_candidates,
                sample_size=contrastive_config.triplets_per_anchor,
                rng=rng,
            )
            hard_negative_cycle = _cycled_label_sample(
                values=hard_label_candidates[label_id],
                sample_size=negative_plan.count("hard_in_scope"),
                rng=rng,
            )
            random_negative_labels = _cycled_label_sample(
                values=[candidate for candidate in in_scope_label_ids if candidate != label_id],
                sample_size=negative_plan.count("random_in_scope"),
                rng=rng,
            )
            hard_negative_position = 0
            random_negative_position = 0

            for triplet_index, negative_source in enumerate(negative_plan):
                if negative_source == "hard_in_scope":
                    negative_label_id = hard_negative_cycle[hard_negative_position]
                    hard_negative_position += 1
                    negative_index = _choose_negative_example(
                        candidate_indices=label_to_indices[negative_label_id],
                        rng=rng,
                    )
                elif negative_source == "random_in_scope":
                    negative_label_id = random_negative_labels[random_negative_position]
                    random_negative_position += 1
                    negative_index = _choose_negative_example(
                        candidate_indices=label_to_indices[negative_label_id],
                        rng=rng,
                    )
                else:
                    negative_label_id = oos_label_id
                    negative_index = oos_sampler.next_index()

                negative_source_counts[negative_source] += 1
                triplets.append(
                    ContrastiveTriplet(
                        anchor_text=dataset.train_texts[anchor_index],
                        positive_text=dataset.train_texts[positive_cycle[triplet_index]],
                        negative_text=dataset.train_texts[negative_index],
                        anchor_label_id=label_id,
                        positive_label_id=label_id,
                        negative_label_id=negative_label_id,
                        negative_source=negative_source,
                    )
                )

    return ContrastiveTrainingData(
        triplets=triplets,
        metadata={
            "triplet_count": len(triplets),
            "triplets_per_anchor": contrastive_config.triplets_per_anchor,
            "negative_source_counts": dict(negative_source_counts),
            "negative_source_ratios": {
                source: float(count / len(triplets)) for source, count in negative_source_counts.items()
            },
            "anchor_counts_by_label": {dataset.label_names[label_id]: anchor_counts[label_id] for label_id in in_scope_label_ids},
            "oos_negative_usage": oos_sampler.usage_summary(),
            "hard_negative_neighbors": {
                dataset.label_names[label_id]: [dataset.label_names[candidate] for candidate in hard_label_candidates[label_id]]
                for label_id in in_scope_label_ids
            },
            "policy": {
                "oos_anchor_allowed": False,
                "oos_positive_allowed": False,
                "negative_mix": {
                    "hard_in_scope": contrastive_config.hard_negative_ratio,
                    "random_in_scope": contrastive_config.random_negative_ratio,
                    "oos": contrastive_config.oos_negative_ratio,
                },
            },
        },
    )


def train_contrastive_model(
    dataset: DatasetBundle,
    config: ExperimentConfig,
) -> ContrastiveTrainingArtifacts:
    """Fine-tune MPNet contrastively and build retrieval artifacts."""

    if not dataset.validation_texts:
        raise ValueError("Contrastive training requires a validation split for OOS threshold tuning.")

    model = _load_sentence_transformer(
        model_name=config.contrastive.model_name,
        encoder_config=config.encoder,
    )
    train_embeddings = model.encode(
        dataset.train_texts,
        batch_size=config.encoder.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    training_data = build_triplet_training_data(
        dataset=dataset,
        contrastive_config=config.contrastive,
        random_seed=config.random_seed,
        bootstrap_embeddings=train_embeddings,
    )
    _fit_sentence_transformer(
        model=model,
        triplets=training_data.triplets,
        contrastive_config=config.contrastive,
    )

    in_scope_mask = np.asarray(dataset.train_labels, dtype=np.int64) != dataset.label_names.index("oos")
    exemplar_embeddings = model.encode(
        [text for text, keep in zip(dataset.train_texts, in_scope_mask, strict=True) if keep],
        batch_size=config.encoder.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    exemplar_label_ids = np.asarray(
        [label for label in dataset.train_labels if label != dataset.label_names.index("oos")],
        dtype=np.int64,
    )

    validation_retrieval = retrieve_contrastive_predictions(
        encoder=model,
        texts=dataset.validation_texts,
        exemplar_embeddings=exemplar_embeddings,
        exemplar_label_ids=exemplar_label_ids,
        label_names=dataset.label_names,
        contrastive_config=config.contrastive,
        oos_threshold=None,
    )
    validation_metrics_by_threshold, threshold_selection_metadata = evaluate_contrastive_threshold_candidates(
        best_scores=validation_retrieval.best_scores,
        per_example_ranked_labels=validation_retrieval.top_k_predicted_labels,
        y_true=dataset.validation_labels,
        label_names=dataset.label_names,
        threshold_candidates=config.contrastive.threshold_candidates,
        analysis_top_k_confusions=config.analysis.top_k_confusions,
        analysis_top_k_features_per_class=config.analysis.top_k_features_per_class,
        selection_metric=config.contrastive.threshold_selection_metric,
        selection_strategy=config.contrastive.threshold_selection_strategy,
        max_in_scope_false_oos_rate=config.contrastive.threshold_max_in_scope_false_oos_rate,
        macro_f1_tolerance_ladder=config.contrastive.threshold_macro_f1_tolerance_ladder,
        fallback_strategy=config.contrastive.threshold_fallback_strategy,
    )
    return ContrastiveTrainingArtifacts(
        encoder=model,
        exemplar_embeddings=exemplar_embeddings,
        exemplar_label_ids=exemplar_label_ids,
        selected_oos_threshold=threshold_selection_metadata.get("selected_threshold"),
        validation_metrics_by_threshold=validation_metrics_by_threshold,
        threshold_selection_metadata=threshold_selection_metadata,
        training_data_metadata=training_data.metadata,
    )


def retrieve_contrastive_predictions(
    encoder: Any,
    texts: list[str],
    exemplar_embeddings: np.ndarray,
    exemplar_label_ids: np.ndarray,
    label_names: list[str],
    contrastive_config: ContrastiveConfig,
    oos_threshold: float | None,
) -> ContrastiveRetrievalArtifacts:
    """Retrieve nearest exemplars and convert them to label predictions."""

    if exemplar_embeddings.size == 0:
        raise ValueError("Exemplar embeddings cannot be empty.")

    query_embeddings = encoder.encode(
        texts,
        batch_size=max(1, contrastive_config.batch_size),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    similarities = np.matmul(query_embeddings, exemplar_embeddings.T)
    neighbor_count = min(contrastive_config.neighbor_count, exemplar_embeddings.shape[0])
    if neighbor_count <= 0:
        raise ValueError("neighbor_count must be at least 1.")

    top_neighbor_indices = np.argsort(similarities, axis=1)[:, -neighbor_count:][:, ::-1]
    oos_label_id = label_names.index("oos") if "oos" in label_names else None
    predictions: list[int] = []
    top_k_predictions: list[list[int]] = []
    best_scores: list[float] = []
    nearest_neighbor_similarities: list[float] = []

    in_scope_label_ids = [label_id for label_id, label_name in enumerate(label_names) if label_name != "oos"]

    for row_index, neighbor_indices in enumerate(top_neighbor_indices):
        per_label_scores = np.full(len(label_names), -np.inf, dtype=np.float64)
        top_similarity = float(similarities[row_index, neighbor_indices[0]])
        nearest_neighbor_similarities.append(top_similarity)
        for neighbor_index in neighbor_indices:
            label_id = int(exemplar_label_ids[neighbor_index])
            similarity = float(similarities[row_index, neighbor_index])
            if contrastive_config.vote_strategy == "sum_similarity":
                existing = 0.0 if not np.isfinite(per_label_scores[label_id]) else per_label_scores[label_id]
                per_label_scores[label_id] = existing + similarity
            elif contrastive_config.vote_strategy == "max_similarity":
                existing = -np.inf if not np.isfinite(per_label_scores[label_id]) else per_label_scores[label_id]
                per_label_scores[label_id] = max(existing, similarity)
            else:
                raise ValueError(f"Unsupported contrastive vote_strategy: {contrastive_config.vote_strategy}")

        ranked_in_scope = sorted(
            in_scope_label_ids,
            key=lambda label_id: (-float(per_label_scores[label_id]), label_id),
        )
        best_label_id = ranked_in_scope[0]
        best_score = float(per_label_scores[best_label_id])
        best_scores.append(best_score)
        predicted_label_id = best_label_id
        if oos_threshold is not None and oos_label_id is not None and best_score < oos_threshold:
            predicted_label_id = oos_label_id
            per_label_scores[oos_label_id] = np.inf
        elif oos_label_id is not None:
            per_label_scores[oos_label_id] = -np.inf

        full_ranking = sorted(
            range(len(label_names)),
            key=lambda label_id: (-float(per_label_scores[label_id]), label_id),
        )
        predictions.append(predicted_label_id)
        top_k_predictions.append(full_ranking[: min(5, len(full_ranking))])

    return ContrastiveRetrievalArtifacts(
        predictions=np.asarray(predictions, dtype=np.int64),
        top_k_predicted_labels=np.asarray(top_k_predictions, dtype=np.int64),
        best_scores=np.asarray(best_scores, dtype=np.float64),
        nearest_neighbor_similarities=np.asarray(nearest_neighbor_similarities, dtype=np.float64),
    )


def evaluate_contrastive_threshold_candidates(
    best_scores: np.ndarray,
    per_example_ranked_labels: np.ndarray,
    y_true: list[int],
    label_names: list[str],
    threshold_candidates: list[float],
    analysis_top_k_confusions: int,
    analysis_top_k_features_per_class: int,
    selection_metric: str,
    selection_strategy: str,
    max_in_scope_false_oos_rate: float,
    macro_f1_tolerance_ladder: list[float],
    fallback_strategy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate OOS thresholds for retrieval scores."""

    if "oos" not in label_names:
        return [], {"strategy": "disabled", "selected_threshold": None}

    oos_label_id = label_names.index("oos")
    base_predictions = per_example_ranked_labels[:, 0].astype(np.int64)
    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates:
        predictions = base_predictions.copy()
        predictions[best_scores < threshold] = oos_label_id
        top_k_predictions = per_example_ranked_labels.copy()
        top_k_predictions[best_scores < threshold, 0] = oos_label_id
        evaluation = evaluate_predictions(
            y_true=y_true,
            y_pred=predictions.tolist(),
            top_k_predicted_labels=top_k_predictions,
            label_names=label_names,
            coefficients=None,
            feature_names=np.array([], dtype=object),
            top_k_confusions=analysis_top_k_confusions,
            top_k_features_per_class=analysis_top_k_features_per_class,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": evaluation.accuracy,
                "macro_f1": evaluation.macro_f1,
                "oos_f1": evaluation.oos_metrics.get("f1", 0.0),
                "in_scope_false_oos_rate": evaluation.oos_metrics.get("in_scope_false_oos_rate", 0.0),
                "oos_metrics": evaluation.oos_metrics,
            }
        )

    if not rows:
        return rows, {"strategy": "manual_override", "selected_threshold": None}

    if selection_strategy == "oos_aware_constrained":
        return _select_oos_aware_threshold_candidates(
            rows=rows,
            selection_metric=selection_metric,
            max_in_scope_false_oos_rate=max_in_scope_false_oos_rate,
            macro_f1_tolerance_ladder=macro_f1_tolerance_ladder,
            fallback_strategy=fallback_strategy,
        )

    ranked_rows = sorted(
        rows,
        key=lambda row: _threshold_sort_key(row=row, selection_metric=selection_metric),
        reverse=True,
    )
    selected_threshold = ranked_rows[0]["threshold"]
    return ranked_rows, {
        "strategy": selection_strategy,
        "selection_metric": selection_metric,
        "selected_threshold": selected_threshold,
    }


def _build_negative_source_plan(contrastive_config: ContrastiveConfig) -> list[str]:
    triplets_per_anchor = contrastive_config.triplets_per_anchor
    source_counts = {
        "hard_in_scope": int(round(triplets_per_anchor * contrastive_config.hard_negative_ratio)),
        "random_in_scope": int(round(triplets_per_anchor * contrastive_config.random_negative_ratio)),
        "oos": int(round(triplets_per_anchor * contrastive_config.oos_negative_ratio)),
    }
    allocated = sum(source_counts.values())
    if allocated != triplets_per_anchor:
        source_counts["hard_in_scope"] += triplets_per_anchor - allocated

    plan = (
        ["hard_in_scope"] * source_counts["hard_in_scope"]
        + ["random_in_scope"] * source_counts["random_in_scope"]
        + ["oos"] * source_counts["oos"]
    )
    return plan


def _build_hard_negative_label_candidates(
    bootstrap_embeddings: np.ndarray,
    train_labels: np.ndarray,
    in_scope_label_ids: list[int],
    contrastive_config: ContrastiveConfig,
) -> dict[int, list[int]]:
    label_centroids: dict[int, np.ndarray] = {}
    for label_id in in_scope_label_ids:
        label_centroids[label_id] = bootstrap_embeddings[train_labels == label_id].mean(axis=0)
    centroid_matrix = np.vstack([label_centroids[label_id] for label_id in in_scope_label_ids])
    centroid_matrix = _normalize_embeddings(centroid_matrix)
    similarity_matrix = np.matmul(centroid_matrix, centroid_matrix.T)

    hard_candidates: dict[int, list[int]] = {}
    for row_index, label_id in enumerate(in_scope_label_ids):
        ranked_other_indices = [
            candidate_index
            for candidate_index in np.argsort(similarity_matrix[row_index])[::-1].tolist()
            if candidate_index != row_index
        ]
        hard_candidates[label_id] = [
            in_scope_label_ids[candidate_index]
            for candidate_index in ranked_other_indices[: contrastive_config.hard_negative_top_k_labels]
        ]
    return hard_candidates


def _choose_negative_example(candidate_indices: list[int], rng: np.random.Generator) -> int:
    return int(rng.choice(candidate_indices))


def _cycled_sample(values: list[int], sample_size: int, rng: np.random.Generator) -> list[int]:
    if not values:
        raise ValueError("Cannot sample from an empty candidate set.")
    output: list[int] = []
    while len(output) < sample_size:
        cycle = list(values)
        rng.shuffle(cycle)
        output.extend(cycle)
    return output[:sample_size]


def _cycled_label_sample(values: list[int], sample_size: int, rng: np.random.Generator) -> list[int]:
    if sample_size == 0:
        return []
    if not values:
        raise ValueError("Cannot sample labels from an empty candidate set.")
    return _cycled_sample(values=values, sample_size=sample_size, rng=rng)


class _OOSSampler:
    """Least-used-first sampler for limited OOS pools."""

    def __init__(self, indices: list[int], rng: np.random.Generator, reuse_limit: int) -> None:
        if not indices:
            raise ValueError("At least one OOS training example is required for OOS negatives.")
        self._indices = list(indices)
        self._rng = rng
        self._reuse_limit = reuse_limit
        self._usage_counts: Counter[int] = Counter()

    def next_index(self) -> int:
        minimum_count = min((self._usage_counts[index] for index in self._indices), default=0)
        candidates = [index for index in self._indices if self._usage_counts[index] == minimum_count]
        if self._reuse_limit > 0:
            candidates = [
                index
                for index in self._indices
                if self._usage_counts[index] <= minimum_count + self._reuse_limit
            ] or candidates
        chosen = int(self._rng.choice(candidates))
        self._usage_counts[chosen] += 1
        return chosen

    def usage_summary(self) -> dict[str, Any]:
        counts = [self._usage_counts[index] for index in self._indices]
        return {
            "pool_size": len(self._indices),
            "min_reuse": int(min(counts, default=0)),
            "max_reuse": int(max(counts, default=0)),
            "mean_reuse": float(np.mean(counts)) if counts else 0.0,
        }


def _fit_sentence_transformer(
    model: Any,
    triplets: list[ContrastiveTriplet],
    contrastive_config: ContrastiveConfig,
) -> None:
    try:
        from sentence_transformers import InputExample, losses
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is required for contrastive training. "
            "Install it with `pip install sentence-transformers`."
        ) from error

    try:
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise ImportError("torch is required for contrastive training.") from error

    training_examples = [
        InputExample(texts=[triplet.anchor_text, triplet.positive_text, triplet.negative_text])
        for triplet in triplets
    ]
    train_dataloader = DataLoader(
        training_examples,
        shuffle=True,
        batch_size=contrastive_config.batch_size,
    )
    train_loss = losses.TripletLoss(
        model=model,
        triplet_margin=contrastive_config.triplet_margin,
    )
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=contrastive_config.epochs,
        warmup_steps=contrastive_config.warmup_steps,
        optimizer_params={"lr": contrastive_config.learning_rate},
        show_progress_bar=False,
    )


def _load_sentence_transformer(model_name: str, encoder_config: EncoderConfig):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is required for contrastive experiments. "
            "Install it with `pip install sentence-transformers`."
        ) from error

    if encoder_config.device is None:
        return SentenceTransformer(model_name)
    return SentenceTransformer(model_name, device=encoder_config.device)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def _select_oos_aware_threshold_candidates(
    rows: list[dict[str, Any]],
    selection_metric: str,
    max_in_scope_false_oos_rate: float,
    macro_f1_tolerance_ladder: list[float],
    fallback_strategy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    best_macro_f1 = max((row["macro_f1"] for row in rows), default=0.0)
    selected_row: dict[str, Any] | None = None
    successful_tolerance: float | None = None

    for tolerance in macro_f1_tolerance_ladder:
        eligible_rows = [
            row
            for row in rows
            if row["in_scope_false_oos_rate"] <= max_in_scope_false_oos_rate
            and row["macro_f1"] >= best_macro_f1 - tolerance
        ]
        if eligible_rows:
            selected_row = max(eligible_rows, key=_oos_constrained_sort_key)
            successful_tolerance = tolerance
            break

    fallback_used = False
    if selected_row is None:
        fallback_used = True
        if fallback_strategy != "best_macro_f1":
            raise ValueError(f"Unsupported threshold fallback strategy: {fallback_strategy}")
        selected_row = max(rows, key=lambda row: _threshold_sort_key(row=row, selection_metric=selection_metric))

    annotated_rows: list[dict[str, Any]] = []
    comparison_tolerance = (
        successful_tolerance
        if successful_tolerance is not None
        else max(macro_f1_tolerance_ladder, default=0.05)
    )
    for row in rows:
        annotated_row = dict(row)
        reasons: list[str] = []
        if row["in_scope_false_oos_rate"] > max_in_scope_false_oos_rate:
            reasons.append("in_scope_false_oos_rate_exceeds_limit")
        if row["macro_f1"] < best_macro_f1 - comparison_tolerance:
            reasons.append("macro_f1_below_tolerance")
        if row["threshold"] == selected_row["threshold"]:
            annotated_row["selection_eligible"] = not fallback_used or not reasons
            annotated_row["eligibility_reason"] = (
                "selected_by_oos_aware_constraints"
                if not fallback_used
                else "fallback_best_macro_f1"
            )
        else:
            annotated_row["selection_eligible"] = len(reasons) == 0 and not fallback_used
            annotated_row["eligibility_reason"] = (
                "eligible_not_selected"
                if annotated_row["selection_eligible"]
                else ",".join(reasons) if reasons else "ranked_below_selected_threshold"
            )
        annotated_rows.append(annotated_row)

    selected_threshold = selected_row["threshold"]
    annotated_rows = sorted(
        annotated_rows,
        key=lambda row: (
            row["threshold"] != selected_threshold,
            -row["oos_f1"],
            -row["macro_f1"],
            row["in_scope_false_oos_rate"],
            row["threshold"],
        ),
    )

    return annotated_rows, {
        "strategy": "oos_aware_constrained",
        "selection_metric": selection_metric,
        "selected_threshold": selected_threshold,
        "best_macro_f1": best_macro_f1,
        "max_in_scope_false_oos_rate": max_in_scope_false_oos_rate,
        "macro_f1_tolerance_ladder": macro_f1_tolerance_ladder,
        "successful_tolerance": successful_tolerance,
        "fallback_strategy": fallback_strategy,
        "fallback_used": fallback_used,
    }


def _oos_constrained_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        row["oos_f1"],
        row["macro_f1"],
        -row["in_scope_false_oos_rate"],
        -row["threshold"],
    )


def _threshold_sort_key(row: dict[str, Any], selection_metric: str) -> tuple[float, float, float]:
    if selection_metric == "oos_recall":
        return (
            row["oos_metrics"].get("recall", 0.0),
            row["macro_f1"],
            row["accuracy"],
        )
    if selection_metric == "oos_f1":
        return (
            row["oos_metrics"].get("f1", 0.0),
            row["macro_f1"],
            row["accuracy"],
        )
    return (
        row["macro_f1"],
        row["oos_metrics"].get("recall", 0.0),
        row["accuracy"],
    )
