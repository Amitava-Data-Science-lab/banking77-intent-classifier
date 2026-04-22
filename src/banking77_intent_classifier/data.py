"""Dataset loading utilities for intent classification experiments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import load_dataset


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DatasetBundle:
    """Normalized in-memory representation of dataset splits."""

    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    label_names: list[str]
    validation_texts: list[str] = field(default_factory=list)
    validation_labels: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_dataset_bundle(
    dataset_type: str,
    dataset_name: str,
    dataset_task: str,
    dataset_source: str | None,
    train_split: str,
    validation_split: str | None,
    test_split: str,
    text_column: str,
    label_column: str,
    include_oos: bool,
) -> DatasetBundle:
    """Load a dataset bundle using the configured dataset type."""

    if dataset_type == "banking77":
        return load_banking77_dataset(
            dataset_name=dataset_name,
            train_split=train_split,
            validation_split=validation_split,
            test_split=test_split,
            text_column=text_column,
            label_column=label_column,
        )

    if dataset_type == "clinc150":
        if dataset_source is None:
            raise ValueError("CLINC150 requires a local dataset_source pointing to data_full.json.")

        return load_clinc150_dataset(
            dataset_source=dataset_source,
            train_split=train_split,
            validation_split=validation_split,
            test_split=test_split,
            include_oos=include_oos,
            dataset_task=dataset_task,
        )

    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def load_banking77_dataset(
    dataset_name: str,
    train_split: str,
    validation_split: str | None,
    test_split: str,
    text_column: str,
    label_column: str,
) -> DatasetBundle:
    """Load Banking77 from Hugging Face datasets and return normalized Python structures."""

    try:
        dataset = _load_banking77_from_parquet(dataset_name=dataset_name)
    except RuntimeError as error:
        if "Dataset scripts are no longer supported" not in str(error):
            raise

        LOGGER.warning(
            "Falling back to the dataset's default loading path for %s after Parquet loading failed.",
            dataset_name,
        )
        dataset = load_dataset(dataset_name)

    train_dataset = dataset[train_split]
    test_dataset = dataset[test_split]
    validation_dataset = dataset[validation_split] if validation_split is not None else None

    label_feature = train_dataset.features[label_column]
    if not hasattr(label_feature, "names"):
        raise ValueError(
            "Expected the label column to be a ClassLabel feature with accessible label names."
        )

    return DatasetBundle(
        train_texts=list(train_dataset[text_column]),
        train_labels=list(train_dataset[label_column]),
        validation_texts=list(validation_dataset[text_column]) if validation_dataset is not None else [],
        validation_labels=list(validation_dataset[label_column]) if validation_dataset is not None else [],
        test_texts=list(test_dataset[text_column]),
        test_labels=list(test_dataset[label_column]),
        label_names=list(label_feature.names),
        metadata={"dataset_type": "banking77", "dataset_name": dataset_name},
    )


def load_clinc150_dataset(
    dataset_source: str | Path,
    train_split: str = "train",
    validation_split: str | None = "val",
    test_split: str = "test",
    include_oos: bool = True,
    dataset_task: str = "full_intent",
) -> DatasetBundle:
    """Load CLINC150 from the UCI JSON export and normalize into integer labels."""

    dataset_path = Path(dataset_source)
    if not dataset_path.exists():
        raise FileNotFoundError(
            "CLINC150 dataset file was not found at "
            f"'{dataset_path}'. "
            "Set `dataset_source` in your config to the correct `data_full.json` location, "
            "or place the file at `data/clinc150/data_full.json` relative to the repo root."
        )

    with dataset_path.open("r", encoding="utf-8") as file_handle:
        raw_dataset = json.load(file_handle)

    required_splits = [train_split, test_split]
    if validation_split is not None:
        required_splits.append(validation_split)

    for split_name in required_splits:
        if split_name not in raw_dataset:
            raise ValueError(f"Split '{split_name}' was not found in CLINC150 dataset {dataset_path}.")

    oos_splits = {
        split_name: _resolve_clinc150_oos_split_name(split_name)
        for split_name in required_splits
    }
    if include_oos:
        for split_name, oos_split_name in oos_splits.items():
            if oos_split_name not in raw_dataset:
                raise ValueError(
                    f"CLINC150 include_oos=True requires split '{oos_split_name}' "
                    f"for base split '{split_name}' in dataset {dataset_path}."
                )

    labels = sorted(
        {
            _normalize_clinc150_label(str(label), dataset_task=dataset_task)
            for split_name in required_splits
            for _, label in _get_clinc150_rows(
                raw_dataset=raw_dataset,
                split_name=split_name,
                include_oos=include_oos,
            )
        }
    )
    label_to_id = {label: index for index, label in enumerate(labels)}

    train_texts, train_labels = _extract_clinc150_split(
        raw_dataset=raw_dataset,
        split_name=train_split,
        label_to_id=label_to_id,
        include_oos=include_oos,
        dataset_task=dataset_task,
    )
    validation_texts, validation_labels = _extract_clinc150_split(
        raw_dataset=raw_dataset,
        split_name=validation_split,
        label_to_id=label_to_id,
        include_oos=include_oos,
        dataset_task=dataset_task,
    )
    test_texts, test_labels = _extract_clinc150_split(
        raw_dataset=raw_dataset,
        split_name=test_split,
        label_to_id=label_to_id,
        include_oos=include_oos,
        dataset_task=dataset_task,
    )

    metadata = {
        "dataset_type": "clinc150",
        "dataset_task": dataset_task,
        "dataset_source": str(dataset_path),
        "include_oos": include_oos,
        "oos_label": "oos" if include_oos and "oos" in label_to_id else None,
    }

    return DatasetBundle(
        train_texts=train_texts,
        train_labels=train_labels,
        validation_texts=validation_texts,
        validation_labels=validation_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        label_names=labels,
        metadata=metadata,
    )


def _extract_clinc150_split(
    raw_dataset: dict[str, list[list[str]]],
    split_name: str | None,
    label_to_id: dict[str, int],
    include_oos: bool,
    dataset_task: str,
) -> tuple[list[str], list[int]]:
    if split_name is None:
        return [], []

    texts: list[str] = []
    labels: list[int] = []
    for row in _get_clinc150_rows(
        raw_dataset=raw_dataset,
        split_name=split_name,
        include_oos=include_oos,
    ):
        if len(row) != 2:
            raise ValueError(
                f"Expected each CLINC150 row to contain [text, label], got {row!r} in split {split_name}."
            )

        text, label = row
        normalized_label = _normalize_clinc150_label(str(label), dataset_task=dataset_task)
        texts.append(str(text))
        labels.append(label_to_id[normalized_label])

    return texts, labels


def _get_clinc150_rows(
    raw_dataset: dict[str, list[list[str]]],
    split_name: str,
    include_oos: bool,
) -> list[list[str]]:
    rows = list(raw_dataset[split_name])
    if include_oos:
        rows.extend(raw_dataset[_resolve_clinc150_oos_split_name(split_name)])
    return rows


def _resolve_clinc150_oos_split_name(split_name: str) -> str:
    return f"oos_{split_name}"


def _normalize_clinc150_label(label: str, dataset_task: str) -> str:
    if dataset_task == "binary_oos":
        return "oos" if label == "oos" else "not_oos"
    return label


def _load_banking77_from_parquet(dataset_name: str):
    return load_dataset(
        "parquet",
        data_files={
            "train": f"hf://datasets/{dataset_name}@refs/convert/parquet/default/train/0000.parquet",
            "test": f"hf://datasets/{dataset_name}@refs/convert/parquet/default/test/0000.parquet",
        },
    )
