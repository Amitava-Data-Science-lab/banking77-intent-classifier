"""Dataset loading utilities for Banking77."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from datasets import load_dataset


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DatasetBundle:
    """In-memory representation of the train/test split."""

    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    label_names: list[str]


def load_banking77_dataset(
    dataset_name: str,
    train_split: str,
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

    label_feature = train_dataset.features[label_column]
    if not hasattr(label_feature, "names"):
        raise ValueError(
            "Expected the label column to be a ClassLabel feature with accessible label names."
        )

    return DatasetBundle(
        train_texts=list(train_dataset[text_column]),
        train_labels=list(train_dataset[label_column]),
        test_texts=list(test_dataset[text_column]),
        test_labels=list(test_dataset[label_column]),
        label_names=list(label_feature.names),
    )


def _load_banking77_from_parquet(dataset_name: str):
    return load_dataset(
        "parquet",
        data_files={
            "train": f"hf://datasets/{dataset_name}@refs/convert/parquet/default/train/0000.parquet",
            "test": f"hf://datasets/{dataset_name}@refs/convert/parquet/default/test/0000.parquet",
        },
    )
