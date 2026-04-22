from banking77_intent_classifier import data


class DummyLabelFeature:
    names = ["balance", "card_arrival"]


class DummySplit:
    def __init__(self, texts: list[str], labels: list[int]) -> None:
        self._texts = texts
        self._labels = labels
        self.features = {"label": DummyLabelFeature()}

    def __getitem__(self, key: str):
        if key == "text":
            return self._texts
        if key == "label":
            return self._labels
        raise KeyError(key)


def test_load_banking77_dataset_prefers_parquet(monkeypatch) -> None:
    train_split = DummySplit(["show balance"], [0])
    test_split = DummySplit(["card arrived?"], [1])
    dataset_bundle = {"train": train_split, "test": test_split}
    calls: list[str] = []

    def fake_load_dataset(path: str, *args, **kwargs):
        calls.append(path)
        if path == "parquet":
            assert kwargs["data_files"]["train"].endswith("/train/0000.parquet")
            assert kwargs["data_files"]["test"].endswith("/test/0000.parquet")
            return dataset_bundle
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(data, "load_dataset", fake_load_dataset)

    result = data.load_banking77_dataset(
        dataset_name="PolyAI/banking77",
        train_split="train",
        validation_split=None,
        test_split="test",
        text_column="text",
        label_column="label",
    )

    assert result.train_texts == ["show balance"]
    assert result.test_labels == [1]
    assert result.label_names == ["balance", "card_arrival"]
    assert calls == ["parquet"]


def test_load_banking77_dataset_falls_back_to_default_loader(monkeypatch) -> None:
    train_split = DummySplit(["show balance"], [0])
    test_split = DummySplit(["card arrived?"], [1])
    dataset_bundle = {"train": train_split, "test": test_split}
    calls: list[str] = []

    def fake_load_dataset(path: str, *args, **kwargs):
        calls.append(path)
        if path == "parquet":
            raise RuntimeError("Dataset scripts are no longer supported, but found banking77.py")
        if path == "PolyAI/banking77":
            return dataset_bundle
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(data, "load_dataset", fake_load_dataset)

    result = data.load_banking77_dataset(
        dataset_name="PolyAI/banking77",
        train_split="train",
        validation_split=None,
        test_split="test",
        text_column="text",
        label_column="label",
    )

    assert result.train_texts == ["show balance"]
    assert result.test_labels == [1]
    assert result.label_names == ["balance", "card_arrival"]
    assert calls == ["parquet", "PolyAI/banking77"]


def test_load_clinc150_dataset_reads_train_val_test_and_oos(tmp_path) -> None:
    dataset_path = tmp_path / "data_full.json"
    dataset_path.write_text(
        (
            "{"
            "\"train\": [[\"book a flight\", \"book_flight\"]], "
            "\"val\": [[\"cancel it\", \"cancel_reservation\"]], "
            "\"test\": [[\"reserve a table\", \"book_restaurant\"]], "
            "\"oos_train\": [[\"what can you do\", \"oos\"]], "
            "\"oos_val\": [[\"who are you\", \"oos\"]], "
            "\"oos_test\": [[\"help me\", \"oos\"]]"
            "}"
        ),
        encoding="utf-8",
    )

    result = data.load_clinc150_dataset(dataset_source=dataset_path)

    assert result.validation_texts == ["cancel it", "who are you"]
    assert result.train_texts == ["book a flight", "what can you do"]
    assert result.test_texts == ["reserve a table", "help me"]
    assert len(result.label_names) == 4
    assert "oos" in result.label_names
    assert result.metadata["dataset_type"] == "clinc150"
    assert result.metadata["oos_label"] == "oos"


def test_load_clinc150_dataset_can_exclude_oos(tmp_path) -> None:
    dataset_path = tmp_path / "data_full.json"
    dataset_path.write_text(
        (
            "{"
            "\"train\": [[\"book a flight\", \"book_flight\"]], "
            "\"val\": [[\"cancel it\", \"cancel_reservation\"]], "
            "\"test\": [[\"reserve a table\", \"book_restaurant\"]], "
            "\"oos_train\": [[\"what can you do\", \"oos\"]], "
            "\"oos_val\": [[\"who are you\", \"oos\"]], "
            "\"oos_test\": [[\"help me\", \"oos\"]]"
            "}"
        ),
        encoding="utf-8",
    )

    result = data.load_clinc150_dataset(dataset_source=dataset_path, include_oos=False)

    assert "oos" not in result.label_names
    assert len(result.train_texts) == 1
    assert len(result.test_texts) == 1


def test_load_dataset_bundle_dispatches_to_clinc150(tmp_path) -> None:
    dataset_path = tmp_path / "data_full.json"
    dataset_path.write_text(
        (
            "{"
            "\"train\": [[\"book a flight\", \"book_flight\"]], "
            "\"val\": [[\"cancel it\", \"cancel_reservation\"]], "
            "\"test\": [[\"reserve a table\", \"book_restaurant\"]], "
            "\"oos_train\": [[\"what can you do\", \"oos\"]], "
            "\"oos_val\": [[\"who are you\", \"oos\"]], "
            "\"oos_test\": [[\"help me\", \"oos\"]]"
            "}"
        ),
        encoding="utf-8",
    )

    result = data.load_dataset_bundle(
        dataset_type="clinc150",
        dataset_name="clinc150",
        dataset_source=str(dataset_path),
        train_split="train",
        validation_split="val",
        test_split="test",
        text_column="text",
        label_column="label",
        include_oos=True,
    )

    assert result.metadata["dataset_type"] == "clinc150"
    assert result.validation_texts == ["cancel it", "who are you"]
