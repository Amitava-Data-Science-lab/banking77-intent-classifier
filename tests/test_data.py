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
        test_split="test",
        text_column="text",
        label_column="label",
    )

    assert result.train_texts == ["show balance"]
    assert result.test_labels == [1]
    assert result.label_names == ["balance", "card_arrival"]
    assert calls == ["parquet", "PolyAI/banking77"]
