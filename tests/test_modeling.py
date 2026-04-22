from banking77_intent_classifier.config import ClassifierConfig, EncoderConfig, TfidfConfig
from banking77_intent_classifier.modeling import (
    build_pipeline,
    extract_weight_export,
    predict_labels,
    predict_top_k_labels,
    train_pipeline,
)


def test_linear_pipeline_trains_and_exports_weights() -> None:
    texts = [
        "check account balance",
        "show me my card delivery",
        "bank transfer pending",
        "credit card has not arrived",
    ]
    labels = [0, 1, 2, 1]

    pipeline = build_pipeline(
        model_family="tfidf_svc",
        tfidf_config=TfidfConfig(max_features=100),
        encoder_config=EncoderConfig(),
        classifier_config=ClassifierConfig(max_iter=1000),
    )
    trained = train_pipeline(pipeline, texts, labels)
    weight_export = extract_weight_export(trained)

    assert weight_export.coefficients.shape[0] == 3
    assert len(weight_export.feature_names) > 0


def test_predict_top_k_labels_returns_ranked_label_ids() -> None:
    texts = [
        "account balance check",
        "card delivery pending",
        "bank transfer issue",
    ]
    labels = [0, 1, 2]

    pipeline = build_pipeline(
        model_family="tfidf_svc",
        tfidf_config=TfidfConfig(max_features=100),
        encoder_config=EncoderConfig(),
        classifier_config=ClassifierConfig(max_iter=1000),
    )
    trained = train_pipeline(pipeline, texts, labels)
    top_k_predictions = predict_top_k_labels(trained, ["card delivery issue"], k=2)

    assert top_k_predictions.shape == (1, 2)
    assert set(top_k_predictions[0].tolist()).issubset({0, 1, 2})


def test_build_pipeline_enables_lemmatizer_tokenizer() -> None:
    pipeline = build_pipeline(
        model_family="tfidf_svc",
        tfidf_config=TfidfConfig(normalization="lemma", max_features=100),
        encoder_config=EncoderConfig(),
        classifier_config=ClassifierConfig(max_iter=1000),
    )

    vectorizer = pipeline.named_steps["vectorizer"]

    assert vectorizer.token_pattern is None
    assert vectorizer.tokenizer is not None


def test_build_pipeline_creates_sentence_transformer_family() -> None:
    pipeline = build_pipeline(
        model_family="sentence_transformer_linear",
        tfidf_config=TfidfConfig(),
        encoder_config=EncoderConfig(model_name="all-MiniLM-L6-v2"),
        classifier_config=ClassifierConfig(max_iter=1000),
    )

    assert "encoder" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_build_pipeline_creates_sentence_transformer_knn_family() -> None:
    pipeline = build_pipeline(
        model_family="sentence_transformer_knn",
        tfidf_config=TfidfConfig(),
        encoder_config=EncoderConfig(model_name="all-MiniLM-L6-v2"),
        classifier_config=ClassifierConfig(knn_neighbors=5, knn_metric="cosine"),
    )

    assert "encoder" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_build_pipeline_rejects_probability_threshold_for_tfidf_svc() -> None:
    try:
        build_pipeline(
            model_family="tfidf_svc",
            tfidf_config=TfidfConfig(),
            encoder_config=EncoderConfig(),
            classifier_config=ClassifierConfig(),
            require_probabilities=True,
        )
    except ValueError as error:
        assert "currently supported for sentence-transformer models" in str(error)
    else:
        raise AssertionError("Expected ValueError when requiring probabilities for tfidf_svc.")


def test_predict_labels_forces_low_probability_predictions_to_oos() -> None:
    class DummyClassifier:
        classes_ = [0, 1, 2]

    class DummyPipeline:
        named_steps = {"classifier": DummyClassifier()}

        def predict(self, texts):
            raise AssertionError("predict should not be used when oos_confidence_threshold is set")

        def predict_proba(self, texts):
            return [
                [0.2, 0.7, 0.1],
                [0.2, 0.25, 0.55],
                [0.2, 0.25, 0.28],
            ]

    predictions = predict_labels(
        DummyPipeline(),
        ["a", "b", "c"],
        oos_label_id=2,
        oos_confidence_threshold=0.3,
    )

    assert predictions.tolist() == [1, 2, 2]


def test_predict_labels_forces_low_margin_predictions_to_oos() -> None:
    class DummyClassifier:
        classes_ = [0, 1, 2]

    class DummyPipeline:
        named_steps = {"classifier": DummyClassifier()}

        def predict(self, texts):
            raise AssertionError("predict should not be used when OOS thresholds are set")

        def predict_proba(self, texts):
            return [
                [0.1, 0.8, 0.1],
                [0.1, 0.52, 0.38],
                [0.1, 0.36, 0.34],
            ]

    predictions = predict_labels(
        DummyPipeline(),
        ["a", "b", "c"],
        oos_label_id=2,
        oos_margin_threshold=0.15,
    )

    assert predictions.tolist() == [1, 2, 2]
