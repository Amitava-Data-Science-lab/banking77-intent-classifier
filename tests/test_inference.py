from banking77_intent_classifier.config import ClassifierConfig, EncoderConfig, TfidfConfig
import numpy as np

from banking77_intent_classifier.inference import Predictor, RerankingPredictor
from banking77_intent_classifier.modeling import build_pipeline, train_pipeline


def test_predictor_wraps_pipeline_predictions() -> None:
    texts = [
        "show account balance",
        "my card has not arrived",
    ]
    labels = [0, 1]

    pipeline = build_pipeline(
        model_family="tfidf_svc",
        tfidf_config=TfidfConfig(max_features=100, min_df=1),
        encoder_config=EncoderConfig(),
        classifier_config=ClassifierConfig(max_iter=1000),
    )
    trained = train_pipeline(pipeline, texts, labels)
    predictor = Predictor(trained, {0: "balance", 1: "card_arrival"})

    prediction = predictor.predict_one("card not arrived")

    assert prediction.label == "card_arrival"


def test_reranking_predictor_uses_reranked_top_candidate() -> None:
    class DummyPipeline:
        def predict(self, texts):
            return np.array([0 for _ in texts])

    class DummyReranker:
        def rerank(self, text, candidate_label_ids, label_text_mapping):
            return np.array([1, 0])

    predictor = RerankingPredictor(
        pipeline=DummyPipeline(),
        label_mapping={0: "verify_my_identity", 1: "unable_to_verify_identity"},
        label_text_mapping={0: "verify my identity", 1: "unable to verify identity"},
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k=2,
    )
    predictor._reranker = DummyReranker()
    predictor._pipeline = DummyPipeline()

    def fake_predict_top_k_labels(pipeline, texts, k):
        return np.array([[0, 1]])

    from banking77_intent_classifier import inference as inference_module

    original = inference_module.predict_top_k_labels
    inference_module.predict_top_k_labels = fake_predict_top_k_labels
    try:
        prediction = predictor.predict_one("cannot verify identity")
    finally:
        inference_module.predict_top_k_labels = original

    assert prediction.label == "unable_to_verify_identity"
