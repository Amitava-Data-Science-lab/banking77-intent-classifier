from pathlib import Path

from banking77_intent_classifier.config import load_config, load_tuning_config


def test_load_config_reads_tfidf_svc_file() -> None:
    config = load_config(Path("configs/tfidf_svc.json"))

    assert config.dataset_type == "banking77"
    assert config.dataset_name == "PolyAI/banking77"
    assert config.tfidf.ngram_range == (1, 2)
    assert config.classifier.max_iter == 5000


def test_load_config_reads_tfidf_svc_trigrams_file() -> None:
    config = load_config(Path("configs/tfidf_svc_trigrams.json"))

    assert config.dataset_name == "PolyAI/banking77"
    assert config.tfidf.ngram_range == (1, 3)
    assert config.classifier.max_iter == 5000


def test_load_config_reads_tfidf_svc_lemmatized_file() -> None:
    config = load_config(Path("configs/tfidf_svc_lemmatized.json"))

    assert config.dataset_name == "PolyAI/banking77"
    assert config.tfidf.normalization == "lemma"
    assert config.tfidf.ngram_range == (1, 2)


def test_load_config_reads_c_sweep_files() -> None:
    low_c_config = load_config(Path("configs/tfidf_svc_lemmatized_c001.json"))
    high_c_config = load_config(Path("configs/tfidf_svc_lemmatized_c100.json"))

    assert low_c_config.classifier.c == 0.01
    assert high_c_config.classifier.c == 100.0
    assert low_c_config.tfidf.normalization == "lemma"
    assert high_c_config.tfidf.normalization == "lemma"


def test_load_tuning_config_normalizes_search_space() -> None:
    config = load_tuning_config(Path("configs/tfidf_svc_lemmatized_random_search.json"))

    assert config.experiment.tfidf.normalization == "lemma"
    assert config.search.search_type == "randomized"
    assert config.search.cv == 5
    assert config.search.param_distributions["vectorizer__ngram_range"] == [(1, 2), (1, 3)]
    assert config.search.param_distributions["classifier__C"] == [0.1, 1.0, 10.0]
    assert config.search.n_iter == 20


def test_load_config_reads_sentence_transformer_file() -> None:
    config = load_config(Path("configs/sentence_transformer_linear.json"))

    assert config.model_family == "sentence_transformer_linear"
    assert config.encoder.model_name == "all-MiniLM-L6-v2"
    assert config.classifier.c == 1.0


def test_load_config_reads_sentence_transformer_knn_file() -> None:
    config = load_config(Path("configs/sentence_transformer_knn.json"))

    assert config.model_family == "sentence_transformer_knn"
    assert config.encoder.model_name == "all-MiniLM-L6-v2"
    assert config.classifier.knn_neighbors == 5
    assert config.classifier.knn_metric == "cosine"


def test_load_config_reads_sentence_transformer_bge_small_file() -> None:
    config = load_config(Path("configs/sentence_transformer_linear_bge_small.json"))

    assert config.model_family == "sentence_transformer_linear"
    assert config.encoder.model_name == "BAAI/bge-small-en-v1.5"
    assert config.classifier.c == 1.0


def test_load_config_reads_champion_file() -> None:
    config = load_config(Path("configs/champion.json"))

    assert config.model_family == "sentence_transformer_linear"
    assert config.encoder.model_name == "BAAI/bge-small-en-v1.5"
    assert config.artifacts_dir.name == "champion"


def test_load_config_reads_reranked_file() -> None:
    config = load_config(Path("configs/sentence_transformer_linear_reranked.json"))

    assert config.model_family == "sentence_transformer_linear_reranked"
    assert config.encoder.model_name == "BAAI/bge-small-en-v1.5"
    assert config.reranker.enabled is True
    assert config.reranker.top_k == 5


def test_load_config_reads_clinc150_file() -> None:
    config = load_config(Path("configs/clinc150_tfidf_svc.json"))

    assert config.dataset_type == "clinc150"
    assert config.dataset_name == "clinc150"
    assert config.dataset_task == "full_intent"
    assert config.dataset_source.name == "data_full.json"
    assert config.validation_split == "val"
    assert config.include_oos is True


def test_load_tuning_config_reads_clinc150_file() -> None:
    config = load_tuning_config(Path("configs/clinc150_tfidf_svc_random_search.json"))

    assert config.experiment.dataset_type == "clinc150"
    assert config.experiment.validation_split == "val"
    assert config.experiment.include_oos is True
    assert config.search.search_type == "randomized"


def test_load_config_reads_clinc150_sentence_threshold_file() -> None:
    config = load_config(Path("configs/clinc150_sentence_transformer_linear_bge_small_oos_threshold_03.json"))

    assert config.model_family == "sentence_transformer_linear"
    assert config.oos_confidence_threshold == 0.3
    assert config.oos_margin_threshold is None
    assert config.classifier.probability_calibration_method == "sigmoid"
    assert config.classifier.probability_calibration_cv == 5


def test_load_config_reads_clinc150_sentence_mpnet_file() -> None:
    config = load_config(Path("configs/clinc150_sentence_transformer_linear_mpnet_base_v2.json"))

    assert config.model_family == "sentence_transformer_linear"
    assert config.dataset_type == "clinc150"
    assert config.encoder.model_name == "sentence-transformers/all-mpnet-base-v2"
    assert config.artifacts_dir.name == "sentence_transformer_linear_mpnet_base_v2"
    assert config.reports_dir.name == "sentence_transformer_linear_mpnet_base_v2"


def test_load_config_reads_clinc150_transformer_mpnet_file() -> None:
    config = load_config(Path("configs/clinc150_transformer_sequence_classifier_mpnet.json"))

    assert config.model_family == "transformer_sequence_classifier"
    assert config.dataset_type == "clinc150"
    assert config.transformer.model_name == "sentence-transformers/all-mpnet-base-v2"
    assert config.transformer.load_best_model_at_end is False
    assert config.transformer.learning_rate == 1e-5
    assert config.transformer.num_train_epochs == 5.0
    assert config.transformer.warmup_steps == 236
    assert config.transformer.threshold_candidates == [0.001, 0.005, 0.01, 0.02, 0.05]
    assert config.transformer.threshold_selection_metric == "macro_f1"
    assert config.transformer.threshold_selection_strategy == "oos_aware_constrained"
    assert config.transformer.threshold_max_in_scope_false_oos_rate == 0.03
    assert config.transformer.threshold_macro_f1_tolerance_ladder == [0.01, 0.02, 0.03, 0.04, 0.05]
    assert config.transformer.threshold_fallback_strategy == "best_macro_f1"
    assert config.transformer.oos_distance_enabled is False


def test_load_config_reads_clinc150_transformer_mpnet_distance_file() -> None:
    config = load_config(Path("configs/clinc150_transformer_sequence_classifier_mpnet_distance.json"))

    assert config.model_family == "transformer_sequence_classifier"
    assert config.dataset_task == "full_intent"
    assert config.transformer.oos_distance_enabled is True
    assert config.transformer.oos_distance_metric == "cosine"
    assert config.transformer.oos_distance_candidate_source == "validation_distances"
    assert config.transformer.oos_distance_selection_strategy == "oos_aware_constrained"
    assert config.transformer.oos_fixed_probability_source == "selected_validation_threshold"
    assert config.artifacts_dir.name == "transformer_sequence_classifier_mpnet_distance"


def test_load_config_reads_clinc150_transformer_mpnet_energy_file() -> None:
    config = load_config(Path("configs/clinc150_transformer_sequence_classifier_mpnet_energy.json"))

    assert config.model_family == "transformer_sequence_classifier"
    assert config.dataset_task == "full_intent"
    assert config.transformer.oos_energy_enabled is True
    assert config.transformer.oos_energy_temperature == 1.0
    assert config.transformer.oos_energy_candidate_source == "validation_energies"
    assert config.transformer.oos_energy_selection_strategy == "oos_aware_constrained"
    assert config.transformer.oos_energy_fixed_probability_source == "selected_validation_threshold"
    assert config.artifacts_dir.name == "transformer_sequence_classifier_mpnet_energy"


def test_load_config_reads_clinc150_transformer_mpnet_temperature_file() -> None:
    config = load_config(Path("configs/clinc150_transformer_sequence_classifier_mpnet_temperature_scaled.json"))

    assert config.model_family == "transformer_sequence_classifier"
    assert config.dataset_task == "full_intent"
    assert config.transformer.temperature_scaling_enabled is True
    assert config.transformer.temperature_scaling_grid == [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    assert config.artifacts_dir.name == "transformer_sequence_classifier_mpnet_temperature_scaled"


def test_load_config_reads_clinc150_binary_oos_tfidf_file() -> None:
    config = load_config(Path("configs/clinc150_binary_oos_tfidf_svc_lemmatized.json"))

    assert config.dataset_type == "clinc150"
    assert config.dataset_task == "binary_oos"
    assert config.tfidf.normalization == "lemma"
    assert config.artifacts_dir.name == "binary_oos_tfidf_svc_lemmatized"


def test_load_config_reads_clinc150_binary_oos_bge_file() -> None:
    config = load_config(Path("configs/clinc150_binary_oos_sentence_transformer_linear_bge_small.json"))

    assert config.model_family == "sentence_transformer_linear"
    assert config.dataset_task == "binary_oos"
    assert config.encoder.model_name == "BAAI/bge-small-en-v1.5"
    assert config.reports_dir.name == "binary_oos_sentence_transformer_linear_bge_small"


def test_load_config_reads_clinc150_binary_oos_transformer_mpnet_file() -> None:
    config = load_config(Path("configs/clinc150_binary_oos_transformer_sequence_classifier_mpnet.json"))

    assert config.model_family == "transformer_sequence_classifier"
    assert config.dataset_task == "binary_oos"
    assert config.transformer.model_name == "sentence-transformers/all-mpnet-base-v2"
    assert config.transformer.threshold_selection_strategy == "oos_aware_constrained"
    assert config.artifacts_dir.name == "binary_oos_transformer_sequence_classifier_mpnet"


def test_load_config_reads_sentence_threshold_and_margin_file() -> None:
    config = load_config(
        Path("configs/clinc150_sentence_transformer_linear_bge_small_oos_threshold_03_margin_01.json")
    )

    assert config.oos_confidence_threshold == 0.3
    assert config.oos_margin_threshold == 0.1


def test_load_config_reads_clinc150_champion_file() -> None:
    config = load_config(Path("configs/clinc150_champion.json"))

    assert config.model_family == "sentence_transformer_linear"
    assert config.encoder.model_name == "sentence-transformers/all-mpnet-base-v2"
    assert config.oos_confidence_threshold == 0.5
    assert config.oos_margin_threshold == 0.0
    assert config.artifacts_dir.name == "champion"


def test_load_config_reads_clinc150_contrastive_mpnet_file() -> None:
    config = load_config(Path("configs/clinc150_sentence_transformer_contrastive_mpnet.json"))

    assert config.model_family == "sentence_transformer_contrastive_knn"
    assert config.dataset_type == "clinc150"
    assert config.contrastive.model_name == "sentence-transformers/all-mpnet-base-v2"
    assert config.contrastive.triplets_per_anchor == 10
    assert config.contrastive.hard_negative_ratio == 0.5
    assert config.contrastive.random_negative_ratio == 0.3
    assert config.contrastive.oos_negative_ratio == 0.2
    assert config.contrastive.threshold_candidates == [3.1]
    assert config.contrastive.threshold_selection_strategy == "oos_aware_constrained"


def test_load_config_reads_clinc150_contrastive_maxsim_candidate_file() -> None:
    config = load_config(Path("configs/clinc150_sentence_transformer_contrastive_mpnet_maxsim_candidate.json"))

    assert config.model_family == "sentence_transformer_contrastive_knn"
    assert config.contrastive.vote_strategy == "max_similarity"
    assert config.contrastive.threshold_candidates == [0.65, 0.69, 0.73, 0.76, 0.8]
    assert config.artifacts_dir.name == "sentence_transformer_contrastive_mpnet_maxsim_candidate"
