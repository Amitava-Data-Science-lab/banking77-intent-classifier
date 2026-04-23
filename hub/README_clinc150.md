---
license: mit
library_name: sentence-transformers
tags:
- intent-classification
- text-classification
- clinc150
- out-of-scope-detection
- sentence-transformers
- contrastive-learning
datasets:
- clinc150
language:
- en
pipeline_tag: text-classification
---

# CLINC150 Intent Classifier

Contrastive intent classifier for CLINC150 using `sentence-transformers/all-mpnet-base-v2`, exemplar retrieval, `max_similarity` voting, and OOS rejection by score threshold.

## Overview

This model is the selected CLINC150 champion from the repo's staged comparison process. It is designed for 151-way English intent classification on short utterances, with explicit out-of-scope (`oos`) handling performed as retrieval-time rejection rather than as a learned semantic class.

Final model:

- Encoder: `sentence-transformers/all-mpnet-base-v2`
- Training objective: contrastive triplet loss
- Retrieval: in-scope exemplar bank
- Vote strategy: `max_similarity`
- Dataset: CLINC150
- Labels: 151 intents including `oos`
- OOS score threshold: `0.71`
- Embedding dimension: 768

## Results

Held-out CLINC150 test split performance:

| Metric | Score |
|---|---:|
| Accuracy | 0.9373 |
| Macro F1 | 0.9530 |
| Top-5 Accuracy | 0.9505 |
| OOS Precision | 0.9348 |
| OOS Recall | 0.8030 |
| OOS F1 | 0.8639 |
| OOS Miss Rate | 0.1970 |
| In-scope False OOS Rate | 0.0124 |

Test split size: 5,500 examples.

## Why This Model

This model was selected after comparing:

- TF-IDF + `LinearSVC`
- lemmatized TF-IDF variants
- TF-IDF hyperparameter search
- sentence-transformer + `LinearSVC`
- sentence-transformer + KNN
- contrastive sentence-transformer retrieval with multiple voting and thresholding rules

The `all-mpnet-base-v2 + contrastive triplet training + max_similarity retrieval + threshold 0.71` combination gave the best overall balance of accuracy, macro F1, OOS F1, and low in-scope false-OOS rate on CLINC150.

## Files

This model repo is expected to contain:

- `encoder/`
- `exemplar_embeddings.npy`
- `exemplar_label_ids.npy`
- `label_mapping.json`
- `threshold_config.json`
- `training_data_config.json`
- `model_metadata.json`
- `champion.json`

## Intended Use

Use this model for:

- English intent classification with explicit OOS handling
- short, single-turn assistant-style utterances
- CLINC150-style taxonomies or similar routing problems with out-of-scope examples

## Out-of-Scope / Limitations

- The model predicts only among the CLINC150 label set plus `oos`.
- OOS behavior depends on the fixed retrieval-time score threshold.
- It may degrade on multilingual, long-form, or very domain-specific text.
- This is a contrastive sentence-transformer checkpoint plus exemplar-bank retrieval, not a scikit-learn pipeline.

## Training Setup

- Dataset: CLINC150
- Train split: `train` + `oos_train`
- Validation split: `val` + `oos_val`
- Test split: `test` + `oos_test`
- Random seed: `42`
- Triplets per anchor: `10`
- Negative mix: `50%` hard in-scope, `30%` random in-scope, `20%` OOS
- Retrieval rule: `max_similarity`
- OOS rule: predict `oos` when best in-scope score is below `0.71`

## Practical Notes

- This model serves by encoding the query and comparing it against a saved in-scope exemplar bank.
- For reproducibility, keep `threshold_config.json`, `training_data_config.json`, and `model_metadata.json` together with the encoder and exemplar files.

## Citation

If you use this model, please cite:

- the CLINC150 dataset
- the upstream `sentence-transformers/all-mpnet-base-v2` encoder
- this repository/model release
