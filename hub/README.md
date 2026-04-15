---
license: mit
library_name: scikit-learn
tags:
- intent-classification
- text-classification
- banking77
- sentence-transformers
- linear-svc
datasets:
- PolyAI/banking77
language:
- en
pipeline_tag: text-classification
---

# Banking77 Intent Classifier

Sentence-embedding intent classifier for Banking77 using `BAAI/bge-small-en-v1.5` embeddings and a `LinearSVC` classification head.

## Overview

This model is the final selected champion from a progression of experiments that started with TF-IDF + linear SVM baselines and moved to sentence-transformer embeddings. It is designed for 77-way English banking support intent classification on short customer utterances.

Final model:

- Encoder: `BAAI/bge-small-en-v1.5`
- Classifier: `LinearSVC`
- Dataset: `PolyAI/banking77`
- Labels: 77 intents
- Embedding dimension: 384

## Results

Held-out Banking77 test split performance:

| Metric | Score |
|---|---:|
| Accuracy | 0.9308 |
| Macro F1 | 0.9307 |
| Top-5 Accuracy | 0.9893 |

Test split size: 3,080 examples.

## Why This Model

This model was selected after comparing:

- TF-IDF + `LinearSVC`
- lemmatized TF-IDF variants
- TF-IDF hyperparameter search
- sentence-transformer + `LinearSVC`
- sentence-transformer + KNN
- cross-encoder reranking on top of Top-5 candidates

The `BAAI/bge-small-en-v1.5 + LinearSVC` combination gave the best overall balance of accuracy, macro F1, simplicity, and production readiness.

## Files

This model repo is expected to contain:

- `model.joblib`
- `label_mapping.json`
- `feature_mapping.json`
- `model_metadata.json`
- `weights.npz`
- `champion.json`
- `requirements.txt`

## Intended Use

Use this model for:

- intent classification of English banking support messages
- short, single-turn customer utterances
- Banking77-style taxonomies or similar support-routing scenarios

This is a good fit when you want:

- strong semantic performance
- simple deployment with a serialized sklearn pipeline
- better performance than sparse lexical baselines without full transformer fine-tuning

## Out-of-Scope / Limitations

- The model predicts only among the Banking77 label set.
- It is tuned for Banking77-style language and may not transfer cleanly to a different intent taxonomy.
- It may degrade on multilingual, long-form, or strongly out-of-domain text.
- This is not a generative or conversational model.

## Training Setup

- Dataset: `PolyAI/banking77`
- Train split: `train`
- Test split: `test`
- Random seed: `42`
- Encoder normalization: enabled
- Classifier: `LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)`

## Inference Example

```python
from banking77_intent_classifier.inference import load_predictor

predictor = load_predictor("model.joblib", "label_mapping.json")
prediction = predictor.predict_one("My card still has not arrived")
print(prediction.label, prediction.label_id)
```

## Practical Notes

- The uploaded model is a serialized sklearn pipeline, not a native Transformers checkpoint.
- The encoder dependency comes from `sentence-transformers`.
- For reproducibility, keep `requirements.txt`, `champion.json`, and `model_metadata.json` together with the model files.

## Citation

If you use this model, please cite:

- the Banking77 dataset
- the upstream `BAAI/bge-small-en-v1.5` encoder
- this repository/model release
