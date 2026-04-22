---
license: mit
library_name: scikit-learn
tags:
- intent-classification
- text-classification
- clinc150
- out-of-scope-detection
- sentence-transformers
- linear-svc
datasets:
- clinc150
language:
- en
pipeline_tag: text-classification
---

# CLINC150 Intent Classifier

Sentence-embedding intent classifier for CLINC150 using `sentence-transformers/all-mpnet-base-v2` embeddings, a calibrated `LinearSVC` head, and a max-probability OOS threshold of `0.5`.

## Overview

This model is the selected CLINC150 champion from a progression of sparse and embedding-based experiments. It is designed for 151-way English intent classification on short utterances, including explicit out-of-scope (`oos`) handling.

Final model:

- Encoder: `sentence-transformers/all-mpnet-base-v2`
- Classifier: calibrated `LinearSVC`
- Dataset: CLINC150
- Labels: 151 intents including `oos`
- OOS max-probability threshold: `0.5`
- Embedding dimension: 768

## Results

Held-out CLINC150 test split performance:

| Metric | Score |
|---|---:|
| Accuracy | 0.9196 |
| Macro F1 | 0.9411 |
| Top-5 Accuracy | 0.9315 |
| OOS Precision | 0.8207 |
| OOS Recall | 0.8150 |
| OOS F1 | 0.8179 |
| OOS Miss Rate | 0.1850 |
| In-scope False OOS Rate | 0.0396 |

Test split size: 5,500 examples.

## Why This Model

This model was selected after comparing:

- TF-IDF + `LinearSVC`
- lemmatized TF-IDF variants
- TF-IDF hyperparameter search
- sentence-transformer + `LinearSVC` with multiple encoders
- OOS thresholded sentence-transformer variants

The `all-mpnet-base-v2 + calibrated LinearSVC + OOS threshold 0.5` combination gave the best balance of overall accuracy, macro F1, and practical OOS handling on CLINC150.

## Files

This model repo is expected to contain:

- `model.joblib`
- `label_mapping.json`
- `feature_mapping.json`
- `model_metadata.json`
- `champion.json`
- `requirements.txt`

## Intended Use

Use this model for:

- English intent classification with explicit OOS handling
- short, single-turn assistant-style utterances
- CLINC150-style taxonomies or similar routing problems with out-of-scope examples

## Out-of-Scope / Limitations

- The model predicts only among the CLINC150 label set plus `oos`.
- The OOS behavior depends on the fixed max-probability threshold of `0.5`.
- It may degrade on multilingual, long-form, or very domain-specific text.
- This is a serialized sklearn pipeline, not a fine-tuned Transformers checkpoint.

## Training Setup

- Dataset: CLINC150
- Train split: `train` + `oos_train`
- Validation split: `val` + `oos_val`
- Test split: `test` + `oos_test`
- Random seed: `42`
- Encoder normalization: enabled
- Classifier: calibrated `LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)`
- Calibration: `sigmoid`, `cv=5`
- OOS rule: predict `oos` when max class probability is below `0.5`

## Inference Example

```python
from banking77_intent_classifier.inference import load_predictor

predictor = load_predictor("model.joblib", "label_mapping.json")
prediction = predictor.predict_one("how much has the dow changed today")
print(prediction.label, prediction.label_id)
```

## Practical Notes

- The uploaded model is a serialized sklearn pipeline, not a native Transformers checkpoint.
- The encoder dependency comes from `sentence-transformers`.
- For reproducibility, keep `requirements.txt`, `champion.json`, and `model_metadata.json` together with the model files.

## Citation

If you use this model, please cite:

- the CLINC150 dataset
- the upstream `sentence-transformers/all-mpnet-base-v2` encoder
- this repository/model release
