# Banking77 Intent Classifier

Production-ready intent classification experiments for [Banking77](https://huggingface.co/datasets/PolyAI/banking77), CLINC150, and future datasets, from TF-IDF baselines through sentence-transformer models.

This repository is designed to be a clean public GitHub repo rather than a notebook experiment. It focuses on:

- reusable package structure
- dataset-pluggable experiment entrypoints
- deterministic training configuration
- artifact persistence for production usage
- confusion-matrix analysis to inspect failure modes
- reusable loading and inference helpers

## Current Best For Banking77

The frozen champion model is:

- model family: `sentence_transformer_linear`
- encoder: `BAAI/bge-small-en-v1.5`
- classifier: `LinearSVC`
- accuracy: `0.9308`
- macro F1: `0.9307`
- top-5 accuracy: `0.9893`

This is the selected production model for the Banking77 track in the repository.

Run it with:

```powershell
banking77-train --config configs/champion.json
```

It writes to:

- `artifacts/champion/`
- `reports/champion/`

## Dataset Workflow

The project now supports dataset-specific model selection. The intended workflow for a new dataset is:

1. run a TF-IDF + LinearSVC baseline
2. run a lemmatized sparse baseline
3. run a small sparse sweep or randomized search
4. run sentence-transformer linear baselines
5. optionally compare KNN or reranking
6. promote a dataset-specific champion only after comparison

This keeps the Banking77 evaluation ladder reusable for CLINC150 and future datasets instead of assuming one champion generalizes everywhere.

## Supported Datasets

- `banking77`
  - source: Hugging Face `PolyAI/banking77`
  - splits: `train`, `test`
- `clinc150`
  - source: local UCI-style `data_full.json`
  - splits: `train`, `val`, `test`
  - default behavior: include `oos` as a normal class, giving a 151-class setup

## Model Families

The repository includes:

- `TfidfVectorizer`
- `LinearSVC`
- sentence-transformer encoders
- KNN over dense embeddings

`LinearSVC` is intentionally used instead of a kernel SVM so the model remains fast, scalable, and interpretable. Sentence-transformer variants trade some interpretability for much stronger semantic understanding.

## Quick Start

### 1. Create a virtual environment

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 2. Train the frozen Banking77 champion

```powershell
banking77-train --config configs/champion.json
```

You can also run the script directly:

```powershell
python scripts/train_phase1.py --config configs/champion.json
```

### 3. Train the frozen TF-IDF baseline

```powershell
banking77-train --config configs/tfidf_svc.json
```

### 4. Run the strongest TF-IDF variant

```powershell
banking77-train --config configs/tfidf_svc_lemmatized.json
```

### 5. Run randomized TF-IDF hyperparameter search

```powershell
banking77-tune --config configs/tfidf_svc_lemmatized_random_search.json
```

This runs `RandomizedSearchCV` on the training split only, samples a moderate number of parameter combinations, selects the best configuration by macro F1, then evaluates the best estimator on the untouched test split. It saves:

- `best_params.json`
- `search_settings.json`
- `cv_results.csv`
- the usual model artifacts and evaluation reports for the best estimator

### 6. Run the first sentence-transformer linear baseline

```powershell
banking77-train --config configs/sentence_transformer_linear.json
```

This uses a pretrained sentence-transformer encoder (`all-MiniLM-L6-v2`) to embed each utterance, then trains a `LinearSVC` classifier on top of those dense embeddings.

### 7. Run the sentence-transformer KNN comparison

```powershell
banking77-train --config configs/sentence_transformer_knn.json
```

### 8. Run the winning encoder comparison directly

```powershell
banking77-train --config configs/sentence_transformer_linear_bge_small.json
```

This keeps the linear SVM head the same, but swaps the encoder to `BAAI/bge-small-en-v1.5` for a final apples-to-apples embedding comparison.

### 9. Run the reranked champion experiment

```powershell
banking77-train --config configs/sentence_transformer_linear_reranked.json
```

This keeps the champion candidate generator (`BAAI/bge-small-en-v1.5 + LinearSVC`), takes its Top-5 candidate labels, and reranks them with `cross-encoder/ms-marco-MiniLM-L-6-v2` using readable label text.

This experiment is kept for reference only. It significantly underperformed the frozen champion and is not recommended for production use.

## CLINC150 Quick Start

1. Download `data_full.json` from the [UCI CLINC150 page](https://archive.ics.uci.edu/dataset/570/clinc150).
2. Place it at `data/clinc150/data_full.json` relative to the repo root.
3. Run the staged experiment configs instead of jumping directly to one model:

```powershell
banking77-train --config configs/clinc150_tfidf_svc.json
banking77-train --config configs/clinc150_tfidf_svc_lemmatized.json
banking77-tune --config configs/clinc150_tfidf_svc_random_search.json
banking77-train --config configs/clinc150_sentence_transformer_linear.json
banking77-train --config configs/clinc150_sentence_transformer_knn.json
banking77-train --config configs/clinc150_sentence_transformer_linear_bge_small.json
```

CLINC150 outputs are namespaced under `artifacts/clinc150/` and `reports/clinc150/` so they do not overwrite Banking77 experiments.

## Outputs

Each run writes production-friendly artifacts to the configured `artifacts/` and `reports/` directories. Banking77 and CLINC150 configs use separate directories so each dataset can maintain its own experiment history and future champion alias.

Champion outputs include:

- serialized model pipeline via `joblib`
- model metadata JSON
- label map JSON
- feature mapping JSON
- embedding-space or linear weights when available
- classification report JSON
- confusion matrix CSV
- normalized confusion matrix CSV
- top confusion pairs CSV
- top features by class CSV
- confusion matrix PNG

## Inference Example

```python
from banking77_intent_classifier.inference import load_predictor

predictor = load_predictor("artifacts/champion/model.joblib", "artifacts/champion/label_mapping.json")
prediction = predictor.predict_one("My card still has not arrived")
print(prediction.label, prediction.label_id)
```

## Example Files

After training the champion config, expect output similar to:

```text
artifacts/champion/
|-- feature_mapping.json
|-- label_mapping.json
|-- model.joblib
|-- model_metadata.json
`-- weights.npz

reports/champion/
|-- classification_report.json
|-- confusion_matrix.csv
|-- confusion_matrix.png
|-- confusion_matrix_normalized.csv
|-- metrics_summary.json
|-- top_confusions.csv
`-- top_features_by_class.csv
```

## Production Notes

- Training is deterministic via an explicit random seed.
- All important outputs are persisted for reproducibility and deployment.
- The code is packaged for reuse instead of being tied to a single notebook.
- Model diagnostics are written to disk so the same training job can support CI, batch training, or future orchestration.
- The recommended Banking77 production config is `configs/champion.json`.
- CLINC150 should be evaluated through its full config ladder before freezing a champion alias.

## Publishing

To publish the champion model:

- Hugging Face model card template: `hub/README.md`
- W&B artifact upload script: `scripts/upload_champion_to_wandb.py`
- Hugging Face upload script: `scripts/upload_champion_to_hf.py`

If you have not run `configs/champion.json` yet, either run it first or point the upload scripts at the existing `sentence_transformer_linear_bge_small` artifact/report directories.

## Tests

```powershell
pytest
```

## Experiment History

- `tfidf_svc`: frozen sparse baseline
- `tfidf_svc_trigrams`: trigram experiment
- `tfidf_svc_lemmatized`: strongest TF-IDF family variant
- `tfidf_svc_lemmatized_c*`: `C` sweep around the lemmatized TF-IDF model
- `tfidf_svc_lemmatized_random_search`: final randomized TF-IDF search
- `sentence_transformer_linear`: first embedding-based linear model with `all-MiniLM-L6-v2`
- `sentence_transformer_knn`: embedding-based KNN comparison
- `sentence_transformer_linear_bge_small`: best model found
- `sentence_transformer_linear_reranked`: cross-encoder reranking experiment, rejected after a large regression
- `champion`: frozen alias for the preferred production candidate
