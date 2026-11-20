<div align="center">
  <h1>Credit Scoring MLOps</h1>
  <p><strong>Reproducible, leakage-aware credit-default modeling with explainability and model-risk guardrails.</strong></p>
  <p>
    <a href="https://github.com/Vishal123-tech/Credit-Scoring-Mlops/actions/workflows/ci.yml">
      <img src="https://img.shields.io/github/actions/workflow/status/Vishal123-tech/Credit-Scoring-Mlops/ci.yml?branch=main&label=CI" alt="CI status">
    </a>
    <a href="https://github.com/Vishal123-tech/Credit-Scoring-Mlops">
      <img src="https://img.shields.io/github/languages/top/Vishal123-tech/Credit-Scoring-Mlops" alt="Top language">
    </a>
    <a href="https://www.python.org/downloads/release/python-3100/">
      <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10">
    </a>
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
    </a>
  </p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/504ad478-e1fe-4ab6-a9ac-82d41de85a78" width="300" alt="Credit scoring project visual">
</div>

> A research and training engine for credit-risk modeling. It is designed to demonstrate reproducible ML engineering, evaluation, and governance patterns—not to make real-world lending decisions without additional validation, controls, and regulatory review.

## Project at a glance

| Area | What this repository delivers |
|---|---|
| Business problem | Estimate the probability that a borrower will experience a serious repayment difficulty within a 12-month horizon. |
| Data | Home Credit Default Risk data joined from seven relational tables into borrower-level features. |
| Modeling | Logistic Regression, LightGBM, XGBoost, and CatBoost with Optuna-based hyperparameter optimization. |
| Validation | Leakage-safe preprocessing, stratified out-of-fold evaluation, a never-fitted holdout set, calibration, and threshold selection. |
| Governance | SHAP explanations, fairness checks, PSI drift monitoring, reliability outputs, and a guarded champion-promotion workflow. |
| MLOps | DVC pipeline orchestration, MLflow experiment tracking and registry, locked dependencies, tests, CI, and serving-artifact export. |
| Companion product | Exported models can be consumed by the separate [LoanSight](https://github.com/mikhailvokhrameev/loan_sight.git) application. |

## Why this project

Credit scoring is not only a classification problem. A useful risk workflow must also answer:

- Can the result be reproduced from versioned inputs and parameters?
- Was information from the holdout or future rows allowed to leak into training?
- Are predicted probabilities calibrated well enough to support business thresholds?
- Can a reviewer understand why a borrower received a high-risk score?
- Are fairness, drift, and promotion decisions visible before a model is used downstream?

This project treats those questions as first-class pipeline outputs rather than afterthoughts.

## Architecture

~~~mermaid
flowchart LR
    A[Home Credit CSV tables] --> B[DVC prepare and split]
    B --> C[Modular feature engineering]
    C --> D[Training feature set]
    D --> E[OOF CV and Optuna HPO]
    E --> F[Final model and probability calibration]
    F --> G[Holdout evaluation]
    G --> H[SHAP, fairness, reliability and PSI]
    F --> I[MLflow tracking and registry]
    I --> J[Promotion gate]
    F --> K[Serving artifact export]
    K --> L[LoanSight application]
~~~

The default DVC graph is:

<code>prepare</code> → <code>split</code> → <code>features</code> → <code>train@{logreg,lgbm,xgb,catboost}</code>

## Core capabilities

### Data and feature engineering

- Converts raw CSV tables to cached Apache Parquet files for faster iteration.
- Builds borrower-level features from application, bureau, previous-application, installment, credit-card, POS-cash, and payment-history sources.
- Fits preprocessing, rare-category handling, and feature selection on the training partition only.
- Selects the configured top 300 features using the training data.

### Modeling

- Provides a shared model contract for Logistic Regression, LightGBM, XGBoost, and CatBoost.
- Uses Optuna to optimize mean cross-validated ROC-AUC.
- Handles severe class imbalance and supports early stopping for boosting models.
- Fits a sigmoid probability calibrator on out-of-fold predictions, then preserves the calibrated wrapper for evaluation and serving.
- Selects business thresholds from out-of-fold predictions rather than tuning on the holdout.

### Evaluation and governance

- Reports ROC-AUC, Gini, KS statistic, average precision, Brier score, business loss, false positives, false negatives, and approval rate.
- Produces reliability tables and calibration plots.
- Audits disparate impact and equal-opportunity gaps by gender and age while keeping those attributes out of the model input set.
- Calculates PSI for the score and top features; drift is surfaced as monitoring evidence.
- Promotes a challenger to champion only when the configured holdout and fairness gates pass.
- Generates global and local SHAP explanations for model review.

## Evaluation design

The pipeline separates model development from final evaluation:

1. A 15% holdout is created by borrower ID and is not used to fit preprocessing, feature selection, hyperparameters, calibration, or thresholds.
2. The training partition is evaluated with stratified out-of-fold predictions.
3. Optuna uses the configured HPO folds; the final training stage uses the configured CV folds.
4. The sigmoid calibrator and business threshold are fitted from out-of-fold predictions.
5. The final model is refit on the complete training partition and scored on the holdout once.
6. Holdout performance, calibration, fairness, and drift outputs are saved with the model artifacts and logged to MLflow.

Protected attributes are retained for audit outputs but excluded from model inputs through [params.yaml](params.yaml). The exact configuration, seeds, gates, costs, and monitoring thresholds are versioned in that file.

## Historical benchmark

> **Important:** The table below contains historical out-of-fold results from before leakage fix F1. It is preserved for research context only and must not be described as current holdout or production performance. Raw data and trained artifacts are intentionally excluded from Git. Run the pipeline locally to generate current leakage-safe holdout metrics.

| Algorithm | Version | ROC-AUC | PR-AUC | Gini | Business loss |
|---|---|---:|---:|---:|---:|
| **XGBoost** | Full | **0.788** | **0.281** | **0.576** | **149,582** |
| CatBoost | Full | 0.787 | 0.278 | 0.573 | 150,005 |
| LightGBM | Full | 0.786 | 0.278 | 0.571 | 150,627 |
| LightGBM | Non-HPO | 0.786 | 0.277 | 0.572 | 150,029 |
| LogReg | Full | 0.773 | 0.253 | 0.546 | 155,780 |
| LogReg | Base baseline | 0.751 | 0.235 | 0.502 | 165,541 |

Historical observations:

- Enriching a single application table with auxiliary history produced a larger reported gain than changing between several full-data algorithms.
- PR-AUC is especially relevant for this imbalanced target because it focuses attention on the rare positive class.
- Optuna improved the historical XGBoost and CatBoost results, while the reported LightGBM comparison showed no gain.

## Visual tour

<details open>
<summary><strong>Experiment tracking and model registry</strong></summary>
<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/ac4c1d9d-8be4-4f1f-b32f-f6e79e03962c" width="800" alt="MLflow UI dashboard">
  <br><em>MLflow experiment dashboard</em>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/1a23353e-d1cd-4bd0-87d5-b5d163a72fb7" width="800" alt="MLflow run list">
  <br><em>Run history and model comparison</em>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/b0ce9a32-1767-431c-be0e-22ca49c46228" width="800" alt="MLflow model registry">
  <br><em>Model registry and aliases</em>
</p>
</details>

<details>
<summary><strong>Explainability and model analysis</strong></summary>
<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/829e180e-bab0-41cf-a42e-02f3684623cb" width="700" alt="SHAP explanation for a high-risk client">
  <br><em>Local SHAP explanation for a high-risk client</em>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/28f71ec9-a480-4446-8b30-efdcdd1a1841" width="650" alt="SHAP summary bar chart">
  <br><em>Global SHAP feature importance</em>
</p>
</details>

<details>
<summary><strong>Research benchmark</strong></summary>
<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/ca246b5c-4787-44d0-801d-d5f5402801af" width="900" alt="Historical model comparison">
  <br><em>Historical out-of-fold model comparison; see the benchmark warning above.</em>
</p>
</details>

## Repository layout

~~~text
credit-scoring-mlops/
├── src/
│   ├── data/          Loading, caching and borrower-level holdout split
│   ├── features/      Table-specific feature engineering and feature selection
│   ├── models/        Training, Optuna tuning and promotion gate
│   ├── evaluation/    Metrics, calibration, fairness and SHAP explanations
│   └── monitoring/    PSI drift monitoring
├── notebooks/         EDA and read-only interpretation notebooks
├── tests/             Leakage, metrics, fairness, drift, calibration and smoke tests
├── docs/              Model card and dataset datasheet
├── params.yaml        Seeds, holdout size, feature exclusions, calibration and gates
├── dvc.yaml           Reproducible pipeline graph
├── export_serving_artifacts.py
├── pyproject.toml
└── Makefile
~~~

## Quickstart

### Prerequisites

- Python 3.10
- [uv](https://docs.astral.sh/uv/)
- A Kaggle API token configured for the [Home Credit Default Risk competition](https://www.kaggle.com/c/home-credit-default-risk)
- Optional: a GPU for faster boosting-model training

### 1. Install the environment

~~~bash
git clone https://github.com/Vishal123-tech/Credit-Scoring-Mlops.git
cd Credit-Scoring-Mlops

uv sync
uv sync --group notebooks
~~~

The second command is optional and installs JupyterLab, profiling, notebook conversion, and Quarto-related dependencies.

### 2. Download the data

Raw Kaggle data is not redistributed in this repository.

~~~bash
uv run kaggle competitions download -c home-credit-default-risk -p data/
~~~

Extract the downloaded archive so the CSV files are under <code>data/raw/</code>, then register the raw directory locally with DVC:

~~~bash
uv run dvc add data/raw
~~~

For provenance, personal fields, licensing notes, and known data anomalies, see the [dataset datasheet](docs/datasheet.md).

### 3. Run the pipeline

To reproduce the complete graph, including all four model families and their configured HPO runs:

~~~bash
uv run dvc repro
~~~

For a faster staged run:

~~~bash
uv run python -m src.data.loader
uv run python -m src.data.split
uv run python -m src.features.pipeline
~~~

The feature stage writes cached data and metadata under <code>data/processed/</code>. It does not fit on the holdout.

### 4. Train a model

~~~bash
uv run python -m src.models.train --model lgbm --trials 30
~~~

Available model names are <code>logreg</code>, <code>lgbm</code>, <code>xgb</code>, and <code>catboost</code>.

Useful options:

| Option | Default | Purpose |
|---|---:|---|
| <code>--model</code> | required | Select the model family. |
| <code>--trials</code> | <code>hpo.trials</code> | Set to <code>0</code> to skip Optuna. |
| <code>--folds</code> | <code>cv.n_splits</code> | Override final CV folds. |
| <code>--device</code> | <code>cpu</code> | Use <code>gpu</code> for supported boosting models. |
| <code>--base</code> | off | Train on the single-table baseline feature set. |
| <code>--params</code> | <code>params.yaml</code> | Use an alternate configuration file. |

### 5. Review experiments

~~~bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
~~~

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to inspect nested HPO runs, metrics, plots, artifacts, and model-registry aliases.

### 6. Export a serving bundle

~~~bash
uv run python export_serving_artifacts.py
~~~

For each complete model artifact, the exporter creates a <code>serving_model.joblib</code> wrapper and <code>metadata.json</code> with the model type, feature names, verified metrics, and decision thresholds. Export fails when required evaluation artifacts are missing.

### 7. Explore notebooks

- [01_EDA.ipynb](notebooks/01_EDA.ipynb) — data profiling, imbalance, missingness, and anomaly analysis.
- [04_interpretation.ipynb](notebooks/04_interpretation.ipynb) — selected features, correlations, global/local SHAP, and cross-model comparison.

Use the notebooks for exploration; use the <code>src/</code> CLI and DVC graph for reproducible training.

## Generated artifacts

| Location | Contents |
|---|---|
| <code>data/cache/</code> | Cached Parquet versions of raw tables. |
| <code>data/processed/</code> | Train, holdout, and test feature sets plus selected-feature metadata. |
| <code>artifacts/&lt;model&gt;_model/</code> | Model wrapper, metrics, thresholds, SHAP outputs, guardrails, and PSI monitoring reports. |
| <code>mlflow.db</code> | Local MLflow tracking store when using the default configuration. |
| <code>optuna.db</code> | Local Optuna study database for resumable HPO. |

These directories are local outputs and are not included in the source repository.

## Tests and CI

Run the quality checks locally:

~~~bash
make check
~~~

Run only the end-to-end synthetic-data smoke test:

~~~bash
make smoke
~~~

The GitHub Actions workflow in [.github/workflows/ci.yml](.github/workflows/ci.yml) installs the locked development environment and runs the repository checks on pushes and pull requests.

## Documentation and responsible use

- [Model card](docs/model_card.md) — intended use, evaluation protocol, fairness and drift guardrails, promotion behavior, and known limitations.
- [Dataset datasheet](docs/datasheet.md) — source, provenance, sensitive fields, licensing notes, and data-quality considerations.
- [Configuration](params.yaml) — seeds, holdout size, feature exclusions, calibration, business costs, fairness thresholds, and drift settings.

This repository intentionally keeps responsible-use boundaries visible:

- Fairness outputs are monitoring evidence, not proof of legal or ethical acceptability.
- Calibration and threshold quality require a representative labelled validation cohort.
- Historical benchmark values are not production claims.
- Any real lending deployment would need domain validation, security, privacy, policy, regulatory, and human-review controls.

## Limitations and next steps

- Raw data and trained artifacts are excluded, so current holdout metrics must be generated locally.
- The temporal holdout is not implemented because the current feature set does not expose a trustworthy application-date field.
- The repository exports model bundles but does not provide an authenticated prediction API, feature store, policy engine, or automated rollback service.
- The legacy model-comparison utility should be aligned with current MLflow run naming before being used for automated comparisons.
- Bank-standard score scaling should be added only after calibration has been validated on a representative labelled cohort.
- A stacking ensemble and fresh labelled-cohort monitoring are natural next extensions.

## LoanSight integration

This repository is the research and training layer for [LoanSight](https://github.com/mikhailvokhrameev/loan_sight.git), a separate web service for experimenting with credit-risk models.

LoanSight can load exported artifacts, retrieve client data, score a selected model, display a Low / Medium / High risk label, and render an interactive SHAP waterfall explanation. The separation keeps model development, governance evidence, and product serving independently testable.

## License

This project is released under the [MIT License](LICENSE). The Home Credit dataset remains subject to its original Kaggle terms and is not redistributed here.
