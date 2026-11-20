# Credit Scoring MLOps Pipeline

This repository contains a reproducible **credit-risk model training and evaluation pipeline** based on the **Home Credit Default Risk** dataset. It aggregates seven relational tables, engineers and selects 300 borrower-level features, trains four candidate models with Bayesian optimization, tracks experiments in MLflow, and produces SHAP explanations and model-risk guardrails.

<div align="center">
<img src="https://github.com/user-attachments/assets/504ad478-e1fe-4ab6-a9ac-82d41de85a78" width="250"><br>
</div>

---

## Project Status

- **Scope:** research and training engine for a credit-default scoring workflow; it is not a production lending decision system.
- **Reproducibility:** the Python environment, pipeline graph, parameters, holdout split, tests, and CI configuration are versioned. Raw Kaggle data and trained artifacts are intentionally excluded from Git.
- **Evaluation:** the holdout is scored once after all preprocessing, feature selection, HPO, early stopping, and threshold selection are fitted on the training partition. Run the pipeline locally to populate the current holdout metrics.
- **Responsible use:** fairness, calibration, and drift outputs are monitoring evidence and promotion gates—not proof that the model is suitable for real lending decisions.

## Brief Description

The project solves a binary classification problem — predicting the probability of borrower default, which is the likelihood that a client will delay payment by more than 90 days within a 12-month horizon. At the output, the model returns an estimate `f(x) ∈ [0, 1]`, which serves as the basis for the credit approval decision.

Key features:

- **Multi-module architecture** separating data loading, preprocessing, feature engineering, training, and interpretation.
- **Reproducible** (`uv.lock` + DVC + holdout): the environment is locked with uv, the pipeline is a DVC graph (`dvc.yaml`) driven by `params.yaml`, a 15% holdout split by `SK_ID_CURR` is never seen by any fit, stratified Out-of-Fold cross-validation is used for HPO and thresholds, and every run is tracked in MLflow.
- **Guarded model registry**: each trained model is registered with the `challenger` alias; `src/models/promote.py` moves `champion` only if holdout ROC-AUC and the fairness check pass (see the [model card](docs/model_card.md)).
- **Interpretability**: global and local SHAP explanations, avoiding "black box" logic.
- **Guardrails on the holdout**: Brier score and reliability table, fairness by gender and age (disparate impact, equal-opportunity gap), PSI drift of the score and top features. See [docs/model_card.md](docs/model_card.md) and [docs/datasheet.md](docs/datasheet.md).
- **Unified CLI**: launching the full training cycle of any model with a single command.
- **Research notes**: historical hypothesis results are retained for context and are explicitly labeled when they require rerunning on the current leakage-safe pipeline.

---

## Motivation

I wanted to build a complete ML pipeline: from raw data to versioned models in the registry, with reproducible training, auditable evaluation, and explainable outputs.

During this work, I gained experience in:

- designing **modular feature engineering** on top of a relational database of 7+ tables with transaction history aggregation into a single feature vector per borrower;
- controlling **data leakage** when preparing training and test sets;
- building a **unified OOP pipeline** based on an abstract base class, where any new algorithm is added with a single subclass;
- **Bayesian hyperparameter optimization** via Optuna;
- organizing **MLOps infrastructure**: experiment tracking, artifact logging, and model registry in MLflow;
- **model interpretation** via SHAP and formulating business explanations for credit refusal;
- working with **severe class imbalance**.

---

## Tech Stack

- **Python 3.10**, **uv** — dependency and environment management (`uv.lock`)
- **DVC** — data and pipeline versioning (`dvc.yaml`)
- **pandas, pyarrow, scipy, kaggle** — data, Parquet cache and statistics (`ydata-profiling` for EDA, notebooks group)
- **scikit-learn, LightGBM, XGBoost, CatBoost** — ML
- **optuna, optuna-dashboard** — hyperparameter optimization
- **mlflow** — experiment tracking
- **shap** — interpretability (XAI)
- **matplotlib, seaborn** — visualization
- **jupyterlab, nbconvert, quarto** — notebooks and reports (notebooks group)
- **ruff, pytest** — lint and tests (dev group)

---

## Repository Layout

```
src/data/        loader (CSV to Parquet), split (holdout), preprocessor
src/features/    per-table feature modules, pipeline.py, pipeline_base.py
src/models/      train.py (CLI), tuner.py (Optuna), promote.py, model wrappers
src/evaluation/  metrics, fairness, SHAP explainer, compare_models (legacy)
src/monitoring/  PSI drift
notebooks/       01_EDA, 04_interpretation
tests/           leakage, metrics, promotion gate, drift, smoke
docs/            datasheet.md, model_card.md
params.yaml      all parameters, seeds and costs
dvc.yaml         pipeline graph
```

---

## Project Features

#### Data Loading and Caching
Upon first access, CSV files are converted into binary Apache Parquet format, significantly speeding up subsequent iterations when working with tables containing millions of rows.

#### Leakage-Free Preprocessing
Encoders and grouping of rare categories into "Other" are fitted strictly on the training sample and applied to test without data leakage.

#### Modular Feature Engineering
Each data source is processed separately: financial ratios, behavioral indicators, and statistical aggregates are generated. The final pipeline merges them via left-join by borrower ID and selects the top 300 features using Ridge regression.

#### Unified Model Training
An abstract base class establishes a single contract upon which four models are implemented: Logistic Regression, LightGBM, XGBoost, and CatBoost. Adding a new algorithm simply requires writing one subclass.

#### Bayesian Hyperparameter Optimization
Optuna maximizes the mean ROC-AUC across folds; trial history is stored in `optuna.db` (SQLite). The study name embeds a hash of the data and config (`make_study_name`), so a search resumes only with the same data and configuration.

#### Experiment Tracking in MLflow
Nested run structure: parent Run → child HPO and Final_Production (for `--base` runs there is no `Final_Production` child: the metrics are logged on the parent run). Parameters, metrics, SHAP reports, and model dumps are logged. Every trained model is registered in the Model Registry with the alias `challenger`; the `champion` alias is set only by the promotion gate `src/models/promote.py` (holdout ROC-AUC not worse than the current champion by more than `gate.auc_epsilon`, and the minimum disparate-impact ratio at least `fairness.di_min`).

<div align="center">
<img src="https://github.com/user-attachments/assets/ac4c1d9d-8be4-4f1f-b32f-f6e79e03962c"><br>
<em>MLflow UI Dashboard</em>
<img src="https://github.com/user-attachments/assets/1a23353e-d1cd-4bd0-87d5-b5d163a72fb7"><br>
<em>Run List</em> 
<img src="https://github.com/user-attachments/assets/1e46a9c3-15c5-4463-9481-e72939f3ac04"><br>
<em>Artifact Logging</em> 
<img src="https://github.com/user-attachments/assets/b0ce9a32-1767-431c-be0e-22ca49c46228"><br>
<em>Model Registry</em> 
</div>


#### Decision Interpretation (XAI)
After training, SHAP values are automatically computed (TreeExplainer for boostings, LinearExplainer for LogReg). Summary Bar, Beeswarm, and local Waterfall charts are generated for specific clients.

<div align="center">
<img src="https://github.com/user-attachments/assets/829e180e-bab0-41cf-a42e-02f3684623cb" width="600"><br>
<em>Prediction Explanation for a High-Risk Client</em>
</div>

---

## Jupyter Notebooks

The notebooks in [notebooks/](notebooks/) are for exploration and analysis only: they read DVC outputs and `artifacts/` and reuse functions from `src/`. Training and promotion go through the `src/` CLI described below.

| Notebook | Content |
|---|---|
| [01_EDA.ipynb](notebooks/01_EDA.ipynb) | Data loading, HTML table profiling, analysis of imbalance, missing values, and anomalies |
| [04_interpretation.ipynb](notebooks/04_interpretation.ipynb) | Read-only analysis of trained models: selected features and correlations, global/local SHAP, cross-model comparison on one client |

---

## Getting Started

### 1. Dependency Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/); the environment is locked in `uv.lock` (Python 3.10).

```bash
# Clone this repository from GitHub, then run the commands below from its root.
cd credit-scoring-mlops

uv sync                      # core + dev dependencies
uv sync --group notebooks    # additionally JupyterLab, ydata-profiling, quarto, ...
```

Quality check (ruff + tests, no Kaggle data needed): `make check`.

### 2. Data Downloading

Data is retrieved from the Kaggle competition [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk). Place CSV files in `data/raw/` (requires a configured `kaggle` API token):

```bash
uv run kaggle competitions download -c home-credit-default-risk -p data/
unzip "data/*.zip" -d data/raw/
uv run dvc add data/raw      # track raw data with DVC (not in git)
```

See [docs/datasheet.md](docs/datasheet.md) for provenance, license terms, personal fields and known anomalies.

### 3. Pipeline: Split and Feature Engineering

The whole pipeline is a DVC graph (`uv run dvc dag`): `prepare` (CSV to Parquet cache) → `split` (15% holdout by `SK_ID_CURR`, seed from `params.yaml`) → `features` / `features_base` → `train@{logreg,lgbm,xgb,catboost}`. To run everything:

```bash
uv run dvc repro
```

> `dvc repro` also trains all four models (`train@foreach`, defaults: 30 trials, CPU), which takes a long time. To stop at the features: `uv run dvc repro features`.

Or stage by stage:

```bash
uv run python -m src.data.loader
uv run python -m src.data.split
uv run python -m src.features.pipeline
```

**Output** (in `data/processed/`): `holdout_ids.json`, `train_features.parquet`, `holdout_features.parquet`, `test_features.parquet`, `selected_features.json`, `feature_importance.csv`. Preprocessing, Ridge feature selection and encoders are fitted on the training part only. All parameters (seed, `top_n`, folds, costs, thresholds) are in `params.yaml`.

> For baseline comparison (main table only): `uv run python -m src.features.pipeline_base`.

### 4. Model Training

Full training cycle with a single command via CLI:

```bash
uv run python -m src.models.train --model lgbm --trials 30 --device gpu
```

Flags:

| Flag | Values | Default | Description |
|---|---|---|---|
| `--model` | `logreg` / `lgbm` / `xgb` / `catboost` | — (required) | Algorithm to train |
| `--trials` | int | `hpo.trials` (30) | Number of Optuna trials (`0` — skip HPO) |
| `--folds` | int | `cv.n_splits` (5) | Number of final CV folds |
| `--device` | `cpu` / `gpu` | `cpu` | Computing device |
| `--base` | flag | off | Use the single-table base feature set (`data/processed_base`); registers `credit_scoring_<model>_base` |
| `--params` | path | `params.yaml` | Alternative parameter file |

Can be run sequentially to train the entire suite of models:

```bash
uv run python -m src.models.train --model logreg   --trials 30
uv run python -m src.models.train --model lgbm     --trials 30 --device gpu
uv run python -m src.models.train --model xgb      --trials 30 --device gpu
uv run python -m src.models.train --model catboost --trials 30 --device gpu
```

**Output** (in `artifacts/<model>_model/`): `model.joblib` (full wrapper), `metrics.json`, `thresholds.json`, `xai/` (SHAP plots and `shap_feature_importance.csv`), `guardrails/` (reliability table and plot, per-group fairness CSVs) and `monitoring/` (PSI reports). The same outputs are logged to MLflow under the training run. The model is refit on the whole training part and scored on the holdout exactly once; holdout metrics, calibration, fairness and drift guardrails are logged with it.

`metrics.json` (and the same names in MLflow) contains:

- `final_oof_*` and `holdout_*`: `roc_auc`, `gini`, `ks_statistic`, `average_precision`, `brier`;
- `holdout_business_loss`, `holdout_fn`, `holdout_fp`, `holdout_approval_rate` (at the business threshold from OOF);
- fairness: `holdout_fairness_{gender,age}_{di_ratio,eo_gap}` and `holdout_fairness_min_di`;
- drift: `psi_score`, `psi_max_feature`, plus the `drift_alert` run tag (monitoring only, never fails training).

> Quick draft model without HPO on a single table: `uv run python -m src.models.train --model lgbm --base --trials 0` (needs `uv run python -m src.features.pipeline_base` first). `--base` artifacts go to `artifacts/<model>_base_model/`; this is not a DVC stage and the folder is git-ignored. Its metrics live on the parent MLflow run.

### 5. Promotion to Champion

```bash
uv run python -m src.models.promote --model lgbm          # add --base for the base feature set
```

The `challenger` version becomes `champion` only if its holdout ROC-AUC is at least the current champion's minus `gate.auc_epsilon` and `holdout_fairness_min_di` is at least `fairness.di_min`. Non-finite metrics fail closed; the command exits with code 1 on rejection.

### 6. Viewing Experiments in MLflow

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000 — model comparison by ROC-AUC, nested HPO/Final_Production runs, artifacts, and Model Registry.

### 7. Interpretation and Export

For deep SHAP analysis, use [04_interpretation.ipynb](notebooks/04_interpretation.ipynb) or run `make notebooks` to execute it (requires processed data and the `lgbm` model and at least one other trained model).

To export models for serving (e.g. LoanSight):

```bash
uv run python export_serving_artifacts.py
```

For each `artifacts/<model>_model/` with a `model.joblib`, it writes `serving_model.joblib` (the full scoring wrapper with feature alignment and calibration) and `metadata.json` containing `model_type`, `feature_names`, verified `metrics`, and decision `thresholds`. Export fails if `metrics.json` or `thresholds.json` is missing, preventing incomplete serving bundles.

---

## Tests and CI

```bash
make check    # ruff + pytest
make smoke    # only `-m smoke`: end-to-end training on synthetic data
```

Tests cover leakage, the promotion gate, drift (PSI), fairness, metric math, seed routing and the notebooks (`src` imports, cleared outputs). CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs `uv sync --locked --group dev` and `make check` on every push and pull request.

---

## Hypotheses

During the study, five hypotheses were formulated and statistically tested, covering data nature, feature importance, and training method efficiency.

> **Reproducibility.** H1 and H2: the code that produced these figures was deleted together with the old notebooks (commit `144b110`); the repository currently contains no Spearman or Mann–Whitney computation, so they cannot be re-run. H3 can be checked from `artifacts/<model>_model/xai/shap_feature_importance.csv`. H4 and H5 were computed by `src/evaluation/compare_models.py` from `cv_mean_roc_auc`; it is legacy: it looks for `*_non_hpo` runs, which the current `train.py` does not create (`--trials 0` creates a run named `<model>`), so it reproduces neither H4 nor H5 as is.

---

#### Hypothesis 1 — Age as a Risk Predictor

**Statement:** At the level of aggregates, default risk decreases monotonically with borrower age: across ten age deciles, the younger the decile, the higher its default rate. This is a statement about groups of borrowers, not about any individual borrower.

**Testing Method:** The training sample is split into 10 equal deciles based on the `DAYS_BIRTH` feature; the default rate is calculated in each decile; monotonicity is evaluated using Spearman's rank correlation between the decile number and its default rate. With 10 points the exact permutation p-value of a perfectly monotone sequence (ρ = −1) is 1/10! ≈ 2.8·10⁻⁷ (one-sided); the earlier "p ≈ 0" was the asymptotic approximation.

**Acceptance Criterion:** ρ < −0.7 and p-value < 0.05.

**Result: Confirmed.** ρ = −1.00, exact permutation p = 1/10! ≈ 2.8·10⁻⁷. Default rate is 12.3% for the youngest vs. 3.73% for the oldest decile, with zero violations of monotonicity. These figures are from the original notebook run (pre-audit, before the holdout split) and have not been recomputed here. Ten points support a conclusion about the age trend of the population, not a prediction for a specific client; age is also a protected attribute, so its use is audited (see the [model card](docs/model_card.md)).

<div align="center">
<img src="https://github.com/user-attachments/assets/e8f7ee0c-d518-43de-a95e-62a07d75ab83" width="600"><br>
</div>

---

#### Hypothesis 2 — Missing Values as a Risk Signal

**Statement:** Defaulted clients have a significantly higher proportion of missing values in their application form than reliable clients — meaning missing values are not random and serve as a credit risk signal in themselves.

**Testing Method:** For each row in `application_train`, the proportion of missing values is calculated; default and reliable client groups are compared using the non-parametric Mann–Whitney U test.

**Acceptance Criterion:** p-value < 0.05 and the median proportion of missing values is higher for defaulted clients than for reliable ones.

**Result: Confirmed.** Median is 39.7% for defaulted vs. 28.1% for reliable clients; Mann–Whitney test p ≈ 0. Missing values are an informative predictor and are exposed to the models as binary `EXT_SOURCE_1/2/3_WAS_MISSING` flags (`src/features/application.py`); other missing values stay NaN for the tree models and are median-imputed for logistic regression.

---

#### Hypothesis 3 — External Credit Scores as Top Predictors

**Statement:** Features `EXT_SOURCE_1/2/3` are the most informative predictors, as they aggregate ready-made credit history from external sources.

**Testing Method:** Evaluation of global feature importance via SHAP (`shap.TreeExplainer`); checking whether `EXT_SOURCE_1/2/3` or their derivatives are in the top 5.

**Acceptance Criterion:** `EXT_SOURCE_1/2/3` or their derivatives appear in the top 5 SHAP feature importances of the final model.

**Result: Confirmed.** The derived feature `EXT_SOURCES_MEAN` (mean of the three sources) ranked **1st** in the SHAP importance ranking.

<div align="center">
<img src="https://github.com/user-attachments/assets/28f71ec9-a480-4446-8b30-efdcdd1a1841" width="500"><br>
<em>SHAP Summary Bar</em>
</div>

---

#### Hypothesis 4 — Value of Auxiliary Tables

**Statement:** Using data from auxiliary tables (`bureau`, `previous_application`, etc.) significantly improves prediction quality compared to using only the main application form — by capturing the temporal dynamics of client behavior.

**Testing Method:** Comparison of ROC-AUC for two LightGBM models: trained on the full set of 7 tables (`lgbm_non_hpo`) and trained only on `application_train` (`lgbm_base`).

**Acceptance Criterion:** ROC-AUC gain ≥ 0.015.

**Result: Confirmed.** ROC-AUC (OOF, before the leakage fix F1) increased from 0.766 (`lgbm_base`) to 0.786 (`lgbm_non_hpo`), a gain of **+0.020** — exceeding the threshold. Aggregated payment history contains critically important signal invisible in a static application form.

---

#### Hypothesis 5 — Efficiency of Bayesian Hyperparameter Optimization

**Statement:** Automated hyperparameter tuning via Optuna outperforms default parameters due to data specificity.

**Testing Method:** Comparison of ROC-AUC between `non_hpo` (default parameters) and `full` (30 Optuna trials) models on the complete dataset for each algorithm.

**Acceptance Criterion:** ROC-AUC gain ≥ 0.005 for at least one algorithm; the hypothesis is confirmed in full only if the gain is consistent across algorithms.

**Result: Partially Confirmed.** The gain is not consistent: XGBoost gains +0.021, CatBoost only reaches the threshold (+0.005, rounded), and LightGBM gains nothing (0.000). The criterion is therefore met for one algorithm clearly, but tuning does not help every algorithm (OOF, before the leakage fix F1):

| Algorithm | ROC-AUC (Non-HPO), OOF pre-F1 | ROC-AUC (Full), OOF pre-F1 | Gain |
|---|---|---|---|
| XGBoost | 0.767 | 0.788 | **+0.021** |
| CatBoost | 0.782 | 0.787 | **+0.005** |
| LightGBM | 0.786 | 0.786 | 0.000 |

---

## Benchmark Results and Evaluation Notes

> **Important:** The benchmark table below is historical OOF output from before the leakage fix (F1). It is retained for reproducibility context only and must not be presented as current holdout or production performance. Run `uv run dvc repro` with the Kaggle data to generate the current, leakage-safe holdout results described in [docs/model_card.md](docs/model_card.md).

In this historical OOF run on the Home Credit sample, the ensemble models ranked above the linear baseline. The best historical result was **`xgb_full`** (ROC-AUC = 0.788); it is not a current holdout or production estimate.

| Algorithm | Version | ROC-AUC (OOF, pre-F1) | PR-AUC (OOF, pre-F1) | Gini (OOF, pre-F1) | Business Loss (OOF, pre-F1) |
|---|---|---|---|---|---|
| **XGBoost** | Full | **0.788** | **0.281** | **0.576** | **149,582** |
| CatBoost | Full | 0.787 | 0.278 | 0.573 | 150,005 |
| LightGBM | Full | 0.786 | 0.278 | 0.571 | 150,627 |
| LightGBM | Non-HPO | 0.786 | 0.277 | 0.572 | 150,029 |
| LogReg | Full | 0.773 | 0.253 | 0.546 | 155,780 |
| LogReg | Base *(baseline)* | 0.751 | 0.235 | 0.502 | 165,541 |

<div align="center">
<img src="https://github.com/user-attachments/assets/ca246b5c-4787-44d0-801d-d5f5402801af" width="800"><br>
<em>Comparison of cv_mean_roc_auc across all models (OOF, before the leakage fix F1)</em>
</div>

**Historical observations:**

- **Data enrichment matters more than algorithm complexity.** Transitioning from one table to seven yielded a ROC-AUC gain of +0.020, whereas transitioning from LogReg to XGBoost on full data yielded only +0.015.
- **PR-AUC is more sensitive than ROC-AUC** under class imbalance: PR-AUC spans 0.235–0.281 across the table while ROC-AUC spans 0.751–0.788, and PR-AUC reacts more strongly to the quality of the rare positive class.
- 4 of 5 hypotheses were confirmed, 1 partially (H5: hyperparameter optimization helps XGBoost and CatBoost, but not LightGBM).

---

## Limitations and Next Improvements

- **Stacking Ensemble**: meta-model (LogReg) on top of predictions from CatBoost + LightGBM + XGBoost to maximize ROC-AUC.
- **Production Monitoring**: PSI of the score and top features against the Kaggle test sample is logged, but evaluation on fresh labelled cohorts and automatic rollback are not implemented in this repository.
- **Current evaluation artifacts**: raw data and trained artifacts are not committed, so current holdout, fairness, calibration, and drift values must be generated locally rather than copied from historical OOF output.
- **Deployment boundary**: the exported models are consumed by the separate LoanSight application; this repository does not provide an API, authentication, feature store, or lender-specific policy engine.
- **Legacy comparison script**: update `src/evaluation/compare_models.py` to the current run names.
- **Serving integration**: add schema validation and an API-level contract in the LoanSight service before accepting production traffic.
- **Temporal validation**: add a time-based holdout when a trustworthy application-date field is available; the current Home Credit feature set does not expose one in this repository.
- **Score scaling**: add bank-standard score scaling only after calibration has been validated on a representative labelled cohort.

---

## Connection to the LoanSight Project

Models trained in this repository formed the core of **[LoanSight](https://github.com/mikhailvokhrameev/loan_sight.git)** — a web service for experimenting with credit risk assessment ML models.

LoanSight loads artifacts exported here, retrieves client data from a database, runs the selected model, and returns default probability along with a human-readable risk label (Low / Medium / High) and an interactive **SHAP waterfall chart**. The service supports side-by-side comparison of two models, manual feature overriding ("what-if" analysis), multi-currency, and experiment history.

Thus, **this repository is the research and training engine**, while LoanSight serves as the product frontend wrapper around the resulting models.
#   C r e d i t - S c o r i n g - M l o p s  
 #   C r e d i t - S c o r i n g - M l o p s  
 #   C r e d i t - S c o r i n g - M l o p s  
 