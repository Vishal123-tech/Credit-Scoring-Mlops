from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent

BENCHMARK = pd.DataFrame(
    [
        {"Algorithm": "XGBoost", "Version": "Full", "ROC-AUC": 0.788, "PR-AUC": 0.281, "Gini": 0.576, "Business loss": 149582},
        {"Algorithm": "CatBoost", "Version": "Full", "ROC-AUC": 0.787, "PR-AUC": 0.278, "Gini": 0.573, "Business loss": 150005},
        {"Algorithm": "LightGBM", "Version": "Full", "ROC-AUC": 0.786, "PR-AUC": 0.278, "Gini": 0.571, "Business loss": 150627},
        {"Algorithm": "LightGBM", "Version": "Non-HPO", "ROC-AUC": 0.786, "PR-AUC": 0.277, "Gini": 0.572, "Business loss": 150029},
        {"Algorithm": "LogReg", "Version": "Full", "ROC-AUC": 0.773, "PR-AUC": 0.253, "Gini": 0.546, "Business loss": 155780},
        {"Algorithm": "LogReg", "Version": "Base", "ROC-AUC": 0.751, "PR-AUC": 0.235, "Gini": 0.502, "Business loss": 165541},
    ]
)

GOVERNANCE_CHECKS = [
    (
        "Leakage control",
        "15% borrower-level holdout; preprocessing, feature selection, HPO, calibration and thresholds stay on training data.",
    ),
    (
        "Probability calibration",
        "Sigmoid calibration is fitted on out-of-fold predictions before holdout scoring and serving export.",
    ),
    (
        "Fairness audit",
        "Disparate impact and equal-opportunity gaps are reported by gender and age while protected fields are excluded from model inputs.",
    ),
    (
        "Drift monitoring",
        "PSI is calculated for the model score and top features and logged as monitoring evidence.",
    ),
    (
        "Promotion gate",
        "A challenger is promoted only when configured holdout ROC-AUC and fairness thresholds pass.",
    ),
    (
        "Explainability",
        "Global and local SHAP outputs support model review and borrower-level explanation.",
    ),
]


st.set_page_config(
    page_title="Credit Scoring MLOps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1320px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }
        .hero {
            padding: 2.2rem 2.4rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #111827 0%, #1e3a8a 58%, #0f766e 100%);
            color: #f8fafc;
            margin-bottom: 1.4rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
        }
        .hero h1 {
            font-size: 2.65rem;
            margin: 0 0 0.5rem 0;
            letter-spacing: -0.04em;
        }
        .hero p {
            font-size: 1.08rem;
            max-width: 850px;
            margin: 0;
            color: #dbeafe;
        }
        .eyebrow {
            color: #93c5fd;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }
        .section-note {
            color: #64748b;
            margin-top: -0.55rem;
            margin-bottom: 1.1rem;
        }
        .status-card {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: rgba(15, 23, 42, 0.035);
            min-height: 125px;
        }
        .status-card strong {
            display: block;
            margin-bottom: 0.35rem;
        }
        .status-pill {
            display: inline-block;
            color: #047857;
            background: #d1fae5;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.72rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .callout {
            border-left: 4px solid #2563eb;
            background: #eff6ff;
            padding: 0.85rem 1rem;
            border-radius: 0 12px 12px 0;
            color: #1e3a8a;
            margin: 1rem 0 1.25rem 0;
        }
        .footer {
            color: #64748b;
            font-size: 0.84rem;
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_heading(title: str, note: str) -> None:
    st.markdown(f"## {title}")
    st.markdown(f"<p class='section-note'>{note}</p>", unsafe_allow_html=True)


def status_card(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="status-card">
            <span class="status-pill">Implemented in repository</span>
            <strong>{title}</strong>
            <span>{description}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("## Credit Scoring MLOps")
    st.caption("Streamlit project showcase")
    st.markdown("---")
    page = st.radio(
        "Explore",
        ["Overview", "Model benchmark", "Governance", "Scoring readiness", "Run locally"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### Deployment status")
    artifact_count = len(list((ROOT / "artifacts").glob("*/serving_model.joblib"))) if (ROOT / "artifacts").exists() else 0
    if artifact_count:
        st.success(f"{artifact_count} serving artifact(s) detected")
    else:
        st.info("Showcase mode")
        st.caption("Training artifacts are intentionally excluded from Git.")

    st.markdown("---")
    st.markdown(
        "[View source on GitHub](https://github.com/Vishal123-tech/Credit-Scoring-Mlops)"
    )


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Credit risk • reproducibility • model governance</div>
        <h1>Credit Scoring MLOps</h1>
        <p>
            A leakage-aware machine-learning workflow that turns relational credit data
            into explainable, calibrated and monitorable default-risk models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Overview":
    section_heading(
        "A complete model-development workflow",
        "The app summarizes the repository without presenting historical research numbers as live production performance.",
    )

    st.markdown(
        """
        This project demonstrates how a credit-default model can be built with
        reproducible data preparation, modular feature engineering, controlled evaluation,
        explainability, fairness checks, drift monitoring, and model-registry promotion gates.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Relational source tables", "7", help="Home Credit source tables used by the feature pipeline.")
    c2.metric("Candidate model families", "4", help="Logistic Regression, LightGBM, XGBoost and CatBoost.")
    c3.metric("Configured top features", "300", help="Feature-selection target from params.yaml.")
    c4.metric("Historical best ROC-AUC", "0.788", help="Historical out-of-fold benchmark only; not a current holdout result.")

    st.markdown(
        """
        <div class="callout">
            <strong>Responsible-use boundary:</strong>
            raw data and trained artifacts are not committed to the repository.
            Current holdout metrics must be generated locally after downloading the Kaggle dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("### Historical model comparison")
        chart_data = BENCHMARK.sort_values("ROC-AUC")
        fig = px.bar(
            chart_data,
            x="ROC-AUC",
            y="Algorithm",
            color="Version",
            orientation="h",
            text="ROC-AUC",
            color_discrete_sequence=["#2563eb", "#14b8a6", "#f59e0b"],
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=20, t=20, b=10),
            xaxis_title="ROC-AUC (historical OOF)",
            yaxis_title="",
            legend_title="",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Pipeline flow")
        st.markdown(
            """
            1. **Prepare** raw Home Credit tables and cache them as Parquet.
            2. **Split** a borrower-level holdout before fitting transformations.
            3. **Engineer** modular behavioral and financial features.
            4. **Train** four candidate model families with optional Optuna HPO.
            5. **Calibrate** probabilities and select thresholds from OOF predictions.
            6. **Evaluate** holdout performance, fairness, drift and explanations.
            7. **Track and promote** models through MLflow and configured gates.
            """
        )

elif page == "Model benchmark":
    section_heading(
        "Model benchmark",
        "Historical out-of-fold comparison retained for context; run the DVC pipeline to generate current holdout results.",
    )

    best = BENCHMARK.loc[BENCHMARK["ROC-AUC"].idxmax()]
    b1, b2, b3 = st.columns(3)
    b1.metric("Best historical model", f"{best['Algorithm']} {best['Version']}")
    b2.metric("Historical ROC-AUC", f"{best['ROC-AUC']:.3f}")
    b3.metric("Historical PR-AUC", f"{best['PR-AUC']:.3f}")

    display_data = BENCHMARK.copy()
    display_data["ROC-AUC"] = display_data["ROC-AUC"].map(lambda value: f"{value:.3f}")
    display_data["PR-AUC"] = display_data["PR-AUC"].map(lambda value: f"{value:.3f}")
    display_data["Gini"] = display_data["Gini"].map(lambda value: f"{value:.3f}")
    display_data["Business loss"] = display_data["Business loss"].map(lambda value: f"{value:,.0f}")
    st.dataframe(display_data, use_container_width=True, hide_index=True)

    csv = BENCHMARK.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download benchmark CSV",
        data=csv,
        file_name="historical_model_benchmark.csv",
        mime="text/csv",
    )

    st.markdown(
        """
        <div class="callout">
            These values are not a live scoring result. They are historical OOF figures from
            before leakage fix F1 and are included to make the research context inspectable.
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Governance":
    section_heading(
        "Model governance by design",
        "These are implemented code paths and documented controls—not a claim that a production lending model has been approved.",
    )

    for row in GOVERNANCE_CHECKS:
        left, right = st.columns([0.32, 0.68])
        with left:
            status_card(row[0], row[1])
        with right:
            st.empty()
        st.markdown("")

    st.markdown("### Evaluation boundary")
    st.info(
        "Protected attributes are available for audit outputs but excluded from model inputs. "
        "A representative labelled cohort is still required to validate calibration, fairness and threshold behavior."
    )

    with st.expander("Read the governance documentation"):
        st.markdown(
            """
            - [Model card](https://github.com/Vishal123-tech/Credit-Scoring-Mlops/blob/main/docs/model_card.md)
            - [Dataset datasheet](https://github.com/Vishal123-tech/Credit-Scoring-Mlops/blob/main/docs/datasheet.md)
            - [Configuration and gates](https://github.com/Vishal123-tech/Credit-Scoring-Mlops/blob/main/params.yaml)
            """
        )

elif page == "Scoring readiness":
    section_heading(
        "Scoring readiness",
        "The repository can export serving bundles after training, but this public deployment does not ship a borrower-scoring model.",
    )

    serving_models = sorted((ROOT / "artifacts").glob("*/serving_model.joblib")) if (ROOT / "artifacts").exists() else []
    if serving_models:
        st.success(f"{len(serving_models)} serving bundle(s) found in this deployment.")
        st.write("Available bundles:")
        for model_path in serving_models:
            st.code(str(model_path.relative_to(ROOT)))
        st.info(
            "For production-style scoring, use the exported bundle with the LoanSight service "
            "or add a reviewed inference layer with schema validation and authentication."
        )
    else:
        st.warning("No serving model is included in the GitHub deployment.")
        st.markdown(
            """
            This is intentional: the repository excludes raw data and trained artifacts.
            To generate a local serving bundle:

            1. Download the Home Credit data.
            2. Run the DVC feature and training stages.
            3. Export the verified model and metadata.
            4. Integrate the bundle into a reviewed application such as LoanSight.
            """
        )
        st.code(
            "uv run dvc repro\n"
            "uv run python export_serving_artifacts.py",
            language="bash",
        )

elif page == "Run locally":
    section_heading(
        "Run the full project locally",
        "Use the repository CLI for reproducible training; use this Streamlit page as a visual project overview.",
    )

    st.markdown("### Install and launch")
    st.code(
        "uv sync\n"
        "uv run streamlit run app.py",
        language="bash",
    )

    st.markdown("### Reproduce the modeling workflow")
    st.code(
        "uv run dvc repro\n"
        "uv run mlflow ui --backend-store-uri sqlite:///mlflow.db",
        language="bash",
    )

    st.markdown("### Train one model directly")
    st.code(
        "uv run python -m src.models.train --model lgbm --trials 30",
        language="bash",
    )

    st.markdown("### Project links")
    st.markdown(
        """
        - [GitHub repository](https://github.com/Vishal123-tech/Credit-Scoring-Mlops)
        - [Model card](https://github.com/Vishal123-tech/Credit-Scoring-Mlops/blob/main/docs/model_card.md)
        - [Dataset datasheet](https://github.com/Vishal123-tech/Credit-Scoring-Mlops/blob/main/docs/datasheet.md)
        - [LoanSight companion application](https://github.com/mikhailvokhrameev/loan_sight.git)
        """
    )

st.markdown(
    """
    <div class="footer">
        Built as a reproducible credit-risk research workflow. Historical benchmark values are clearly labeled;
        live lending decisions require independent validation and governance.
    </div>
    """,
    unsafe_allow_html=True,
)
