import json
from pathlib import Path

import joblib

from src.config import ROOT

ARTIFACTS_DIR = ROOT / "artifacts"


def export_model(model_dir: Path) -> None:
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        print(f"[SKIP] {model_dir.name} — model.joblib not found")
        return

    metrics_path = model_dir / "metrics.json"
    thresholds_path = model_dir / "thresholds.json"
    missing = [p.name for p in (metrics_path, thresholds_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{model_dir.name} is missing {missing}; retrain the model before exporting serving artifacts"
        )

    serving_model_path = model_dir / "serving_model.joblib"
    metadata_path = model_dir / "metadata.json"
    print(f"[Processing] {model_dir.name}...")

    wrapper = joblib.load(model_path)
    if not serving_model_path.exists():
        joblib.dump(wrapper.model, serving_model_path)
        print(f"  saved: {serving_model_path}")
    else:
        print(f"  kept: {serving_model_path}")

    metadata = {
        "model_type": model_dir.name.replace("_model", ""),
        "feature_names": wrapper.features_,
        "metrics": json.loads(metrics_path.read_text()),
        "thresholds": json.loads(thresholds_path.read_text()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"  saved: {metadata_path}")


def main() -> None:
    if not ARTIFACTS_DIR.exists():
        raise FileNotFoundError(f"Missing {ARTIFACTS_DIR}; run model training first")
    for model_dir in ARTIFACTS_DIR.iterdir():
        if model_dir.is_dir():
            export_model(model_dir)
    print("\nDone. Copy serving_model.joblib + metadata.json to the web service.")


if __name__ == "__main__":
    main()
