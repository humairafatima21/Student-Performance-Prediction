"""Automated integrity and prediction tests. Run: python -m pytest -q"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = ["studytime", "failures", "absences", "G1", "G2", "age"]


def load_artifacts():
    model = joblib.load(ROOT / "student_performance_model.pkl")
    metrics = json.loads((ROOT / "model_metrics.json").read_text(encoding="utf-8"))
    return model, metrics


def test_required_files_exist():
    for name in ["app.py", "test.py", "predict.py", "student-mat.csv",
                 "student_performance_model.pkl", "model_metrics.json",
                 "model_comparison.csv", "actual_vs_predicted.png"]:
        assert (ROOT / name).is_file(), f"Missing required file: {name}"


def test_dataset_schema_and_rows():
    df = pd.read_csv(ROOT / "student-mat.csv", sep=";")
    assert len(df) == 395
    assert set(FEATURES + ["G3"]).issubset(df.columns)
    assert not df[FEATURES + ["G3"]].isna().any().any()
    assert df["G3"].between(0, 20).all()


def test_saved_model_feature_schema_matches_metrics():
    model, metrics = load_artifacts()
    assert metrics["features"] == FEATURES
    assert list(model.feature_names_in_) == FEATURES
    assert metrics["target"] == "G3"


def test_metrics_are_finite_and_reasonable():
    _, metrics = load_artifacts()
    for key in ["r2", "mae", "mse", "rmse"]:
        assert np.isfinite(metrics[key]), key
    assert 0 <= metrics["mae"] <= 20
    assert 0 <= metrics["rmse"] <= 20
    assert np.isclose(metrics["rmse"] ** 2, metrics["mse"], rtol=0.01)
    assert metrics["training_records"] + metrics["testing_records"] == 395


def test_prediction_is_finite_and_within_grade_range():
    model, _ = load_artifacts()
    examples = pd.DataFrame([
        [1, 0, 0, 0, 0, 15],
        [3, 0, 2, 15, 16, 16],
        [4, 4, 100, 20, 20, 22],
    ], columns=FEATURES)
    predictions = np.asarray(model.predict(examples), dtype=float)
    assert len(predictions) == 3
    assert np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 20)).all()


def test_model_comparison_contains_both_algorithms():
    comparison = pd.read_csv(ROOT / "model_comparison.csv")
    assert {"Linear Regression", "Random Forest Regression"}.issubset(set(comparison["model"]))
    assert comparison["cv_rmse_mean"].notna().all()


def test_history_file_is_valid_list():
    path = ROOT / "prediction_history.json"
    if path.exists():
        history = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(history, list)
        for record in history:
            assert isinstance(record, dict)
