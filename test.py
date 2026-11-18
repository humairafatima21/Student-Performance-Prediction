"""
Student Performance Prediction: model comparison and training.

Compares Linear Regression and Random Forest Regression using 5-fold
cross-validation on the training split, then evaluates the selected model
once on the held-out test split.

Run from the project folder:
    python test.py
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student-mat.csv"
MODEL_FILE = BASE_DIR / "student_performance_model.pkl"
GRAPH_FILE = BASE_DIR / "actual_vs_predicted.png"
METRICS_FILE = BASE_DIR / "model_metrics.json"
COMPARISON_FILE = BASE_DIR / "model_comparison.csv"

FEATURES = ["studytime", "failures", "absences", "G1", "G2", "age"]
TARGET = "G3"
RANDOM_STATE = 42


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Place student-mat.csv in the same folder as test.py."
        )

    df = pd.read_csv(DATA_FILE, sep=";")
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    model_df = df[required_columns].copy()
    for column in required_columns:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")
    original_count = len(model_df)
    model_df = model_df.dropna()
    removed_count = original_count - len(model_df)

    if len(model_df) < 25:
        raise ValueError("Not enough valid records for a reliable train/test comparison.")

    X = model_df[FEATURES]
    y = model_df[TARGET]

    print("\n=== DATASET SUMMARY ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Valid records used: {len(model_df)}")
    if removed_count:
        print(f"Rows removed due to missing/non-numeric selected values: {removed_count}")
    print(f"Features: {FEATURES}")
    print(f"Target: {TARGET} (final grade out of 20)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # Select using cross-validation on training data only. The held-out test
    # set remains untouched until final evaluation.
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison_rows = []
    fitted_candidates = {}

    print("\n=== 5-FOLD CROSS-VALIDATION (TRAINING SPLIT ONLY) ===")
    for name, estimator in candidates.items():
        scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=None,
        )
        cv_rmse = -scores
        estimator.fit(X_train, y_train)
        fitted_candidates[name] = estimator
        comparison_rows.append(
            {
                "model": name,
                "cv_rmse_mean": float(cv_rmse.mean()),
                "cv_rmse_std": float(cv_rmse.std()),
            }
        )
        print(
            f"{name}: mean CV RMSE = {cv_rmse.mean():.3f} "
            f"(standard deviation {cv_rmse.std():.3f})"
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        "cv_rmse_mean", ascending=True
    )
    selected_name = str(comparison_df.iloc[0]["model"])
    selected_model = fitted_candidates[selected_name]

    # Evaluate the selected model on the held-out test split once.
    test_predictions = selected_model.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, test_predictions))
    test_mse = float(mean_squared_error(y_test, test_predictions))
    test_rmse = float(np.sqrt(test_mse))
    test_r2 = float(r2_score(y_test, test_predictions))

    # Include held-out metrics for both models for transparent comparison.
    # The selected model is still chosen only by training-set CV RMSE.
    test_rows = []
    for name, estimator in fitted_candidates.items():
        pred = estimator.predict(X_test)
        test_rows.append(
            {
                "model": name,
                "cv_rmse_mean": float(
                    comparison_df.loc[comparison_df["model"] == name, "cv_rmse_mean"].iloc[0]
                ),
                "test_mae": float(mean_absolute_error(y_test, pred)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "test_r2": float(r2_score(y_test, pred)),
            }
        )
    full_comparison = pd.DataFrame(test_rows).sort_values("cv_rmse_mean")
    full_comparison.to_csv(COMPARISON_FILE, index=False)

    print("\n=== MODEL COMPARISON ===")
    print(full_comparison.round(3).to_string(index=False))
    print(f"\nSelected model (lowest mean CV RMSE): {selected_name}")
    print("\n=== SELECTED MODEL: HELD-OUT TEST RESULTS ===")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"MAE:  {test_mae:.2f} grade points")
    print(f"MSE:  {test_mse:.2f}")
    print(f"RMSE: {test_rmse:.2f} grade points")
    print(f"R²:   {test_r2:.3f}")
    print("Note: R² is a regression metric, not classification accuracy.")

    # Save actual-versus-predicted graph for selected model.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, test_predictions, alpha=0.7)
    ax.plot([0, 20], [0, 20], linestyle="--")
    ax.set_xlabel("Actual Final Grade")
    ax.set_ylabel("Predicted Final Grade")
    ax.set_title(f"Actual vs Predicted Grades — {selected_name}")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(GRAPH_FILE, dpi=150)
    plt.close(fig)

    # Persist selected estimator and metrics for the Streamlit app.
    joblib.dump(selected_model, MODEL_FILE)
    metrics = {
        "model_name": selected_name,
        "selection_method": "5-fold cross-validation on training split, lowest mean RMSE",
        "r2": test_r2,
        "mae": test_mae,
        "mse": test_mse,
        "rmse": test_rmse,
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "cv_results": comparison_rows,
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    # Example prediction using the same feature order as training.
    example_student = pd.DataFrame(
        [{
            "studytime": 3,
            "failures": 0,
            "absences": 2,
            "G1": 15,
            "G2": 16,
            "age": 16,
        }],
        columns=FEATURES,
    )
    example_prediction = float(selected_model.predict(example_student)[0])
    example_prediction = max(0.0, min(20.0, example_prediction))

    print("\n=== SAVED FILES ===")
    print(f"Model: {MODEL_FILE}")
    print(f"Metrics: {METRICS_FILE}")
    print(f"Graph: {GRAPH_FILE}")
    print(f"Comparison table: {COMPARISON_FILE}")
    print("\n=== EXAMPLE PREDICTION ===")
    print(example_student.to_string(index=False))
    print(f"Predicted final grade: {example_prediction:.2f} / 20")
    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    main()
