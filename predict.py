"""Command-line prediction tool for Student Performance Prediction."""
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "student_performance_model.pkl"
FEATURES = ["studytime", "failures", "absences", "G1", "G2", "age"]


def read_int(prompt, minimum, maximum=None):
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if value < minimum or (maximum is not None and value > maximum):
            limit = f" and at most {maximum}" if maximum is not None else ""
            print(f"Value must be at least {minimum}{limit}.")
            continue
        return value


def read_grade(prompt):
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number from 0 to 20.")
            continue
        if not 0 <= value <= 20:
            print("Grade must be between 0 and 20.")
            continue
        return value


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}. Run 'python test.py' first."
        )
    model = joblib.load(MODEL_FILE)

    print("\n===== STUDENT PERFORMANCE PREDICTION =====")
    studytime = read_int("Weekly study time category (1-4): ", 1, 4)
    failures = read_int("Previous class failures (0-4): ", 0, 4)
    absences = read_int("School absences (0-100): ", 0, 100)
    g1 = read_grade("First period grade (0-20): ")
    g2 = read_grade("Second period grade (0-20): ")
    age = read_int("Student age (15-22): ", 15, 22)

    values = {
        "studytime": studytime, "failures": failures, "absences": absences,
        "G1": g1, "G2": g2, "age": age,
    }
    feature_order = list(getattr(model, "feature_names_in_", FEATURES))
    missing = [name for name in feature_order if name not in values]
    if missing:
        raise ValueError(f"Model requires unsupported input features: {missing}")
    student = pd.DataFrame([[values[name] for name in feature_order]], columns=feature_order)
    prediction = float(model.predict(student)[0])
    prediction = max(0.0, min(20.0, prediction))

    print("\n===== PREDICTION RESULT =====")
    print(f"Predicted final grade: {prediction:.2f} / 20")
    print(f"Predicted percentage: {prediction / 20 * 100:.1f}%")
    if prediction >= 16:
        category = "High predicted performance"
    elif prediction >= 10:
        category = "Moderate predicted performance"
    else:
        category = "Low predicted performance"
    print(category)
    print("Note: This is an estimate, not a guaranteed academic outcome.")


if __name__ == "__main__":
    main()
