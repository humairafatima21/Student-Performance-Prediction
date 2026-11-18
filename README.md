# Student Performance Prediction

A Python machine-learning project that estimates a student's final grade (G3, 0-20) from study time, previous failures, absences, first-period grade (G1), second-period grade (G2), and age. It uses the UCI Student Performance dataset and compares Linear Regression with Random Forest Regression.

## Project Files

* `app.py` - Streamlit dashboard and prediction interface.
* `test.py` - Trains and compares models using 5-fold cross-validation on the training split, evaluates them on a held-out test split, and writes the model, metrics, comparison CSV, and graph.
* `predict.py` - Validated command-line prediction tool.
* `test_project.py` - Automated project-integrity and model smoke tests.
* `student-mat.csv` - Dataset.
* `student_performance_model.pkl` - Selected trained model.
* `model_metrics.json` - Held-out test metrics and model metadata.
* `model_comparison.csv` - Model comparison results.
* `actual_vs_predicted.png` - Selected model's held-out predictions graph.
* `prediction_history.json` - Local prediction history created or updated by the app.

## Setup

Python 3.10-3.12 is recommended for broad package compatibility.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Application

Start the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

## Retrain and Evaluate the Model

From the project directory, run:

```bash
python test.py
```

This regenerates the model, metrics, model comparison, and actual-versus-predicted graph.

**Note:** Running this command retrains the model and may overwrite existing model artifacts.

## Run Tests

Run the automated test suite:

```bash
python -m pytest -v
```

## Command-Line Prediction

Start the command-line prediction tool:

```bash
python predict.py
```

Follow the prompts to enter the student's details.

## Evaluation Note

The target is a continuous final grade, so this is a regression task. R-squared (R2) is a regression metric, not classification accuracy.

The reported evaluation uses a fixed held-out test split with random state 42. Results describe that split and do not guarantee the same performance for every student or school.

G1 and G2 are previous-period grades and can be strong predictors of G3. Predictions made before those grades are available will not have the same inputs or reliability.

## Limitations

* The dataset contains 395 records from two Portuguese schools and may not generalize to other schools or educational systems.
* Prediction history is stored locally in `prediction_history.json`; it is not a database.
* Predictions are estimates and should not be used as the sole basis for consequential decisions about students.

