from pathlib import Path
import tempfile

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def _load_split(data_path):
    split = np.load(data_path)
    return split["X_train"], split["X_test"], split["y_train"], split["y_test"]


def preprocess_data():
    run_dir = Path(tempfile.mkdtemp(prefix="ml-quiz-"))
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, stratify=data.target, random_state=SEED
    )
    data_path = run_dir / "split.npz"
    np.savez(data_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print(
        f"Dataset: breast cancer, {data.data.shape[0]} samples, {data.data.shape[1]} features. "
        f"Train rows: {len(y_train)}, test rows: {len(y_test)}."
    )
    return str(data_path)


def train_baseline(data_path):
    X_train, _, y_train, _ = _load_split(data_path)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    model_path = Path(data_path).parent / "baseline.joblib"
    joblib.dump(model, model_path)
    print(f"Baseline (logistic regression) trained. Training accuracy: {model.score(X_train, y_train):.4f}")
    return str(model_path)


def train_candidate(data_path):
    X_train, _, y_train, _ = _load_split(data_path)
    model = RandomForestClassifier(n_estimators=200, random_state=SEED)
    model.fit(X_train, y_train)
    model_path = Path(data_path).parent / "candidate.joblib"
    joblib.dump(model, model_path)
    print(f"Candidate (random forest) trained. Training accuracy: {model.score(X_train, y_train):.4f}")
    return str(model_path)


def evaluate_models(baseline_path, candidate_path):
    _, X_test, _, y_test = _load_split(Path(baseline_path).parent / "split.npz")
    models = {
        "baseline": joblib.load(baseline_path),
        "candidate": joblib.load(candidate_path),
    }
    scores = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        scores[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }
        metrics = ", ".join(f"{metric}={value:.4f}" for metric, value in scores[name].items())
        print(f"{name}: {metrics}")
    best = max(scores, key=lambda name: scores[name]["roc_auc"])
    print(f"Best model by ROC AUC: {best}")
    print("Evaluation complete.")
