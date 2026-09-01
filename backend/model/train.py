from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "placement.csv"
MODEL_PATH = Path(__file__).resolve().parent / "placement_model.pkl"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "cgpa",
    "tenth_percentage",
    "twelfth_percentage",
    "internships",
    "projects",
    "aptitude_score",
    "communication_score",
    "coding_score",
    "backlogs",
    "certifications",
]
TARGET_COLUMN = "placed"


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the placement dataset."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    return df[required_columns]


def split_dataset(df: pd.DataFrame):
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_models() -> dict[str, Pipeline | RandomForestClassifier]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }


def print_metrics(model_name: str, metrics: dict[str, float]) -> None:
    print(f"\n{model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")


def train_and_select_model():
    df = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(df)

    best_model_name = ""
    best_model = None
    best_metrics = None

    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        print_metrics(model_name, metrics)

        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_model_name = model_name
            best_model = model
            best_metrics = metrics

    return best_model_name, best_model, best_metrics


def save_model(model, model_path: Path = MODEL_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
        },
        model_path,
    )


def main() -> None:
    best_model_name, best_model, best_metrics = train_and_select_model()
    save_model(best_model)

    print(f"\nBest model: {best_model_name}")
    print(f"Best F1: {best_metrics['f1']:.4f}")
    print(f"Model saved successfully: {MODEL_PATH}")


if __name__ == "__main__":
    main()
