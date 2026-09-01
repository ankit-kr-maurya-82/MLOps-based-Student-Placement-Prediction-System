from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / "placement_model.pkl"
DEFAULT_FEATURE_COLUMNS = [
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


@lru_cache(maxsize=1)
def load_model() -> tuple[Any, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run backend/model/train.py first."
        )

    model_bundle = joblib.load(MODEL_PATH)

    if isinstance(model_bundle, dict) and "model" in model_bundle:
        model = model_bundle["model"]
        feature_columns = model_bundle.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    else:
        model = model_bundle
        feature_columns = DEFAULT_FEATURE_COLUMNS

    return model, list(feature_columns)


def validate_student_data(
    student_data: dict[str, Any], feature_columns: list[str]
) -> dict[str, float]:
    missing_features = [
        feature for feature in feature_columns if feature not in student_data
    ]

    if missing_features:
        raise ValueError(f"Missing required features: {', '.join(missing_features)}")

    validated_data = {}
    for feature in feature_columns:
        try:
            validated_data[feature] = float(student_data[feature])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{feature}' must be numeric.") from exc

    return validated_data


def predict_placement(student_data: dict[str, Any]) -> dict[str, float | str]:
    model, feature_columns = load_model()
    validated_data = validate_student_data(student_data, feature_columns)
    input_df = pd.DataFrame([validated_data], columns=feature_columns)

    predicted_class = int(model.predict(input_df)[0])
    probability = get_placement_probability(model, input_df, predicted_class)

    return {
        "prediction": "Placed" if predicted_class == 1 else "Not Placed",
        "probability": round(probability, 4),
    }


def get_placement_probability(model: Any, input_df: pd.DataFrame, predicted_class: int) -> float:
    if hasattr(model, "predict_proba"):
        class_labels = list(model.classes_)
        placed_index = class_labels.index(1)
        return float(model.predict_proba(input_df)[0][placed_index])

    return 1.0 if predicted_class == 1 else 0.0
