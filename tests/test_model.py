import pytest

from backend.model.predict import predict_placement, validate_student_data
from tests.test_api import VALID_PAYLOAD


def test_model_prediction_format():
    result = predict_placement(VALID_PAYLOAD)

    assert set(result) == {"prediction", "probability"}
    assert result["prediction"] in {"Placed", "Not Placed"}
    assert isinstance(result["probability"], float)


def test_model_probability_between_zero_and_one():
    result = predict_placement(VALID_PAYLOAD)

    assert 0 <= result["probability"] <= 1


def test_model_validation_rejects_missing_feature():
    payload = VALID_PAYLOAD.copy()
    payload.pop("cgpa")

    with pytest.raises(ValueError, match="Missing required features: cgpa"):
        validate_student_data(payload, list(VALID_PAYLOAD))


def test_model_validation_rejects_non_numeric_feature():
    payload = VALID_PAYLOAD | {"cgpa": "excellent"}

    with pytest.raises(ValueError, match="Feature 'cgpa' must be numeric"):
        validate_student_data(payload, list(VALID_PAYLOAD))
