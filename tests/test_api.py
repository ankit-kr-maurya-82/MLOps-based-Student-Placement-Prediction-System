from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


VALID_PAYLOAD = {
    "cgpa": 8.4,
    "tenth_percentage": 82,
    "twelfth_percentage": 79,
    "internships": 2,
    "projects": 4,
    "aptitude_score": 85,
    "communication_score": 78,
    "coding_score": 88,
    "backlogs": 0,
    "certifications": 3,
}


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Placement Prediction API is running"}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_with_valid_values():
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()

    assert response.status_code == 200
    assert body["prediction"] in {"Placed", "Not Placed"}
    assert 0 <= body["probability"] <= 1


def test_predict_endpoint_with_invalid_values():
    payload = VALID_PAYLOAD | {"cgpa": 12}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
