# Student Placement Prediction System

An MLOps-focused machine learning project for predicting student placement outcomes from academic, skill, and experience data.

## Planned Stack

- Python 3.12
- FastAPI
- Pandas
- NumPy
- Scikit-learn
- Joblib
- MLflow
- Pytest
- HTML, CSS, and vanilla JavaScript
- Docker

## Project Structure

```text
backend/
  __init__.py
  main.py
  data/
    placement.csv
  model/
    __init__.py
    train.py
    predict.py

frontend/
  index.html
  style.css
  script.js

tests/
  __init__.py
  test_api.py
  test_model.py

.github/
  workflows/
    ci.yml

requirements.txt
Dockerfile
docker-compose.yml
.dockerignore
.gitignore
README.md
```

## Phase Status

- Phase 1: Project scaffold - complete
- Phase 2: Synthetic dataset - complete
- Phase 3: Model training - complete
- Phase 4: Prediction service - complete
- Phase 5: FastAPI backend - complete
- Phase 6: Frontend - complete
- Phase 7: Docker - complete
- Phase 8: MLflow - complete
- Phase 9: Tests - complete
- Phase 10: GitHub Actions CI - complete

## Tests

Run the local test suite from the project root:

```bash
python -m pytest
```

The CI workflow also retrains the model artifact before running tests:

```bash
python -m backend.model.train
python -m pytest
docker build -t student-placement-prediction .
```
