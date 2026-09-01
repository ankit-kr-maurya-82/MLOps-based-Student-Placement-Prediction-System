from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.model.predict import predict_placement


app = FastAPI(
    title="Student Placement Prediction API",
    description="Predict student placement outcomes from academic and skill data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudentPlacementRequest(BaseModel):
    cgpa: float = Field(..., ge=0, le=10, description="Current CGPA out of 10")
    tenth_percentage: float = Field(..., ge=0, le=100)
    twelfth_percentage: float = Field(..., ge=0, le=100)
    internships: int = Field(..., ge=0, le=10)
    projects: int = Field(..., ge=0, le=20)
    aptitude_score: float = Field(..., ge=0, le=100)
    communication_score: float = Field(..., ge=0, le=100)
    coding_score: float = Field(..., ge=0, le=100)
    backlogs: int = Field(..., ge=0, le=20)
    certifications: int = Field(..., ge=0, le=20)


class PlacementPredictionResponse(BaseModel):
    prediction: Literal["Placed", "Not Placed"]
    probability: float = Field(..., ge=0, le=1)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Placement Prediction API is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=PlacementPredictionResponse)
def predict(request: StudentPlacementRequest) -> dict[str, float | str]:
    try:
        return predict_placement(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
