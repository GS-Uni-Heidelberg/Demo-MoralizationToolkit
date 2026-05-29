from __future__ import annotations

import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (
    ROOT_DIR
    / "models"
    / "FacebookAI-xlm-roberta-base-finetuned-base_params"
    / "checkpoint-1473"
)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))

app = FastAPI(title="Moralization Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"]
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class PredictResponse(BaseModel):
    label: str
    confidence: float


class ModelBundle:
    def __init__(self, model_dir: Path) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        self.id2label = self.model.config.id2label or {0: "no_moralization", 1: "moralization"}

    def predict(self, text: str) -> PredictResponse:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)

        best_idx = int(torch.argmax(probs).item())
        label = str(self.id2label.get(best_idx, best_idx))
        confidence = float(probs[best_idx].item())

        return PredictResponse(label=label, confidence=confidence)


try:
    MODEL = ModelBundle(MODEL_DIR)
except Exception as exc:
    MODEL = None
    MODEL_ERROR = exc
else:
    MODEL_ERROR = None


@app.get("/health")
async def health() -> dict[str, str]:
    if MODEL_ERROR:
        return {"status": "error", "detail": str(MODEL_ERROR)}
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    if MODEL_ERROR or MODEL is None:
        raise HTTPException(status_code=500, detail="Model failed to load")

    try:
        return MODEL.predict(request.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
