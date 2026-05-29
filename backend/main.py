from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
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
        confidence = round(float(probs[best_idx].item()), 4)

        return PredictResponse(label=label, confidence=confidence)


def parse_csv_texts(raw_bytes: bytes) -> list[str]:
    content = raw_bytes.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(content))
    texts: list[str] = []

    if reader.fieldnames and "text" in [name.strip().lower() for name in reader.fieldnames]:
        for row in reader:
            text = str(row.get("text", "")).strip()
            if text:
                texts.append(text)
        return texts

    fallback_reader = csv.reader(io.StringIO(content))
    for row in fallback_reader:
        if not row:
            continue
        text = str(row[0]).strip()
        if text:
            texts.append(text)

    return texts


def parse_json_texts(raw_bytes: bytes) -> list[str]:
    content = raw_bytes.decode("utf-8", errors="ignore")
    payload = json.loads(content)
    texts: list[str] = []

    if not isinstance(payload, list):
        raise ValueError("JSON must be a list of objects with a text field.")

    for item in payload:
        if not isinstance(item, dict) or set(item.keys()) != {"text"}:
            raise ValueError("Each JSON item must be an object with only a text field.")
        text = str(item.get("text", "")).strip()
        if text:
            texts.append(text)

    if not texts:
        raise ValueError("No text values found in JSON.")

    return texts


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


@app.post("/batch")
async def batch_predict(
    file: UploadFile = File(...),
    output_format: str = Form("csv"),
) -> Response:
    if MODEL_ERROR or MODEL is None:
        raise HTTPException(status_code=500, detail="Model failed to load")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()

    try:
        if filename.endswith(".csv") or content_type in {"text/csv", "application/csv"}:
            texts = parse_csv_texts(raw_bytes)
        elif filename.endswith(".json") or content_type in {"application/json", "text/json"}:
            texts = parse_json_texts(raw_bytes)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload CSV or JSON.",
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not texts:
        raise HTTPException(status_code=400, detail="No texts found in upload")

    results = []
    for text in texts:
        prediction = MODEL.predict(text)
        confidence_str = f"{prediction.confidence:.4f}"
        results.append(
            {
                "text": text,
                "label": prediction.label,
                "confidence": confidence_str,
            }
        )

    fmt = output_format.strip().lower()
    if fmt == "json":
        return JSONResponse(
            content=results,
            headers={"Content-Disposition": "attachment; filename=predictions.json"},
        )

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["text", "label", "confidence"])
    writer.writeheader()
    writer.writerows(results)
    csv_bytes = output.getvalue().encode("utf-8")

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )
