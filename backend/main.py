from __future__ import annotations

import csv
import io
import json
import os
import threading
import uuid
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
    allow_headers=["*"],
    expose_headers=[
        "X-Metrics-Accuracy",
        "X-Metrics-Precision",
        "X-Metrics-Recall",
        "X-Metrics-F1",
        "X-Metrics-TP",
        "X-Metrics-FP",
        "X-Metrics-TN",
        "X-Metrics-FN",
    ],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class PredictResponse(BaseModel):
    label: str
    confidence: float


class BatchJob:
    def __init__(
        self,
        job_id: str,
        output_format: str,
        texts: list[str],
        labels: list[int] | None,
        ids: list[str],
    ) -> None:
        self.job_id = job_id
        self.output_format = output_format
        self.texts = texts
        self.labels = labels
        self.ids = ids
        self.total = len(texts)
        self.processed = 0
        self.status = "queued"
        self.error: str | None = None
        self.metrics: dict[str, str] | None = None
        self.result_bytes: bytes | None = None
        self.result_json: dict[str, object] | None = None


JOBS: dict[str, BatchJob] = {}
JOB_LOCK = threading.Lock()


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


def parse_label(value: object) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and value in {0, 1}:
        return int(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "moralization"}:
        return 1
    if text in {"0", "false", "no_moralization"}:
        return 0

    raise ValueError(f"Invalid label value: {value}")


def parse_csv_texts(raw_bytes: bytes) -> tuple[list[str], list[int] | None, list[str]]:
    content = raw_bytes.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(content))
    texts: list[str] = []
    labels: list[int] = []
    ids: list[str] = []

    if reader.fieldnames:
        normalized = [name.strip().lower() for name in reader.fieldnames]
        text_field = next(
            (reader.fieldnames[idx] for idx, name in enumerate(normalized) if name == "text"),
            None,
        )
        label_field = next(
            (reader.fieldnames[idx] for idx, name in enumerate(normalized) if name == "label"),
            None,
        )
        id_field = next(
            (reader.fieldnames[idx] for idx, name in enumerate(normalized) if name == "id"),
            None,
        )
    else:
        text_field = None
        label_field = None
        id_field = None

    if text_field:
        for row in reader:
            text = str(row.get(text_field, "")).strip()
            if text:
                texts.append(text)
                if id_field:
                    ids.append(str(row.get(id_field, "")).strip())
            if label_field:
                raw_label = row.get(label_field, "")
                if raw_label in {None, ""}:
                    raise ValueError("Label column is present but has empty values.")
                labels.append(parse_label(raw_label))
        if not id_field:
            ids = [str(index) for index in range(1, len(texts) + 1)]
        return texts, labels if label_field else None, ids

    fallback_reader = csv.reader(io.StringIO(content))
    for row in fallback_reader:
        if not row:
            continue
        text = str(row[0]).strip()
        if text:
            texts.append(text)
    ids = [str(index) for index in range(1, len(texts) + 1)]

    return texts, None, ids


def parse_json_texts(raw_bytes: bytes) -> tuple[list[str], list[int] | None, list[str]]:
    content = raw_bytes.decode("utf-8", errors="ignore")
    payload = json.loads(content)
    texts: list[str] = []
    labels: list[int] = []
    ids: list[str | None] = []

    if not isinstance(payload, list):
        raise ValueError("JSON must be a list of objects with a text field.")

    label_present = False
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each JSON item must be an object with a text field.")
        if set(item.keys()) - {"text", "label", "id"}:
            raise ValueError("Only text, optional label, and optional id fields are allowed in JSON items.")

        text = str(item.get("text", "")).strip()
        if text:
            texts.append(text)
            ids.append(str(item.get("id")).strip() if "id" in item else None)
        if "label" in item:
            label_present = True
            labels.append(parse_label(item.get("label")))
        elif label_present:
            raise ValueError("All JSON items must include label when any label is provided.")

    if not texts:
        raise ValueError("No text values found in JSON.")

    if label_present and len(labels) != len(texts):
        raise ValueError("All JSON items must include label when any label is provided.")

    resolved_ids: list[str] = []
    for index, value in enumerate(ids, start=1):
        resolved_ids.append(value if value else str(index))

    return texts, labels if label_present else None, resolved_ids


def label_to_binary(label: str) -> int:
    value = label.strip().lower()
    if value == "moralization":
        return 1
    if value == "no_moralization":
        return 0
    raise ValueError(f"Unknown prediction label: {label}")


def compute_metrics(true_labels: list[int], predicted_labels: list[int]) -> dict[str, str]:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Label lengths do not match.")

    tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(true_labels) if true_labels else 0.0

    return {
        "accuracy": f"{accuracy:.4f}",
        "precision": f"{precision:.4f}",
        "recall": f"{recall:.4f}",
        "f1": f"{f1:.4f}",
        "tp": str(tp),
        "fp": str(fp),
        "tn": str(tn),
        "fn": str(fn),
    }


def run_batch_job(job: BatchJob) -> None:
    job.status = "running"
    results = []
    predicted_binary: list[int] = []

    for index, text in enumerate(job.texts, start=1):
        prediction = MODEL.predict(text)
        predicted_binary.append(label_to_binary(prediction.label))
        confidence_str = f"{prediction.confidence:.4f}"
        result_item = {"id": job.ids[index - 1], "text": text}
        if job.labels:
            result_item["true_label"] = "moralization" if job.labels[index - 1] == 1 else "no_moralization"
        result_item["label"] = prediction.label
        result_item["confidence"] = confidence_str
        results.append(result_item)
        job.processed = index

    # Ensure true_label is always the normalized string
    if job.labels:
        for idx, item in enumerate(results):
            item["true_label"] = "moralization" if job.labels[idx] == 1 else "no_moralization"
        job.metrics = compute_metrics(job.labels, predicted_binary)

    if job.output_format == "json":
        job.result_json = {"results": results}
    else:
        output = io.StringIO()
        fieldnames = ["id", "text", "label", "confidence"]
        if job.labels:
            fieldnames = ["id", "text", "true_label", "label", "confidence"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        job.result_bytes = output.getvalue().encode("utf-8")

    job.status = "completed"


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


@app.post("/batch/start")
async def batch_start(
    file: UploadFile = File(...),
    output_format: str = Form("csv"),
) -> dict[str, str]:
    if MODEL_ERROR or MODEL is None:
        raise HTTPException(status_code=500, detail="Model failed to load")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()

    try:
        if filename.endswith(".csv") or content_type in {"text/csv", "application/csv"}:
            texts, labels, ids = parse_csv_texts(raw_bytes)
        elif filename.endswith(".json") or content_type in {"application/json", "text/json"}:
            texts, labels, ids = parse_json_texts(raw_bytes)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload CSV or JSON.",
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not texts:
        raise HTTPException(status_code=400, detail="No texts found in upload")
    fmt = output_format.strip().lower()
    if fmt not in {"csv", "json"}:
        raise HTTPException(status_code=400, detail="Unsupported output format")

    job_id = str(uuid.uuid4())
    job = BatchJob(job_id=job_id, output_format=fmt, texts=texts, labels=labels, ids=ids)
    with JOB_LOCK:
        JOBS[job_id] = job

    thread = threading.Thread(target=run_batch_job, args=(job,), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/batch/status/{job_id}")
async def batch_status(job_id: str) -> dict[str, object]:
    with JOB_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    progress = int((job.processed / job.total) * 100) if job.total else 0
    payload: dict[str, object] = {
        "job_id": job.job_id,
        "status": job.status,
        "processed": job.processed,
        "total": job.total,
        "progress": progress,
    }
    if job.error:
        payload["error"] = job.error
    if job.metrics:
        payload["metrics"] = job.metrics

    return payload


@app.get("/batch/result/{job_id}")
async def batch_result(job_id: str) -> Response:
    with JOB_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Job not completed")
    if job.output_format == "json":
        if job.result_json is None:
            raise HTTPException(status_code=500, detail="Missing JSON result")
        return JSONResponse(
            content=job.result_json,
            headers={"Content-Disposition": "attachment; filename=predictions.json"},
        )

    if job.result_bytes is None:
        raise HTTPException(status_code=500, detail="Missing CSV result")

    return Response(
        content=job.result_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )
