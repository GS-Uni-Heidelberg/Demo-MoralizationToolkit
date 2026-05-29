# Moralization Detection Demo

Quick start for the minimal frontend + backend demo.

## Requirements

- Python 3.10+
- Node.js 20+

## Backend (FastAPI)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The backend loads the model from:

```
models/FacebookAI-xlm-roberta-base-finetuned-base_params/checkpoint-1473
```

To override the model path:

```bash
export MODEL_DIR=/absolute/path/to/checkpoint-1473
```

Health check:

```
http://127.0.0.1:8000/health
```

## Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Open:

```
http://localhost:3000
```

## Notes

- CPU-only inference.
- Backend uses CORS for http://localhost:3000.

## Batch processing

Upload CSV or JSON from the UI and download predictions.

CSV format (first column or column named `text`):

```csv
text
Das ist absolut richtig.
So etwas darf niemand tolerieren.
```

JSON format (list of objects with only `text`):

```json
[
	{"text": "Das ist absolut richtig."},
	{"text": "So etwas darf niemand tolerieren."}
]
```
