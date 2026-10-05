# Moralization Detection Demo

Quick start for the minimal frontend + backend demo.

## Requirements

- Python 3.10+
- Node.js 22.23.2 (or newer)

If `npm run dev` reports an unsupported Node.js version, use the pinned version from `.nvmrc`:

```bash
nvm install
nvm use
```

## Backend (FastAPI)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Backend configuration

Create a local env file before starting the backend:

```bash
cp .env.example .env
```

The backend reads these variables from `backend/.env` or `backend/.env.local`:

- `MODEL_DIR`: path to the local XLM-RoBERTa checkpoint.
- `LEMMAS_DIR`: path to the DiMi lemma files.
- `OPENAI_API_KEY`: OpenAI API credential for the OpenAI branch.
- `OPENAI_MODEL`: OpenAI chat model name, default `gpt-4o-mini`.
- `ANTHROPIC_API_KEY`: reserved for the Claude branch.
- `ANTHROPIC_MODEL`: reserved Claude model name, default `claude-3-5-sonnet-latest`.

Example:

```bash
export MODEL_DIR=/absolute/path/to/checkpoint-1473
export LEMMAS_DIR=/absolute/path/to/dimi
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=claude-3-5-sonnet-latest
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

From the repository root:

```bash
cd frontend
npm install
npm run dev
```

If your terminal is already in `frontend`, omit `cd frontend` and run only
`npm install` followed by `npm run dev`.

If you still see the Node.js version error, confirm the active version with `node --version` and switch to Node 22.23.2 or newer.

Open:

```
http://localhost:3000
```

For a deployed frontend, set `NEXT_PUBLIC_API_BASE_URL` to the HTTPS origin of the
backend before building. The local fallback is `http://localhost:8000`.

The backend also supports these safety settings in its environment file:

- `MAX_UPLOAD_BYTES` (default `50000000`)
- `RATE_LIMIT_REQUESTS` (default `120` per client and endpoint)
- `RATE_LIMIT_WINDOW_SECONDS` (default `60`)

The in-process rate limit is a baseline for a single worker. Use a reverse proxy
or shared rate-limit store for multi-worker or multi-instance deployments.

## Notes

- CPU-only inference.
- Backend uses CORS for http://localhost:3000.
- Billing and token posting instructions are in [backend/BILLING.md](backend/BILLING.md).

## Batch processing

Upload CSV or JSON from the UI and download predictions.

CSV format (first column or column named `text`, optional `label` column):

```csv
text,label
Das ist absolut richtig.,moralization
So etwas darf niemand tolerieren.,1
```

JSON format (list of objects with `text` and optional `label`):

```json
[
	{"text": "Das ist absolut richtig.", "label": "moralization"},
	{"text": "So etwas darf niemand tolerieren.", "label": 1}
]

Accepted label values: true/false, 0/1, moralization/no_moralization.
```
