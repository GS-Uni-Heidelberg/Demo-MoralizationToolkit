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

- `HF_TOKEN`: private Hugging Face token used only by the backend.
- `HF_MODEL_REPO`: Hugging Face Hub repository containing the model and tokenizer.
- `HF_INFERENCE_ENDPOINT_URL`: private Hugging Face Inference Endpoint URL.
- `HF_SCALE_UP_TIMEOUT`: seconds to wait for a scale-to-zero replica, default `300`.
- `HF_REQUEST_TIMEOUT_SECONDS`: backend timeout for cold starts, default `360`.
- `HF_MORALIZATION_LABEL` and `HF_NON_MORALIZATION_LABEL`: labels stored in the model config, default `moralization` and `no_moralization` for the included checkpoint.
- `LEMMAS_DIR`: path to the DiMi lemma files.
- `OPENAI_API_KEY`: OpenAI API credential for the OpenAI branch.
- `OPENAI_MODEL`: OpenAI chat model name, default `gpt-4o-mini`.
- `ANTHROPIC_API_KEY`: reserved for the Claude branch.
- `ANTHROPIC_MODEL`: reserved Claude model name, default `claude-3-5-sonnet-latest`.

Example:

```bash
export LEMMAS_DIR=/absolute/path/to/dimi
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

Upload the fine-tuned checkpoint, tokenizer, and custom Endpoint handler to a private Hub repository:

```bash
export HF_TOKEN=hf_...
export HF_MODEL_REPO=your-account/moralization-xlm-roberta
export MODEL_DIR=/absolute/path/to/checkpoint-2500
python upload_model.py
```

Create a Hugging Face Inference Endpoint for that repository with task `Custom`, the cheapest CPU hardware, minimum replicas `0`, maximum replicas `1`, and scale-to-zero enabled. The `handler.py` file is required for the Custom task. Put the resulting Endpoint URL and the same read token in `backend/.env`. The browser sends raw text to this backend; tokenization and model inference happen inside the Hugging Face Endpoint.

If the Endpoint already exists, upload the model again so that `handler.py` appears in the Hub repository, then redeploy or restart the Endpoint. Test it with:

```bash
curl -X POST "$HF_INFERENCE_ENDPOINT_URL" \
	-H "Authorization: Bearer $HF_TOKEN" \
	-H "Content-Type: application/json" \
	-d '{"inputs":"Das ist absolut richtig."}'
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

## Docker Compose deployment

Create `backend/.env` from `backend/.env.example` and fill in the provider credentials. The Compose file expects an existing external Docker network named `web`, shared with the running Traefik container. Adjust `TRAEFIK_NETWORK` if your network has another name. Then build and start both services from the repository root:

```bash
docker compose up --build -d
```

The application is available at `https://moralization-toolkit.chai-lab.de`, and the API health check is at `https://moralization-toolkit.chai-lab.de/api/health`. Traefik routes `/api` to the backend and removes that prefix before forwarding. Billing data is stored in the named `billing_data` volume.

The frontend is built to call `/api` on the same hostname. To use a different public API URL, set `NEXT_PUBLIC_API_BASE_URL` before building:

```bash
NEXT_PUBLIC_API_BASE_URL=https://api.example.com docker compose up --build -d
```

The frontend API URL is baked into the Next.js build because browser requests must use a URL reachable from the user's browser. Set `FRONTEND_ORIGINS` to the deployed frontend origin when it differs from `https://moralization-toolkit.chai-lab.de`.

If the Traefik installation uses a named ACME certificate resolver, add its name to both routers as `traefik.http.routers.<router-name>.tls.certresolver=<resolver-name>`.

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
