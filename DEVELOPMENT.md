# Development

This guide covers local development, backend configuration, model hosting, and
billing administration for the Moralization Toolkit.

## Requirements

Install Python and Node.js before running the application locally. The backend
provides the FastAPI API and model integrations; the frontend provides the
interactive analysis interface and batch-upload workflow.

- Python 3.10+
- Node.js 22.23.2 (or newer)

If `npm run dev` reports an unsupported Node.js version, use the pinned version
from `.nvmrc`:

```bash
nvm install
nvm use
```

## Local Development

Run the backend and frontend in separate terminals.

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

### Frontend

From the repository root, run:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3030`. The backend health check is available at
`http://localhost:8000/health`.

## Deployment

Create `backend/.env` from `backend/.env.example`, then start both services
from the repository root:

```bash
docker compose up --build -d
```

The Compose deployment expects an existing external Docker network named
`web`, shared with Traefik. Set `TRAEFIK_NETWORK` if your network uses another
name. Billing data is persisted in the named `billing_data` volume.

For the default production routing, open
`https://moralization-toolkit.chai-lab.de`. To use another frontend or API
origin, set `FRONTEND_ORIGINS` and `NEXT_PUBLIC_API_BASE_URL` before building.

Stop the services with:

```bash
docker compose down
```

## Backend Configuration

Create a local environment file before starting the backend:

```bash
cd backend
cp .env.example .env
```

The backend reads these variables from `backend/.env` or `backend/.env.local`:

- `HF_TOKEN`: private Hugging Face token used only by the backend.
- `HF_MODEL_REPO`: Hugging Face Hub repository containing the model and tokenizer.
- `HF_INFERENCE_ENDPOINT_URL`: private Hugging Face Inference Endpoint URL.
- `HF_MMBERT_INFERENCE_ENDPOINT_URL`: optional private Hugging Face Inference Endpoint URL for mmBERT.
- `HF_SCALE_UP_TIMEOUT`: seconds to wait for a scale-to-zero replica, default `300`.
- `HF_REQUEST_TIMEOUT_SECONDS`: backend timeout for cold starts, default `360`.
- `HF_MORALIZATION_LABEL` and `HF_NON_MORALIZATION_LABEL`: labels stored in the model config, default `moralization` and `no_moralization` for the included checkpoint.
- `LEMMAS_DIR`: path to the DiMi lemma files.
- `ENABLED_LANGUAGES`: comma-separated language codes enabled by the backend, default `de,en,fr`.
- `OPENAI_API_KEY`: OpenAI API credential for the OpenAI branch.
- `OPENAI_MODEL`: OpenAI chat model name, default `gpt-4o-mini`.
- `ANTHROPIC_API_KEY`: Anthropic API credential for the Claude branch.
- `ANTHROPIC_MODEL`: Claude model name, default `claude-3-5-sonnet-latest`.

Billing and request-limit variables:

- `DATABASE_PATH`: SQLite billing database path, default `backend/data/billing.sqlite3`.
- `FREE_TIER_DAILY_CREDITS`: daily credits available to anonymous requests, default `20`.
- `MAX_BATCH_INSTANCES`: maximum number of batch instances per request, default `200000`.
- `MAX_INPUT_TEXT_LENGTH`: maximum input text length, default `5000`.
- `MAX_UPLOAD_BYTES`: maximum uploaded file size in bytes.
- `RATE_LIMIT_REQUESTS`: maximum requests per client and endpoint within the rate-limit window.
- `RATE_LIMIT_WINDOW_SECONDS`: rate-limit window length in seconds.
- `EXTERNAL_REQUEST_CREDIT_COST`: credits charged for `openai` and `claude` requests, default `1`.
- `LOCAL_REQUEST_CREDIT_COST`: credits charged for local model requests, default `0`.
- `ADMIN_API_KEY`: optional key required for administrative token endpoints.

## Encoder-Only Model Configuration

The encoder-only models are fine-tuned multilingual sequence classifiers. The
`xlm-roberta` and `mmbert` options classify text as `moralization` or
`no_moralization`; their endpoint URLs, labels, timeouts, and Hugging Face
credentials are configured through the variables above.

### Upload Models to Hugging Face

Upload the fine-tuned checkpoint, tokenizer, and custom Endpoint handler to a
private Hub repository:

```bash
export HF_TOKEN=hf_...
export HF_MODEL_REPO=your-account/moralization-xlm-roberta
export MODEL_DIR=/absolute/path/to/checkpoint-2500
python upload_model.py
```

To upload the included mmBERT checkpoint, use a separate Hub repository and
Endpoint:

```bash
export HF_MODEL_REPO=your-account/moralization-mmbert
export MODEL_VARIANT=mmbert
python upload_model.py
```

### Create a Hugging Face Inference Endpoint

Create a Hugging Face Inference Endpoint for that repository with task `Custom`,
the cheapest CPU hardware, minimum replicas `0`, maximum replicas `1`, and
scale-to-zero enabled. The `handler.py` file is required for the Custom task.
Put the resulting Endpoint URL and the same read token in `backend/.env`. The
browser sends raw text to this backend; tokenization and model inference happen
inside the Hugging Face Endpoint.

If the Endpoint already exists, upload the model again so that `handler.py`
appears in the Hub repository, then redeploy or restart the Endpoint. Test it
with:

```bash
curl -X POST "$HF_INFERENCE_ENDPOINT_URL" \
	-H "Authorization: Bearer $HF_TOKEN" \
	-H "Content-Type: application/json" \
	-d '{"inputs":"Das ist absolut richtig."}'
```

## Billing and API Tokens

The backend uses a SQLite billing database that is created automatically on
startup. It is stored at `backend/data/billing.sqlite3` by default, or at the
path configured by `DATABASE_PATH`.

### Create an API Token

Send a `POST` request to `/admin/api-tokens`. The required fields are
`accredited_to` and `credits`; optional fields are `note`, `allowed_providers`,
`max_batch_instances`, `max_input_text_length`, and `expires_at`.

```bash
curl -X POST http://localhost:8000/admin/api-tokens \
	-H "Content-Type: application/json" \
	-H "X-Admin-Key: your-admin-secret" \
	-d '{
		"accredited_to": "Research Demo",
		"credits": 500,
		"note": "Conference access",
		"max_batch_instances": 1000,
		"max_input_text_length": 10000,
		"allowed_providers": ["openai", "claude"]
	}'
```

The response contains the generated token. Save it in the frontend panel or
send it with requests. If `max_batch_instances` or `max_input_text_length` is
omitted, the current global default is stored for that token.

If `ADMIN_API_KEY` is configured, token creation requires the `X-Admin-Key`
header.

### List API Tokens

```bash
curl http://localhost:8000/admin/api-tokens \
	-H "X-Admin-Key: your-admin-secret"
```

The response includes each token's accreditation, remaining and maximum
credits, limits, allowed providers, active status, creation time, and expiry.
