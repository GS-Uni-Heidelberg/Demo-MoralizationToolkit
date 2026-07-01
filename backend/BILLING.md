# Billing and API Tokens

This backend uses a SQLite billing database that is created automatically on startup.

## How to create an API token

Send a `POST` request to `/admin/api-tokens`.

Required JSON fields:

- `accredited_to`
- `credits`

Optional JSON fields:

- `note`
- `allowed_providers`
- `max_batch_instances`
- `max_input_text_length`
- `expires_at`

Example:

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

The response returns the generated token. Save it in the frontend panel or send it in request headers.
If you omit `max_batch_instances` or `max_input_text_length`, the backend stores the current global defaults for that token.

## How to see all tokens

Send a `GET` request to `/admin/api-tokens`.

Example:

```bash
curl http://localhost:8000/admin/api-tokens \
  -H "X-Admin-Key: your-admin-secret"
```

The response is a JSON array with:

- `token`
- `accredited_to`
- `note`
- `credits_remaining`
- `max_credits`
- `max_batch_instances`
- `max_input_text_length`
- `allowed_providers`
- `is_active`
- `created_at`
- `expires_at`

## How to post requests with a token

Use the `X-API-Token` header for any billable request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Token: mk_your_generated_token" \
  -d '{
    "text": "Example text",
    "model": "claude"
  }'
```

You can also use `Authorization: Bearer <token>`.

The token-specific `max_batch_instances` and `max_input_text_length` are enforced automatically on prediction and batch endpoints.

## What costs credits

- `openai` and `claude` predictions cost credits.
- `xlm-roberta` and DiMi/local lemmatization do not cost credits.
- Anonymous requests use the free tier defined in the backend settings.

## Configuration

Relevant environment variables:

- `DATABASE_PATH`
- `FREE_TIER_DAILY_CREDITS`
- `MAX_BATCH_INSTANCES`
- `MAX_INPUT_TEXT_LENGTH`
- `EXTERNAL_REQUEST_CREDIT_COST`
- `ADMIN_API_KEY`

## Notes

- The database is stored in `backend/data/billing.sqlite3` by default.
- If `ADMIN_API_KEY` is set, token creation requires `X-Admin-Key`.
