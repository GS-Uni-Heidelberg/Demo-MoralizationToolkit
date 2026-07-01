from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = Path(
    os.environ.get("DATABASE_PATH", ROOT_DIR / "backend" / "data" / "billing.sqlite3")
)
FREE_TIER_DAILY_CREDITS_ENV = os.environ.get("FREE_TIER_DAILY_CREDITS")
MAX_BATCH_INSTANCES_ENV = os.environ.get("MAX_BATCH_INSTANCES")
MAX_INPUT_TEXT_LENGTH_ENV = os.environ.get("MAX_INPUT_TEXT_LENGTH")
EXTERNAL_REQUEST_CREDIT_COST_ENV = os.environ.get("EXTERNAL_REQUEST_CREDIT_COST")

DEFAULT_FREE_TIER_DAILY_CREDITS = int(FREE_TIER_DAILY_CREDITS_ENV or "5")
DEFAULT_MAX_BATCH_INSTANCES = int(MAX_BATCH_INSTANCES_ENV or "200000")
DEFAULT_MAX_INPUT_TEXT_LENGTH = int(MAX_INPUT_TEXT_LENGTH_ENV or "5000")
DEFAULT_EXTERNAL_REQUEST_CREDIT_COST = int(EXTERNAL_REQUEST_CREDIT_COST_ENV or "1")
DEFAULT_LOCAL_REQUEST_CREDIT_COST = 0

DB_LOCK = threading.Lock()


class BillingError(Exception):
    """Base class for billing-related problems."""


class CreditLimitError(BillingError):
    pass


class TokenNotFoundError(BillingError):
    pass


class TokenInactiveError(BillingError):
    pass


class TokenExpiredError(BillingError):
    pass


class ProviderNotAllowedError(BillingError):
    pass


@dataclass(frozen=True)
class BillingSettings:
    free_tier_daily_credits: int
    max_batch_instances: int
    max_input_text_length: int
    external_request_credit_cost: int
    local_request_credit_cost: int


@dataclass(frozen=True)
class ApiTokenRecord:
    token: str
    accredited_to: str
    note: str | None
    credits_remaining: int
    max_credits: int
    max_batch_instances: int | None
    max_input_text_length: int | None
    allowed_providers: list[str]
    is_active: bool
    created_at: str
    expires_at: str | None


@dataclass(frozen=True)
class CreditChargeResult:
    token_type: str
    token: str | None
    remaining_credits: int | None
    daily_credits_remaining: int | None
    credits_used: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _connect() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_database() -> None:
    with DB_LOCK:
        with _connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS api_tokens (
                    token TEXT PRIMARY KEY,
                    accredited_to TEXT NOT NULL,
                    note TEXT,
                    credits_remaining INTEGER NOT NULL,
                    max_credits INTEGER NOT NULL,
                    max_batch_instances INTEGER,
                    max_input_text_length INTEGER,
                    allowed_providers TEXT NOT NULL DEFAULT '[]',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                );

                CREATE TABLE IF NOT EXISTS api_usage (
                    id TEXT PRIMARY KEY,
                    token TEXT,
                    token_type TEXT NOT NULL,
                    accredited_to TEXT,
                    provider TEXT NOT NULL,
                    request_kind TEXT NOT NULL,
                    credits_used INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS free_tier_usage (
                    usage_date TEXT PRIMARY KEY,
                    credits_used INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            _seed_setting(connection, "free_tier_daily_credits", DEFAULT_FREE_TIER_DAILY_CREDITS)
            _seed_setting(connection, "max_batch_instances", DEFAULT_MAX_BATCH_INSTANCES)
            _seed_setting(connection, "max_input_text_length", DEFAULT_MAX_INPUT_TEXT_LENGTH)
            _seed_setting(
                connection,
                "external_request_credit_cost",
                DEFAULT_EXTERNAL_REQUEST_CREDIT_COST,
            )
            _seed_setting(
                connection,
                "local_request_credit_cost",
                DEFAULT_LOCAL_REQUEST_CREDIT_COST,
            )
            _sync_env_override(connection, "free_tier_daily_credits", FREE_TIER_DAILY_CREDITS_ENV)
            _sync_env_override(connection, "max_batch_instances", MAX_BATCH_INSTANCES_ENV)
            _sync_env_override(connection, "max_input_text_length", MAX_INPUT_TEXT_LENGTH_ENV)
            _sync_env_override(
                connection,
                "external_request_credit_cost",
                EXTERNAL_REQUEST_CREDIT_COST_ENV,
            )
            _ensure_token_limit_columns(connection)
            _backfill_token_limit_values(connection)
            connection.commit()


def _seed_setting(connection: sqlite3.Connection, key: str, default_value: int) -> None:
    existing = connection.execute(
        "SELECT key FROM app_settings WHERE key = ?",
        (key,),
    ).fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, str(default_value), _utc_now()),
        )


def _sync_env_override(
    connection: sqlite3.Connection,
    key: str,
    override_value: str | None,
) -> None:
    if override_value is None:
        return

    connection.execute(
        """
        INSERT INTO app_settings (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
        """,
        (key, override_value, _utc_now()),
    )


def _ensure_token_limit_columns(connection: sqlite3.Connection) -> None:
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(api_tokens)").fetchall()
    }
    if "max_batch_instances" not in columns:
        connection.execute("ALTER TABLE api_tokens ADD COLUMN max_batch_instances INTEGER")
    if "max_input_text_length" not in columns:
        connection.execute("ALTER TABLE api_tokens ADD COLUMN max_input_text_length INTEGER")


def _backfill_token_limit_values(connection: sqlite3.Connection) -> None:
    settings = _get_settings_map(connection)
    connection.execute(
        """
        UPDATE api_tokens
        SET max_batch_instances = COALESCE(max_batch_instances, ?),
            max_input_text_length = COALESCE(max_input_text_length, ?)
        """,
        (
            int(settings["max_batch_instances"]),
            int(settings["max_input_text_length"]),
        ),
    )


def get_free_tier_remaining_credits() -> tuple[int, int]:
    with _connect() as connection:
        settings = _get_settings_map(connection)
        free_tier_limit = int(settings["free_tier_daily_credits"])
        today = _utc_today()
        row = connection.execute(
            "SELECT credits_used FROM free_tier_usage WHERE usage_date = ?",
            (today,),
        ).fetchone()
        current_usage = int(row["credits_used"]) if row else 0
    remaining = max(free_tier_limit - current_usage, 0)
    return remaining, free_tier_limit


def _get_settings_map(connection: sqlite3.Connection) -> dict[str, str]:
    rows = connection.execute("SELECT key, value FROM app_settings").fetchall()
    settings = {row["key"]: row["value"] for row in rows}
    settings.setdefault("free_tier_daily_credits", str(DEFAULT_FREE_TIER_DAILY_CREDITS))
    settings.setdefault("max_batch_instances", str(DEFAULT_MAX_BATCH_INSTANCES))
    settings.setdefault("max_input_text_length", str(DEFAULT_MAX_INPUT_TEXT_LENGTH))
    settings.setdefault(
        "external_request_credit_cost", str(DEFAULT_EXTERNAL_REQUEST_CREDIT_COST)
    )
    settings.setdefault("local_request_credit_cost", str(DEFAULT_LOCAL_REQUEST_CREDIT_COST))
    return settings


def get_billing_settings() -> BillingSettings:
    with _connect() as connection:
        settings = _get_settings_map(connection)
    return BillingSettings(
        free_tier_daily_credits=int(settings["free_tier_daily_credits"]),
        max_batch_instances=int(settings["max_batch_instances"]),
        max_input_text_length=int(settings["max_input_text_length"]),
        external_request_credit_cost=int(settings["external_request_credit_cost"]),
        local_request_credit_cost=int(settings["local_request_credit_cost"]),
    )


def list_billing_settings() -> dict[str, int]:
    settings = get_billing_settings()
    return {
        "free_tier_daily_credits": settings.free_tier_daily_credits,
        "max_batch_instances": settings.max_batch_instances,
        "max_input_text_length": settings.max_input_text_length,
        "external_request_credit_cost": settings.external_request_credit_cost,
        "local_request_credit_cost": settings.local_request_credit_cost,
    }


def _normalize_allowed_providers(allowed_providers: list[str] | None) -> list[str]:
    if not allowed_providers:
        return []
    normalized: list[str] = []
    for provider in allowed_providers:
        code = provider.strip().lower()
        if code and code not in normalized:
            normalized.append(code)
    return normalized


def generate_api_token() -> str:
    return f"mk_{secrets.token_urlsafe(32)}"


def create_api_token(
    accredited_to: str,
    credits: int,
    note: str | None = None,
    allowed_providers: list[str] | None = None,
    max_batch_instances: int | None = None,
    max_input_text_length: int | None = None,
    expires_at: str | None = None,
) -> ApiTokenRecord:
    if credits <= 0:
        raise ValueError("Credits must be greater than zero.")

    billing_settings = get_billing_settings()
    effective_max_batch_instances = (
        max_batch_instances if max_batch_instances is not None else billing_settings.max_batch_instances
    )
    effective_max_input_text_length = (
        max_input_text_length if max_input_text_length is not None else billing_settings.max_input_text_length
    )
    token = generate_api_token()
    normalized_allowed_providers = _normalize_allowed_providers(allowed_providers)
    now = _utc_now()

    with DB_LOCK:
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO api_tokens (
                    token,
                    accredited_to,
                    note,
                    credits_remaining,
                    max_credits,
                    max_batch_instances,
                    max_input_text_length,
                    allowed_providers,
                    is_active,
                    created_at,
                    expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token,
                    accredited_to.strip(),
                    note.strip() if note else None,
                    credits,
                    credits,
                    effective_max_batch_instances,
                    effective_max_input_text_length,
                    json.dumps(normalized_allowed_providers),
                    1,
                    now,
                    expires_at,
                ),
            )
            connection.commit()

    return ApiTokenRecord(
        token=token,
        accredited_to=accredited_to.strip(),
        note=note.strip() if note else None,
        credits_remaining=credits,
        max_credits=credits,
        max_batch_instances=effective_max_batch_instances,
        max_input_text_length=effective_max_input_text_length,
        allowed_providers=normalized_allowed_providers,
        is_active=True,
        created_at=now,
        expires_at=expires_at,
    )


def _row_to_token_record(row: sqlite3.Row) -> ApiTokenRecord:
    allowed_providers = json.loads(row["allowed_providers"] or "[]")
    return ApiTokenRecord(
        token=row["token"],
        accredited_to=row["accredited_to"],
        note=row["note"],
        credits_remaining=int(row["credits_remaining"]),
        max_credits=int(row["max_credits"]),
        max_batch_instances=int(row["max_batch_instances"]) if row["max_batch_instances"] is not None else None,
        max_input_text_length=int(row["max_input_text_length"]) if row["max_input_text_length"] is not None else None,
        allowed_providers=list(allowed_providers),
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        expires_at=row["expires_at"],
    )


def get_api_token(token: str) -> ApiTokenRecord | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM api_tokens WHERE token = ?",
            (token,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_token_record(row)


def list_api_tokens() -> list[ApiTokenRecord]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM api_tokens ORDER BY created_at DESC",
        ).fetchall()
    return [_row_to_token_record(row) for row in rows]


def _is_token_expired(record: ApiTokenRecord) -> bool:
    if record.expires_at is None:
        return False
    expires_at = datetime.fromisoformat(record.expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= datetime.now(timezone.utc)


def _provider_is_allowed(record: ApiTokenRecord, provider: str) -> bool:
    if not record.allowed_providers:
        return True
    return provider.lower() in record.allowed_providers


def _providers_are_allowed(record: ApiTokenRecord, providers: list[str]) -> bool:
    if not record.allowed_providers:
        return True
    return all(provider.lower() in record.allowed_providers for provider in providers)


def _record_usage(
    connection: sqlite3.Connection,
    *,
    token: str | None,
    token_type: str,
    accredited_to: str | None,
    provider: str,
    request_kind: str,
    credits_used: int,
) -> None:
    connection.execute(
        """
        INSERT INTO api_usage (
            id,
            token,
            token_type,
            accredited_to,
            provider,
            request_kind,
            credits_used,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            token,
            token_type,
            accredited_to,
            provider,
            request_kind,
            credits_used,
            _utc_now(),
        ),
    )


def charge_credits(
    *,
    api_token: str | None,
    provider: str,
    credits_used: int,
    request_kind: str,
) -> CreditChargeResult:
    if credits_used < 0:
        raise ValueError("credits_used must be positive or zero.")

    if credits_used == 0:
        with _connect() as connection:
            _record_usage(
                connection,
                token=api_token,
                token_type="free_tier" if api_token is None else "api_token",
                accredited_to=None,
                provider=provider,
                request_kind=request_kind,
                credits_used=0,
            )
            connection.commit()
        return CreditChargeResult(
            token_type="free_tier" if api_token is None else "api_token",
            token=api_token,
            remaining_credits=None,
            daily_credits_remaining=None,
            credits_used=0,
        )

    with DB_LOCK:
        with _connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if api_token:
                row = connection.execute(
                    "SELECT * FROM api_tokens WHERE token = ?",
                    (api_token,),
                ).fetchone()
                if row is None:
                    raise TokenNotFoundError("API token not found.")

                record = _row_to_token_record(row)
                if not record.is_active:
                    raise TokenInactiveError("API token is inactive.")
                if _is_token_expired(record):
                    raise TokenExpiredError("API token has expired.")
                if not _provider_is_allowed(record, provider):
                    raise ProviderNotAllowedError("API token is not allowed for this provider.")
                if record.credits_remaining < credits_used:
                    raise CreditLimitError("Not enough credits left on the API token.")

                remaining = record.credits_remaining - credits_used
                connection.execute(
                    "UPDATE api_tokens SET credits_remaining = ? WHERE token = ?",
                    (remaining, api_token),
                )
                _record_usage(
                    connection,
                    token=api_token,
                    token_type="api_token",
                    accredited_to=record.accredited_to,
                    provider=provider,
                    request_kind=request_kind,
                    credits_used=credits_used,
                )
                connection.commit()
                return CreditChargeResult(
                    token_type="api_token",
                    token=api_token,
                    remaining_credits=remaining,
                    daily_credits_remaining=None,
                    credits_used=credits_used,
                )

            settings = _get_settings_map(connection)
            free_tier_limit = int(settings["free_tier_daily_credits"])
            today = _utc_today()
            row = connection.execute(
                "SELECT credits_used FROM free_tier_usage WHERE usage_date = ?",
                (today,),
            ).fetchone()
            current_usage = int(row["credits_used"]) if row else 0
            next_usage = current_usage + credits_used
            if next_usage > free_tier_limit:
                raise CreditLimitError("Free-tier daily credits exhausted.")

            connection.execute(
                """
                INSERT INTO free_tier_usage (usage_date, credits_used, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(usage_date) DO UPDATE SET
                    credits_used = excluded.credits_used,
                    updated_at = excluded.updated_at
                """,
                (today, next_usage, _utc_now()),
            )
            _record_usage(
                connection,
                token=None,
                token_type="free_tier",
                accredited_to=None,
                provider=provider,
                request_kind=request_kind,
                credits_used=credits_used,
            )
            connection.commit()
            return CreditChargeResult(
                token_type="free_tier",
                token=None,
                remaining_credits=None,
                daily_credits_remaining=free_tier_limit - next_usage,
                credits_used=credits_used,
            )


def charge_batch_credits(
    *,
    api_token: str | None,
    providers: list[str],
    credits_used: int,
    request_kind: str,
) -> CreditChargeResult:
    if credits_used < 0:
        raise ValueError("credits_used must be positive or zero.")
    normalized_providers = [provider.strip().lower() for provider in providers if provider.strip()]
    provider_label = ",".join(normalized_providers) if normalized_providers else "batch"

    if credits_used == 0:
        return charge_credits(
            api_token=api_token,
            provider=provider_label,
            credits_used=0,
            request_kind=request_kind,
        )

    with DB_LOCK:
        with _connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if api_token:
                row = connection.execute(
                    "SELECT * FROM api_tokens WHERE token = ?",
                    (api_token,),
                ).fetchone()
                if row is None:
                    raise TokenNotFoundError("API token not found.")

                record = _row_to_token_record(row)
                if not record.is_active:
                    raise TokenInactiveError("API token is inactive.")
                if _is_token_expired(record):
                    raise TokenExpiredError("API token has expired.")
                if not _providers_are_allowed(record, normalized_providers):
                    raise ProviderNotAllowedError("API token is not allowed for one or more selected providers.")
                if record.credits_remaining < credits_used:
                    raise CreditLimitError("Not enough credits left on the API token.")

                remaining = record.credits_remaining - credits_used
                connection.execute(
                    "UPDATE api_tokens SET credits_remaining = ? WHERE token = ?",
                    (remaining, api_token),
                )
                _record_usage(
                    connection,
                    token=api_token,
                    token_type="api_token",
                    accredited_to=record.accredited_to,
                    provider=provider_label,
                    request_kind=request_kind,
                    credits_used=credits_used,
                )
                connection.commit()
                return CreditChargeResult(
                    token_type="api_token",
                    token=api_token,
                    remaining_credits=remaining,
                    daily_credits_remaining=None,
                    credits_used=credits_used,
                )

            settings = _get_settings_map(connection)
            free_tier_limit = int(settings["free_tier_daily_credits"])
            today = _utc_today()
            row = connection.execute(
                "SELECT credits_used FROM free_tier_usage WHERE usage_date = ?",
                (today,),
            ).fetchone()
            current_usage = int(row["credits_used"]) if row else 0
            next_usage = current_usage + credits_used
            if next_usage > free_tier_limit:
                raise CreditLimitError("Free-tier daily credits exhausted.")

            connection.execute(
                """
                INSERT INTO free_tier_usage (usage_date, credits_used, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(usage_date) DO UPDATE SET
                    credits_used = excluded.credits_used,
                    updated_at = excluded.updated_at
                """,
                (today, next_usage, _utc_now()),
            )
            _record_usage(
                connection,
                token=None,
                token_type="free_tier",
                accredited_to=None,
                provider=provider_label,
                request_kind=request_kind,
                credits_used=credits_used,
            )
            connection.commit()
            return CreditChargeResult(
                token_type="free_tier",
                token=None,
                remaining_credits=None,
                daily_credits_remaining=free_tier_limit - next_usage,
                credits_used=credits_used,
            )


def provider_credit_cost(provider: str, settings: BillingSettings) -> int:
    return settings.local_request_credit_cost if provider == "xlm-roberta" else settings.external_request_credit_cost


def summarize_batch_credit_need(
    *,
    eligible_rows: int,
    selected_models: list[str],
    settings: BillingSettings,
) -> int:
    external_models = [model for model in selected_models if model != "xlm-roberta"]
    if not external_models or eligible_rows <= 0:
        return 0
    return eligible_rows * len(external_models) * settings.external_request_credit_cost
