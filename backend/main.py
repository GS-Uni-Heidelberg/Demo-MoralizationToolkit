from __future__ import annotations

import csv
import importlib
import io
import json
import os
import re
import time
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import openpyxl
import stanza
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None

ROOT_DIR = Path(__file__).resolve().parents[1]

if load_dotenv is not None:
    load_dotenv(ROOT_DIR / "backend" / ".env")
    load_dotenv(ROOT_DIR / "backend" / ".env.local", override=True)

DEFAULT_MODEL_DIR = (
    ROOT_DIR
    / "models"
    / "FacebookAI-xlm-roberta-base-finetuned-base_params"
    / "checkpoint-1473"
)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
MODEL_NAME = "roberta-finetuned"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
OPENAI_CLIENT = None
ANTHROPIC_CLIENT = None
OPENAI_PROMPT_DIR = ROOT_DIR / "backend" / "prompts" / "openai"
ANTHROPIC_PROMPT_DIR = ROOT_DIR / "backend" / "prompts" / "anthropic"

MODEL_COLUMN_SUFFIXES = {
    "xlm-roberta": MODEL_NAME,
    "claude": "claude",
    "openai": "openai",
}

LEMMAS_DIR = Path(os.environ.get("LEMMAS_DIR", ROOT_DIR / "models" / "dimi"))

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


# ── MORALIZATION MODELS ───────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    model: str = Field(
        default="xlm-roberta",
        description="Selected model for moralization prediction.",
    )


class PredictResponse(BaseModel):
    label: str
    confidence: float
    explanation: str | None = None


@dataclass
class ModelPredictionResult:
    label: str
    confidence: float | None = None
    explanation: str | None = None
    raw_output: str | None = None


# ── LEMMATIZER MODELS ─────────────────────────────────────────────────────────


class LemmatizerRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)
    language: str = Field(..., description="ISO 639-1 code, e.g. 'en', 'de'")


class LemmatizerResponse(BaseModel):
    matches: list[dict]
    total_sentences: int


class LemmaBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=50000)
    language: str = Field(..., description="ISO 639-1 code, e.g. 'en', 'de'")


# ── BATCH JOB ─────────────────────────────────────────────────────────────────


class BatchJob:
    def __init__(
        self,
        job_id: str,
        output_format: str,
        models: list[str],
        texts: list[str],
        labels: list[int] | None,
        ids: list[str],
        extras: list[dict[str, object]],
        extra_fieldnames: list[str],
        dimi_matches: list[int],
        skip_no_dimi_matches: bool,
    ) -> None:
        self.job_id = job_id
        self.output_format = output_format
        self.models = models
        self.texts = texts
        self.labels = labels
        self.ids = ids
        self.extras = extras
        self.extra_fieldnames = extra_fieldnames
        self.dimi_matches = dimi_matches
        self.skip_no_dimi_matches = skip_no_dimi_matches
        self.total = len(texts)
        self.processed = 0
        self.status = "queued"
        self.error: str | None = None
        self.metrics: dict[str, str] | None = None
        self.result_bytes: bytes | None = None
        self.result_json: dict[str, object] | None = None


class LemmaBatchJob:
    def __init__(
        self,
        job_id: str,
        texts: list[str],
        language: str,
    ) -> None:
        self.job_id = job_id
        self.texts = texts
        self.language = language
        self.total = len(texts)
        self.processed = 0
        self.status = "queued"
        self.error: str | None = None
        self.result_json: list[dict] | None = None


JOBS: dict[str, BatchJob] = {}
JOB_LOCK = threading.Lock()
LEMMA_JOBS: dict[str, LemmaBatchJob] = {}
LEMMA_JOB_LOCK = threading.Lock()


# ── MORALIZATION MODEL BUNDLE ─────────────────────────────────────────────────


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


def load_prompt_assets(prompt_path: Path) -> tuple[str, str]:
    system_prompt = ""
    user_lines: list[str] = []
    in_user_block = False

    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("system:"):
            system_prompt = json.loads(line.split("system:", 1)[1].strip())
            continue

        if line.startswith("user:"):
            in_user_block = True
            continue

        if line.startswith("output_format:"):
            break

        if in_user_block:
            if line.startswith("    "):
                user_lines.append(line[4:])
            elif line == "":
                user_lines.append("")

    if not system_prompt:
        raise RuntimeError(f"Missing system prompt in {prompt_path}")
    if not user_lines:
        raise RuntimeError(f"Missing user prompt in {prompt_path}")

    return system_prompt, "\n".join(user_lines).rstrip()


def load_response_format(schema_path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("openai_output_format", schema_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load response format from {schema_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_format = getattr(module, "output", None)
    if not isinstance(output_format, dict):
        raise RuntimeError(f"Missing output dict in {schema_path}")

    return output_format


OPENAI_SYSTEM_PROMPT, OPENAI_USER_PROMPT = load_prompt_assets(OPENAI_PROMPT_DIR / "prompt.yaml")
OPENAI_RESPONSE_FORMAT = load_response_format(OPENAI_PROMPT_DIR / "output_format.py")
ANTHROPIC_SYSTEM_PROMPT, ANTHROPIC_USER_PROMPT = load_prompt_assets(
    ANTHROPIC_PROMPT_DIR / "prompt.yaml"
)
ANTHROPIC_RESPONSE_FORMAT = load_response_format(ANTHROPIC_PROMPT_DIR / "output_format.py")


def analyze_with_openai(
    text: str,
    sys_prompt: str,
    user_prompt: str,
    response_format: dict,
    model: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> tuple[dict, float, str]:
    prompt = user_prompt.format(text=text)

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            client = get_openai_client()
            response = client.chat.completions.create(
                model=model,
                seed=42,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format=response_format,
            )
            end_time = time.time()
            generation_time = end_time - start_time
            content = response.choices[0].message.content or "{}"
            try:
                data = parse_json_response(content, source="OpenAI")
            except json.JSONDecodeError:
                print(f"Raw OpenAI response that failed to parse: {content}")
                raise
            return data, generation_time, content
        except Exception as exc:
            if attempt < max_retries - 1:
                print(f"Retrying ({attempt + 1}/{max_retries}) after error: {exc}")
                time.sleep(retry_delay)
            else:
                raise


def analyze_with_anthropic(
    text: str,
    sys_prompt: str,
    user_prompt: str,
    response_format: dict,
    model: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> tuple[dict, float, str]:
    prompt = user_prompt.format(text=text)

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            client = get_anthropic_client()
            response = client.messages.create(
                model=model,
                max_tokens=1500,
                system=sys_prompt,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            end_time = time.time()
            generation_time = end_time - start_time
            content = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
            try:
                data = parse_json_response(content, source="Anthropic")
            except json.JSONDecodeError:
                print(f"Raw Anthropic response that failed to parse: {content}")
                raise
            return data, generation_time, content
        except Exception as exc:
            if attempt < max_retries - 1:
                print(f"Retrying ({attempt + 1}/{max_retries}) after error: {exc}")
                time.sleep(retry_delay)
            else:
                raise


def get_openai_client() -> Any:
    global OPENAI_CLIENT

    if OPENAI_CLIENT is None:
        try:
            openai_module = importlib.import_module("openai")
        except ImportError as exc:
            raise RuntimeError("The openai package is not installed.") from exc

        openai_client_class = getattr(openai_module, "OpenAI", None)
        if openai_client_class is None:
            raise RuntimeError("The openai package does not expose OpenAI.")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        OPENAI_CLIENT = openai_client_class(api_key=api_key)

    return OPENAI_CLIENT


def get_anthropic_client() -> Any:
    global ANTHROPIC_CLIENT

    if ANTHROPIC_CLIENT is None:
        try:
            anthropic_module = importlib.import_module("anthropic")
        except ImportError as exc:
            raise RuntimeError("The anthropic package is not installed.") from exc

        anthropic_client_class = getattr(anthropic_module, "Anthropic", None)
        if anthropic_client_class is None:
            raise RuntimeError("The anthropic package does not expose Anthropic.")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        ANTHROPIC_CLIENT = anthropic_client_class(api_key=api_key)

    return ANTHROPIC_CLIENT


def parse_json_response(content: str, source: str) -> dict:
    cleaned = content.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"{source} response after fence cleanup: {cleaned}")
        raise


def parse_prediction_payload(payload: dict) -> PredictResponse:
    if isinstance(payload.get("moralisierung"), dict):
        moralisierung = payload["moralisierung"]
        contains_moralisierung = bool(moralisierung.get("enthaelt_moralisierung", False))
        return PredictResponse(
            label="moralization" if contains_moralisierung else "no_moralization",
            confidence=1.0 if contains_moralisierung else 0.0,
            explanation=str(moralisierung.get("begruendung", "")) or None,
        )

    label = str(payload.get("label", "no_moralization"))
    confidence_value = payload.get("confidence", 0.0)

    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))
    return PredictResponse(label=label, confidence=round(confidence, 4))


def normalize_model_codes(models: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for model in models:
        code = str(model).strip().lower()
        if not code:
            continue
        if code not in MODEL_COLUMN_SUFFIXES:
            raise ValueError(f"Unknown model: {model}")
        if code in seen:
            continue
        seen.add(code)
        normalized.append(code)

    if not normalized:
        raise ValueError("At least one model must be selected.")

    return normalized


def get_model_suffix(model_code: str) -> str:
    return MODEL_COLUMN_SUFFIXES[model_code]


def run_model_prediction(text: str, model_code: str) -> ModelPredictionResult:
    if model_code == "xlm-roberta":
        prediction = MODEL.predict(text)
        return ModelPredictionResult(
            label=prediction.label,
            confidence=prediction.confidence,
        )

    if model_code == "claude":
        payload, _generation_time, raw_output = analyze_with_anthropic(
            text=text,
            sys_prompt=ANTHROPIC_SYSTEM_PROMPT,
            user_prompt=ANTHROPIC_USER_PROMPT,
            response_format=ANTHROPIC_RESPONSE_FORMAT,
            model=ANTHROPIC_MODEL,
        )
        prediction = parse_prediction_payload(payload)
        return ModelPredictionResult(
            label=prediction.label,
            explanation=prediction.explanation,
            raw_output=raw_output,
        )

    if model_code == "openai":
        payload, _generation_time, raw_output = analyze_with_openai(
            text=text,
            sys_prompt=OPENAI_SYSTEM_PROMPT,
            user_prompt=OPENAI_USER_PROMPT,
            response_format=OPENAI_RESPONSE_FORMAT,
            model=OPENAI_MODEL,
        )
        prediction = parse_prediction_payload(payload)
        return ModelPredictionResult(
            label=prediction.label,
            explanation=prediction.explanation,
            raw_output=raw_output,
        )

    raise ValueError(f"Unknown model: {model_code}")


# ── LEMMA LOADER ──────────────────────────────────────────────────────────────


def load_lemmas(language: str) -> list[str]:
    """
    Read lemmas from the dimi directory.
    Handles both naming conventions present on disk:
      - Moralization-Dictionary_DE-lemmatized.xlsx  (hyphen before 'lemmatized')
      - Moralization-Dictionary_EN_lemmatized.xlsx  (underscore before 'lemmatized')
    """
    lang = language.upper()
    candidates = [
        LEMMAS_DIR / f"Moralization-Dictionary_{lang}-lemmatized.xlsx",
        LEMMAS_DIR / f"Moralization-Dictionary_{lang}_lemmatized.xlsx",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"No lemma file found for language '{language}'. "
            f"Tried: {', '.join(str(p) for p in candidates)}"
        )

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active

    seen: set[str] = set()
    lemmas: list[str] = []
    for row in ws.iter_rows(min_col=1, max_col=1, values_only=True):
        cell = row[0]
        if cell is None:
            continue
        value = str(cell).strip().lower()
        if value and value not in seen:
            seen.add(value)
            lemmas.append(value)

    wb.close()
    return lemmas


# ── LEMMA CACHE (loaded once at startup per language file found) ───────────────

def _preload_lemmas() -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    for lang in ("de", "en", "fr", "it"):
        try:
            cache[lang] = load_lemmas(lang)
        except FileNotFoundError as exc:
            print(f"[lemmas] {exc}")
        except Exception as exc:
            print(f"[lemmas] Could not load '{lang}': {exc}")
    return cache


LEMMA_CACHE: dict[str, list[str]] = _preload_lemmas()





@dataclass
class LemmaMatch:
    sentence_index: int
    matched_lemmas: list[str]
    context_sentences: list[str]
    context_html: list[str]
    center_sentence: str


class LemmatizerBundle:
    """
    Maintains one stanza.Pipeline per language, created lazily on first use.
    Uses package='default' (smallest available model) for every language.
    The cached single-text path serialises access to shared pipelines.
    Batch jobs can opt into their own pipeline instance for parallelism.
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, stanza.Pipeline] = {}
        self._cache_lock = threading.Lock()
        self._build_lock = threading.Lock()
        for lang in LEMMA_CACHE:
            try:
                self._get_pipeline(lang)
                print(f"[stanza] Loaded pipeline for '{lang}'")
            except Exception as exc:
                print(f"[stanza] Could not load pipeline for '{lang}': {exc}")

    # ── private ────────────────────────────────────────────────────────────

    def _build_pipeline(self, language: str) -> stanza.Pipeline:
        with self._build_lock:
            stanza.download(
                language,
                package="default",
                processors="tokenize,pos,lemma",
                logging_level="WARN",
            )
            return stanza.Pipeline(
                language,
                package="default",
                processors="tokenize,pos,lemma",
                logging_level="WARN",
            )

    def _get_pipeline(self, language: str) -> stanza.Pipeline:
        """Return cached pipeline, downloading + building it on first use."""
        if language not in self._pipelines:
            self._pipelines[language] = self._build_pipeline(language)
        return self._pipelines[language]

    @staticmethod
    def _mark_tokens(sentence_text: str, surface_forms: set[str]) -> str:
        """Wrap matched surface forms in <mark> tags (case-insensitive)."""
        if not surface_forms:
            return sentence_text
        pattern = re.compile(
            r"\b("
            + "|".join(re.escape(f) for f in sorted(surface_forms, key=len, reverse=True))
            + r")\b",
            flags=re.IGNORECASE,
        )
        return pattern.sub(r"<mark>\1</mark>", sentence_text)

    def _find_lemmas_with_pipeline(
        self,
        text: str,
        dict_set: set[str],
        pipeline: stanza.Pipeline,
    ) -> LemmatizerResponse:
        doc = pipeline(text)

        sentences = doc.sentences
        total = len(sentences)
        matches: list[LemmaMatch] = []

        for sent_idx, sentence in enumerate(sentences):
            matched_lemmas: set[str] = set()
            matched_surface: set[str] = set()

            for token in sentence.tokens:
                for word in token.words:
                    if word.lemma and word.lemma.lower() in dict_set:
                        matched_lemmas.add(word.lemma.lower())
                        matched_surface.add(word.text)

            if not matched_lemmas:
                continue

            window_start = max(0, sent_idx - 2)
            window_end = min(total, sent_idx + 3)  # exclusive
            context_plain: list[str] = []
            context_html: list[str] = []

            for ctx_idx in range(window_start, window_end):
                ctx_text = sentences[ctx_idx].text
                context_plain.append(ctx_text)
                context_html.append(
                    self._mark_tokens(ctx_text, matched_surface)
                    if ctx_idx == sent_idx
                    else ctx_text
                )

            matches.append(
                LemmaMatch(
                    sentence_index=sent_idx,
                    matched_lemmas=sorted(matched_lemmas),
                    context_sentences=context_plain,
                    context_html=context_html,
                    center_sentence=sentence.text,
                )
            )

        return LemmatizerResponse(
            matches=[
                {
                    "sentence_index": m.sentence_index,
                    "matched_lemmas": m.matched_lemmas,
                    "context_sentences": m.context_sentences,
                    "context_html": m.context_html,
                    "center_sentence": m.center_sentence,
                }
                for m in matches
            ],
            total_sentences=total,
        )

    # ── public ─────────────────────────────────────────────────────────────

    def find_lemmas(
        self,
        text: str,
        language: str,
    ) -> LemmatizerResponse:
        """
        Lemmatise *text* and return every sentence whose tokens include a lemma
        from the pre-loaded dictionary for *language*, together with ±2 sentences
        of context.

        Parameters
        ----------
        text:
            Raw input text.
        language:
            ISO 639-1 code (e.g. ``"de"``, ``"en"``). A lemma file for this
            language must exist in ``LEMMAS_DIR``.
        """
        if language not in LEMMA_CACHE:
            raise ValueError(
                f"No lemma file found for language '{language}'. "
                f"Expected: {LEMMAS_DIR / f'lemmas_{language}.xlsx'}"
            )
        dict_set = set(LEMMA_CACHE[language])  # already lowercased
        with self._cache_lock:
            pipeline = self._get_pipeline(language)
            return self._find_lemmas_with_pipeline(text, dict_set, pipeline)

    def find_lemmas_batch_isolated(
        self,
        texts: list[str],
        language: str,
        on_progress: Callable[[int], None] | None = None,
    ) -> list[LemmatizerResponse]:
        if language not in LEMMA_CACHE:
            raise ValueError(
                f"No lemma file found for language '{language}'. "
                f"Expected: {LEMMAS_DIR / f'lemmas_{language}.xlsx'}"
            )
        dict_set = set(LEMMA_CACHE[language])  # already lowercased
        pipeline = self._build_pipeline(language)
        results: list[LemmatizerResponse] = []
        for index, text in enumerate(texts, start=1):
            results.append(self._find_lemmas_with_pipeline(text, dict_set, pipeline))
            if on_progress:
                on_progress(index)
        return results

    def find_lemmas_batch(
        self,
        texts: list[str],
        language: str,
        on_progress: Callable[[int], None] | None = None,
    ) -> list[LemmatizerResponse]:
        if language not in LEMMA_CACHE:
            raise ValueError(
                f"No lemma file found for language '{language}'. "
                f"Expected: {LEMMAS_DIR / f'lemmas_{language}.xlsx'}"
            )
        dict_set = set(LEMMA_CACHE[language])  # already lowercased
        with self._cache_lock:
            pipeline = self._get_pipeline(language)
            results: list[LemmatizerResponse] = []
            for index, text in enumerate(texts, start=1):
                results.append(self._find_lemmas_with_pipeline(text, dict_set, pipeline))
                if on_progress:
                    on_progress(index)
            return results


# ── HELPERS ───────────────────────────────────────────────────────────────────


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


def parse_bool_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0

    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def parse_csv_texts(
    raw_bytes: bytes,
) -> tuple[
    list[str],
    list[int] | None,
    list[str],
    list[dict[str, object]],
    list[str],
    list[int],
]:
    content = raw_bytes.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(content))
    texts: list[str] = []
    labels: list[int] = []
    ids: list[str] = []
    extras: list[dict[str, object]] = []
    extra_fieldnames: list[str] = []
    dimi_matches: list[int] = []

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
        no_dimi_field = next(
            (
                reader.fieldnames[idx]
                for idx, name in enumerate(normalized)
                if name == "no_dimi_match"
            ),
            None,
        )
        dimi_matches_field = next(
            (
                reader.fieldnames[idx]
                for idx, name in enumerate(normalized)
                if name == "dimi_matches"
            ),
            None,
        )
    else:
        text_field = None
        label_field = None
        id_field = None
        no_dimi_field = None
        dimi_matches_field = None

    if text_field:
        excluded_fields = {text_field}
        if label_field:
            excluded_fields.add(label_field)
        if id_field:
            excluded_fields.add(id_field)
        if no_dimi_field:
            excluded_fields.add(no_dimi_field)
        if dimi_matches_field:
            excluded_fields.add(dimi_matches_field)

        if reader.fieldnames:
            extra_fieldnames = [name for name in reader.fieldnames if name not in excluded_fields]

        for row in reader:
            text = str(row.get(text_field, "")).strip()
            if text:
                texts.append(text)
                if id_field:
                    ids.append(str(row.get(id_field, "")).strip())
                extras.append({name: row.get(name, "") for name in extra_fieldnames})
                dimi_matches_value = 0
                if dimi_matches_field:
                    raw_dimi_matches = row.get(dimi_matches_field, 0)
                    try:
                        dimi_matches_value = int(str(raw_dimi_matches).strip() or "0")
                    except ValueError:
                        dimi_matches_value = 0
                elif no_dimi_field and parse_bool_flag(row.get(no_dimi_field, False)):
                    dimi_matches_value = 0
                dimi_matches.append(dimi_matches_value)
            if label_field:
                raw_label = row.get(label_field, "")
                if raw_label in {None, ""}:
                    raise ValueError("Label column is present but has empty values.")
                labels.append(parse_label(raw_label))
        if not id_field:
            ids = [str(index) for index in range(1, len(texts) + 1)]
        return texts, labels if label_field else None, ids, extras, extra_fieldnames, dimi_matches

    fallback_reader = csv.reader(io.StringIO(content))
    for row in fallback_reader:
        if not row:
            continue
        text = str(row[0]).strip()
        if text:
            texts.append(text)
    ids = [str(index) for index in range(1, len(texts) + 1)]
    extras = [{} for _ in texts]
    dimi_matches = [0 for _ in texts]
    return texts, None, ids, extras, [], dimi_matches


def parse_json_texts(
    raw_bytes: bytes,
) -> tuple[
    list[str],
    list[int] | None,
    list[str],
    list[dict[str, object]],
    list[str],
    list[int],
]:
    content = raw_bytes.decode("utf-8", errors="ignore")
    payload = json.loads(content)
    texts: list[str] = []
    labels: list[int] = []
    ids: list[str | None] = []
    extras: list[dict[str, object]] = []
    extra_fieldnames: list[str] = []
    dimi_matches: list[int] = []

    if not isinstance(payload, list):
        raise ValueError("JSON must be a list of objects with a text field.")

    label_present = False
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each JSON item must be an object with a text field.")

        text = str(item.get("text", "")).strip()
        if text:
            texts.append(text)
            ids.append(str(item.get("id")).strip() if "id" in item else None)
            extra_item = {
                key: value
                for key, value in item.items()
                if key not in {"text", "label", "id", "no_dimi_match", "dimi_matches"}
            }
            extras.append(extra_item)
            for key in extra_item.keys():
                if key not in extra_fieldnames:
                    extra_fieldnames.append(key)
            raw_dimi_matches = item.get("dimi_matches", 0)
            try:
                dimi_matches.append(int(str(raw_dimi_matches).strip() or "0"))
            except ValueError:
                dimi_matches.append(0)
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

    if len(dimi_matches) != len(texts):
        dimi_matches = [0 for _ in texts]

    return (
        texts,
        labels if label_present else None,
        resolved_ids,
        extras,
        extra_fieldnames,
        dimi_matches,
    )


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
    metric_true_labels: list[int] = []
    metric_predicted_labels: list[int] = []
    reserved_fields = {
        "id",
        "text",
        "full_text",
        "label",
        "dimi_matches",
        "dimi_matched_lemmas",
        "no_dimi_match",
    }
    for model_code in job.models:
        suffix = get_model_suffix(model_code)
        reserved_fields.update(
            {
                f"prediction_{suffix}",
                f"confidence_{suffix}",
                f"explanation_{suffix}",
                f"raw_output_{suffix}",
            }
        )
    output_extra_fieldnames = [
        name for name in job.extra_fieldnames if name not in reserved_fields
    ]
    metric_model_code = job.models[0] if len(job.models) == 1 else None

    for index, text in enumerate(job.texts, start=1):
        dimi_match_count = job.dimi_matches[index - 1] if index - 1 < len(job.dimi_matches) else 0
        skip_no_dimi = job.skip_no_dimi_matches and dimi_match_count == 0

        input_extras = job.extras[index - 1]
        result_item: dict[str, object] = {
            "id": job.ids[index - 1],
            "text": text,
        }
        if job.labels:
            result_item["label"] = (
                "moralization" if job.labels[index - 1] == 1 else "no_moralization"
            )

        for model_code in job.models:
            suffix = get_model_suffix(model_code)
            result_item[f"prediction_{suffix}"] = "no_dimi"
            if model_code == "xlm-roberta":
                result_item[f"confidence_{suffix}"] = None
            else:
                result_item[f"explanation_{suffix}"] = None
                result_item[f"raw_output_{suffix}"] = None

        if not skip_no_dimi:
            for model_code in job.models:
                suffix = get_model_suffix(model_code)
                prediction = run_model_prediction(text, model_code)
                result_item[f"prediction_{suffix}"] = prediction.label
                if model_code == "xlm-roberta":
                    result_item[f"confidence_{suffix}"] = (
                        f"{prediction.confidence:.4f}" if prediction.confidence is not None else None
                    )
                else:
                    result_item[f"explanation_{suffix}"] = prediction.explanation
                    result_item[f"raw_output_{suffix}"] = prediction.raw_output

                if job.labels and model_code == metric_model_code:
                    metric_true_labels.append(job.labels[index - 1])
                    metric_predicted_labels.append(label_to_binary(prediction.label))

        if "full_text" in input_extras:
            result_item["full_text"] = input_extras.get("full_text", text)
        # Always include the parsed dimi match count from the job (falls back to 0)
        result_item["dimi_matches"] = dimi_match_count
        if "dimi_matched_lemmas" in input_extras:
            result_item["dimi_matched_lemmas"] = input_extras.get("dimi_matched_lemmas", "")

        for fieldname in output_extra_fieldnames:
            result_item[fieldname] = input_extras.get(fieldname, "")

        results.append(result_item)
        job.processed = index

    if job.labels and metric_true_labels:
        job.metrics = compute_metrics(metric_true_labels, metric_predicted_labels)
    elif job.labels:
        job.metrics = None

    if job.output_format == "json":
        job.result_json = {"results": results}
    else:
        output = io.StringIO()
        if job.labels:
            fieldnames = ["id", "text"]
            if "full_text" in job.extra_fieldnames:
                fieldnames.append("full_text")
            # include dimi_matches column when dimi match counts were provided
            if job.dimi_matches is not None:
                fieldnames.append("dimi_matches")
            if "dimi_matched_lemmas" in job.extra_fieldnames:
                fieldnames.append("dimi_matched_lemmas")
            fieldnames.append("label")
            for model_code in job.models:
                suffix = get_model_suffix(model_code)
                fieldnames.append(f"prediction_{suffix}")
                if model_code == "xlm-roberta":
                    fieldnames.append(f"confidence_{suffix}")
                else:
                    fieldnames.extend([f"explanation_{suffix}", f"raw_output_{suffix}"])
            fieldnames.extend(output_extra_fieldnames)
        else:
            fieldnames = ["id", "text"]
            if "full_text" in job.extra_fieldnames:
                fieldnames.append("full_text")
            # include dimi_matches column when dimi match counts were provided
            if job.dimi_matches is not None:
                fieldnames.append("dimi_matches")
            if "dimi_matched_lemmas" in job.extra_fieldnames:
                fieldnames.append("dimi_matched_lemmas")
            for model_code in job.models:
                suffix = get_model_suffix(model_code)
                fieldnames.append(f"prediction_{suffix}")
                if model_code == "xlm-roberta":
                    fieldnames.append(f"confidence_{suffix}")
                else:
                    fieldnames.extend([f"explanation_{suffix}", f"raw_output_{suffix}"])
            fieldnames.extend(output_extra_fieldnames)

        deduped_fieldnames: list[str] = []
        for fieldname in fieldnames:
            if fieldname not in deduped_fieldnames:
                deduped_fieldnames.append(fieldname)
        writer = csv.DictWriter(output, fieldnames=deduped_fieldnames)
        writer.writeheader()
        writer.writerows(results)
        job.result_bytes = output.getvalue().encode("utf-8")

    job.status = "completed"


def run_lemma_batch_job(job: LemmaBatchJob) -> None:
    job.status = "running"

    def _update_progress(value: int) -> None:
        job.processed = value

    try:
        results = LEMMATIZER.find_lemmas_batch_isolated(
            texts=job.texts,
            language=job.language,
            on_progress=_update_progress,
        )
        job.result_json = [result.dict() for result in results]
        job.status = "completed"
    except Exception as exc:
        job.error = str(exc)
        job.status = "failed"


# ── SINGLETONS ────────────────────────────────────────────────────────────────

try:
    MODEL = ModelBundle(MODEL_DIR)
except Exception as exc:
    MODEL = None
    MODEL_ERROR = exc
else:
    MODEL_ERROR = None

try:
    LEMMATIZER = LemmatizerBundle()
except Exception as exc:
    LEMMATIZER = None
    LEMMATIZER_ERROR = exc
else:
    LEMMATIZER_ERROR = None


# ── MORALIZATION ENDPOINTS ────────────────────────────────────────────────────


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
        selected_model = request.model.strip().lower()
        prediction = run_model_prediction(request.text, selected_model)
        return PredictResponse(
            label=prediction.label,
            confidence=prediction.confidence if prediction.confidence is not None else 0.0,
            explanation=prediction.explanation,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/batch/start")
async def batch_start(
    file: UploadFile = File(...),
    output_format: str = Form("csv"),
    models: list[str] = Form(["xlm-roberta"]),
    skip_no_dimi_matches: str = Form("false"),
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
            texts, labels, ids, extras, extra_fieldnames, dimi_matches = parse_csv_texts(raw_bytes)
        elif filename.endswith(".json") or content_type in {"application/json", "text/json"}:
            texts, labels, ids, extras, extra_fieldnames, dimi_matches = parse_json_texts(raw_bytes)
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

    try:
        selected_models = normalize_model_codes(models)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    skip_flag = parse_bool_flag(skip_no_dimi_matches)

    job_id = str(uuid.uuid4())
    job = BatchJob(
        job_id=job_id,
        output_format=fmt,
        models=selected_models,
        texts=texts,
        labels=labels,
        ids=ids,
        extras=extras,
        extra_fieldnames=extra_fieldnames,
        dimi_matches=dimi_matches,
        skip_no_dimi_matches=skip_flag,
    )
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


# ── LEMMATIZER ENDPOINTS ──────────────────────────────────────────────────────


@app.get("/lemmas/{language}")
async def get_lemmas(language: str) -> dict[str, object]:
    """Return the loaded lemma list for *language* (for frontend use)."""
    if language not in LEMMA_CACHE:
        raise HTTPException(
            status_code=404,
            detail=f"No lemma file found for language '{language}'.",
        )
    return {"language": language, "lemmas": LEMMA_CACHE[language]}


@app.post("/lemmatize", response_model=LemmatizerResponse)
async def lemmatize(request: LemmatizerRequest) -> LemmatizerResponse:
    """
    Find all sentences in *text* whose tokens match a lemma in the
    pre-loaded dictionary for *language*.  Returns each match with ±2
    sentences of context; matched tokens are wrapped in ``<mark>`` tags
    inside ``context_html``.
    """
    if LEMMATIZER_ERROR or LEMMATIZER is None:
        raise HTTPException(status_code=500, detail="Lemmatizer failed to load")

    try:
        return LEMMATIZER.find_lemmas(
            text=request.text,
            language=request.language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/lemmatize/batch")
async def lemmatize_batch(request: LemmaBatchRequest) -> list[dict]:
    """
    Run lemma search over multiple texts (same language for all).
    Returns one ``LemmatizerResponse`` payload per input text, in the same order.
    """
    if LEMMATIZER_ERROR or LEMMATIZER is None:
        raise HTTPException(status_code=500, detail="Lemmatizer failed to load")

    try:
        return [
            result.dict()
            for result in LEMMATIZER.find_lemmas_batch(
                texts=request.texts,
                language=request.language,
            )
        ]
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/lemmatize/batch/start")
async def lemmatize_batch_start(request: LemmaBatchRequest) -> dict[str, str]:
    if LEMMATIZER_ERROR or LEMMATIZER is None:
        raise HTTPException(status_code=500, detail="Lemmatizer failed to load")

    job_id = str(uuid.uuid4())
    job = LemmaBatchJob(
        job_id=job_id,
        texts=request.texts,
        language=request.language,
    )
    with LEMMA_JOB_LOCK:
        LEMMA_JOBS[job_id] = job

    thread = threading.Thread(target=run_lemma_batch_job, args=(job,), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/lemmatize/batch/status/{job_id}")
async def lemmatize_batch_status(job_id: str) -> dict[str, object]:
    with LEMMA_JOB_LOCK:
        job = LEMMA_JOBS.get(job_id)
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

    return payload


@app.get("/lemmatize/batch/result/{job_id}")
async def lemmatize_batch_result(job_id: str) -> JSONResponse:
    with LEMMA_JOB_LOCK:
        job = LEMMA_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Job not completed")
    if job.result_json is None:
        raise HTTPException(status_code=500, detail="Missing batch result")

    return JSONResponse(content=job.result_json)