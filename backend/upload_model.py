from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "xlm-roberta").strip().lower()
MODEL_DIRECTORIES = {
    "xlm-roberta": ROOT_DIR
    / "models"
    / "FacebookAI-xlm-roberta-base-finetuned-base_params"
    / "checkpoint-2500",
    "mmbert": ROOT_DIR
    / "models"
    / "jhu-clsp-mmBERT-base-sft-morcorp-10epochs"
    / "checkpoint-4400",
}
if MODEL_VARIANT not in MODEL_DIRECTORIES:
    raise RuntimeError("MODEL_VARIANT must be one of: xlm-roberta, mmbert")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", MODEL_DIRECTORIES[MODEL_VARIANT]))
HANDLER_PATH = Path(__file__).with_name("handler.py")
REPO_ID = os.environ.get("HF_MODEL_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not REPO_ID:
    raise RuntimeError("Set HF_MODEL_REPO, for example: your-account/moralization-xlm-roberta")
if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN to a Hugging Face write token.")
if not MODEL_DIR.is_dir():
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
if not HANDLER_PATH.is_file():
    raise FileNotFoundError(f"Custom handler not found: {HANDLER_PATH}")

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=REPO_ID, repo_type="model", private=True, exist_ok=True)
api.upload_folder(
    repo_id=REPO_ID,
    repo_type="model",
    folder_path=MODEL_DIR,
    commit_message=f"Upload fine-tuned {MODEL_VARIANT} sequence classifier",
)
api.upload_file(
    path_or_fileobj=str(HANDLER_PATH),
    path_in_repo="handler.py",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add custom Inference Endpoint handler",
)
print(f"Uploaded {MODEL_DIR} to https://huggingface.co/{REPO_ID}")