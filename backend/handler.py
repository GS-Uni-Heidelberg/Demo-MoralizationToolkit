from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer(path: str) -> Any:
    tokenizer_config_path = Path(path) / "tokenizer_config.json"
    if tokenizer_config_path.is_file():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        if tokenizer_config.get("tokenizer_class") == "TokenizersBackend":
            tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
            return PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                bos_token=tokenizer_config.get("bos_token", "<bos>"),
                eos_token=tokenizer_config.get("eos_token", "<eos>"),
                cls_token=tokenizer_config.get("cls_token", "<bos>"),
                sep_token=tokenizer_config.get("sep_token", "<eos>"),
                pad_token=tokenizer_config.get("pad_token", "<pad>"),
                mask_token=tokenizer_config.get("mask_token", "<mask>"),
                unk_token=tokenizer_config.get("unk_token", "<unk>"),
                model_max_length=tokenizer_config.get("model_max_length", 8192),
            )

    return AutoTokenizer.from_pretrained(path)


class EndpointHandler:
    def __init__(self, path: str = "") -> None:
        self.tokenizer = load_tokenizer(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()
        self.id2label = {
            int(index): str(label)
            for index, label in self.model.config.id2label.items()
        }

    def __call__(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        inputs = data.get("inputs")
        if not isinstance(inputs, str) or not inputs.strip():
            raise ValueError("The 'inputs' field must contain non-empty text.")

        encoded = self.tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**encoded).logits, dim=-1)[0]

        return [
            {
                "label": self.id2label.get(index, str(index)),
                "score": float(probability),
            }
            for index, probability in enumerate(probabilities.tolist())
        ]