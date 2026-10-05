from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EndpointHandler:
    def __init__(self, path: str = "") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(path)
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