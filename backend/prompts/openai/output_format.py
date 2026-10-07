from pydantic import BaseModel
from typing import List

class MoralValue(BaseModel):
    text: str
    moral_foundations_theory_categories: List[str]

class Protagonist(BaseModel):
    text: str
    category: str
    roles: List[str]

class Moralization(BaseModel):
    moral_values: List[MoralValue]
    demand: str
    rationale: str
    contains_moralization: bool

class MoralizationOutput(BaseModel):
    moralization: Moralization
    protagonists: List[Protagonist]


output = {
    "type": "json_schema",
    "json_schema": {
        "name": "MoralizationOutput",
        "schema": {
            "type": "object",
            "properties": {
                "moralization": {
                    "type": "object",
                    "properties": {
                        "moral_values": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                        "moral_foundations_theory_categories": {"type": "array", "items": {"type": "string", "enum": [
                                            "Care", "Harm",
                                            "Fairness", "Cheating",
                                            "Loyalty", "Betrayal",
                                            "Authority", "Subversion",
                                            "Purity", "Degradation",
                                            "Liberty", "Oppression"
                                        ]}}
                                },
                                "required": ["text", "moral_foundations_theory_categories"],
                                "additionalProperties": False
                            }
                        },
                        "demand": {"type": "string"},
                        "rationale": {"type": "string"},
                        "contains_moralization": {"type": "boolean"}
                    },
                    "required": ["moral_values", "demand", "rationale", "contains_moralization"],
                    "additionalProperties": False,
                },
                "protagonists": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "category": {"type": "string", "enum": [
                                "Individuals",
                                "Institutions",
                                "Social Groups",
                                "Generic Human",
                                "OTHER"
                            ]},
                            "roles": {"type": "array", "items": {"type": "string", "enum": [
                                "Demander",
                                "Adressee",
                                "Beneficiary",
                                "Maleficiary",
                                "Unclear",
                                "NONE"
                            ]}},
                        },
                        "required": ["text", "category", "roles"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["moralization", "protagonists"],
            "additionalProperties": False,
        },
        "strict": True,
    }}