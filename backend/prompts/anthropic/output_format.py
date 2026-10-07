output = {
  "name": "MoralizationOutput",
  "input_schema": {
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
                "text": { "type": "string" },
                "moral_foundations_theory_categories": {
                  "type": "array",
                  "items": {
                    "type": "string",
                    "enum": [
                      "Care", "Harm",
                      "Fairness", "Cheating",
                      "Loyalty", "Betrayal",
                      "Authority", "Subversion",
                      "Purity", "Degradation",
                      "Liberty", "Oppression"
                    ]
                  }
                }
              },
              "required": ["text", "moral_foundations_theory_categories"]
            }
          },
          "demand": { "type": "string" },
          "rationale": { "type": "string" },
          "contains_moralization": { "type": "boolean" }
        },
        "required": ["moral_values", "demand", "rationale", "contains_moralization"]
      },
      "protagonists": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "text": { "type": "string" },
            "category": {
              "type": "string",
              "enum": ["Individuals", "Institutions", "Social Groups", "Generic Human", "OTHER"]
            },
            "roles": {
              "type": "array",
              "items": {
                "type": "string",
                "enum": [
                  "Demander", "Adressee",
                  "Beneficiary", "Maleficiary",
                  "Unclear", "NONE"
                ]
              }
            }
          },
          "required": ["text", "category", "roles"]
        }
      }
    },
    "required": ["moralization", "protagonists"]
  }
}