import pandas as pd
import jsonschema
from jsonschema import validate
import json

# LiveRAG Answer JSON schema:
json_schema = """
{ 
"$schema": "http://json-schema.org/draft-07/schema#", 

  "title": "Answer file schema", 
  "type": "object", 
  "properties": { 
    "id": { 
      "type": "integer", 
      "description": "Question ID" 
    }, 
    "question": { 
      "type": "string", 
      "description": "The question" 
    }, 
    "passages": { 
      "type": "array", 
      "description": "Passages used and related FineWeb doc IDs, ordered by decreasing importance", 
      "items": { 
        "type": "object", 
        "properties": {
          "passage": { 
            "type": "string", 
            "description": "Passage text" 
          }, 
          "doc_IDs": {
            "type": "array", 
            "description": "Passage related FineWeb doc IDs, ordered by decreasing importance", 
            "items": { 
              "type": "string", 
              "description": "FineWeb doc ID, e.g., <urn:uuid:d69cbebc-133a-4ebe-9378-68235ec9f091>"
            } 
          } 
        },
        "required": ["passage", "doc_IDs"]
      }
    }, 
    "final_prompt": {
      "type": "string",
      "description": "Final prompt, as submitted to Falcon LLM"
    },
    "answer": {
      "type": "string",
      "description": "Your answer"
    }
  },
  "required": ["id", "question", "passages", "final_prompt", "answer"]
}
"""

# Code that generates the output
answers = pd.DataFrame({
    "id": [1, 2],
    "question": ["What is the capital of France?", "What is the capital of Germany?"],
    "passages": [
        [
            {"passage": "Paris is the capital and most populous city of France.",
             "doc_IDs": ["<urn:uuid:1234abcd-5678-efgh-9101-ijklmnopqrst>", "<urn:uuid:1234abcd-5678-efgh-9202-ijklmnopqrst>"]},
            {"passage": "France is located in Western Europe.",
             "doc_IDs": ["<urn:uuid:1234abcd-5678-efgh-9101-ijklmnopqrst>"]}
        ],
        [
            {"passage": "Berlin is the capital of Germany.",
             "doc_IDs": ["<urn:uuid:1234abcd-5678-efgh-9101-ijklmnopqrst>"]}
        ]
    ],
    "final_prompt": [
        "Using the following - Paris is the capital and most populous city of France - and - France is located in Western Europe - answer the question: What is the capital of France?",
        "Using the following - Berlin is the capital of Germany - answer the question: What is the capital of Germany?"
    ],
    "answer": ["Paris", "Berlin"]
})

# Convert to JSON format
answers_json = answers.to_json(orient='records', lines=True, force_ascii=False)

# Or just save to a file
answers.to_json("answers.jsonl", orient='records', lines=True, force_ascii=False)

# Load the file to make sure it is ok
loaded_answers = pd.read_json("answers.jsonl", lines=True)

# Load the JSON schema
schema = json.loads(json_schema)

# Validate each Answer JSON object against the schema
for answer in loaded_answers.to_dict(orient='records'):
    try:
        validate(instance=answer, schema=schema)
        print(f"Answer {answer['id']} is valid.")
    except jsonschema.exceptions.ValidationError as e:
        print(f"Answer {answer['id']} is invalid: {e.message}")