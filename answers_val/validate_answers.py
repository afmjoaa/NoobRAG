import json
import jsonschema
import re

# Load the schema from answer_validator.py
with open('answer_validator.py', 'r', encoding='utf-8') as file:
    validator_code = file.read()

schema_match = re.search(r'json_schema\s*=\s*"""(.*?)"""', validator_code, re.DOTALL)
if schema_match:
    schema_str = schema_match.group(1)
    schema = json.loads(schema_str)
else:
    raise ValueError("JSON schema not found in validator file.")

# Load answers from JSONL file
answers = []
with open('answers.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        answers.append(json.loads(line.strip()))

# Validate each answer
required_keys = {"id", "question", "passages", "final_prompt", "answer"}

for idx, answer in enumerate(answers):
    try:
        missing_keys = required_keys - answer.keys()
        if missing_keys:
            print(f"Answer {idx} is missing required keys: {missing_keys}")
            continue

        empty_keys = [key for key in required_keys if not answer.get(key)]
        if empty_keys:
            print(f"Answer {idx} has empty values for keys: {empty_keys}")
            continue

        jsonschema.validate(instance=answer, schema=schema)
        print(f"Answer {idx} is valid.")
    except jsonschema.ValidationError as e:
        print(f"Answer {idx} is invalid: {e.message}")
    except Exception as e:
        print(f"Unexpected error validating answer {idx}: {e}")