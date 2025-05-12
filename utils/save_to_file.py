import csv
import os
import json

from utils.utils import test_answer_path


def save_to_csv(prompt, answer, filename='../data/generated_answer.csv'):
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['prompt', 'answer'])

        if not file_exists:
            writer.writeheader()

        writer.writerow({'prompt': prompt, 'answer': answer})


def save_to_jsonl(final_output, file_path=test_answer_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        json_line = json.dumps(final_output, ensure_ascii=False)
        f.write(json_line + '\n')
        # f.flush()
        # f.close()


if __name__ == "__main__":
    save_to_csv("What is AI?", "AI stands for Artificial Intelligence.")
