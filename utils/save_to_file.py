import csv
import os


def save_to_csv(prompt, answer, filename='../data/generated_answer.csv'):
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['prompt', 'answer'])

        if not file_exists:
            writer.writeheader()

        writer.writerow({'prompt': prompt, 'answer': answer})


if __name__ == "__main__":
    save_to_csv("What is AI?", "AI stands for Artificial Intelligence.")
