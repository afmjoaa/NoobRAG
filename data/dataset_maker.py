import os
import json
from torch.utils.data import Dataset, DataLoader


class JsonlDataset(Dataset):
    def __init__(self, directory):
        self.samples = []
        self._load_jsonl_files(directory)

    def _load_jsonl_files(self, directory):
        for filename in os.listdir(directory):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Modify this based on your data structure
        return sample

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def save_to_jsonl(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample) + '\n')


if __name__ == "__main__":
    # Usage
    directory = "./generated"
    dataset = JsonlDataset(directory)

    # Get dataloader
    # dataloader = dataset.get_dataloader(batch_size=32)

    # Save dataset to a single JSONL file
    output_path = "combined_data.jsonl"
    dataset.save_to_jsonl(output_path)
    print(f"{output_path} is created, Thank you !!!")
