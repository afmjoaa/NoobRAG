import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import sent_tokenize
import random

# ------------------------------
# 1. Reranker Model: MixedBread Reranker
# ------------------------------

class MixedBreadReranker(nn.Module):
    def __init__(self, model_name="mixedbread-ai/mxbai-rerank-large-v1"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.scorer = nn.Linear(self.hidden_size, 1)

    def forward(self, query, sentences):
        """
        MixedBread expects inputs like: 'query: ... passage: ...'
        """
        inputs = [f"query: {query} passage: {sent}" for sent in sentences]
        tokenized = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        tokenized = {k: v.to(next(self.parameters()).device) for k, v in tokenized.items()}
        output = self.encoder(**tokenized)
        cls_emb = output.last_hidden_state[:, 0, :]  # [CLS] token embedding
        scores = self.scorer(cls_emb).squeeze(-1)
        return scores  # shape: (batch_size,)

# ------------------------------
# 2. Differentiable Top-k (Gumbel Softmax Approximation)
# ------------------------------

def differentiable_top_k(scores, max_k, tau=0.5):
    probs = F.gumbel_softmax(scores, tau=tau, hard=False)
    sorted_probs, indices = torch.sort(probs, descending=True)
    mask = torch.zeros_like(probs)
    for i in range(max_k):
        mask[indices[i]] = sorted_probs[i]
    return mask / (mask.sum() + 1e-9)

# ------------------------------
# 3. Custom Dataset with Metadata
# ------------------------------

class RerankerDataset(Dataset):
    def __init__(self, data_samples):
        self.samples = []
        for sample in data_samples:
            query = sample["query"]
            doc = sample["document"]
            teacher_scores = sample["teacher_scores"]

            sentences = sent_tokenize(doc)
            for i, sent in enumerate(sentences):
                self.samples.append({
                    "query": query,
                    "sentence": sent,
                    "teacher_score": teacher_scores[i] if i < len(teacher_scores) else random.uniform(0, 1),
                    "metadata": {
                        "sentence_index": i,
                        "document": doc
                    }
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    return {
        "queries": [x["query"] for x in batch],
        "sentences": [x["sentence"] for x in batch],
        "teacher_scores": torch.tensor([x["teacher_score"] for x in batch], dtype=torch.float32),
        "metadata": [x["metadata"] for x in batch]
    }

# ------------------------------
# 4. Training Loop
# ------------------------------

def train_reranker(model, dataset, max_k=3, tau=0.5, epochs=3, batch_size=8, lr=2e-5, device='cuda'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            query = batch["queries"][0]  # assume batch shares the same query
            sentences = batch["sentences"]
            teacher_scores = batch["teacher_scores"].to(device)

            optimizer.zero_grad()
            pred_scores = model(query, sentences)
            mask = differentiable_top_k(pred_scores, max_k, tau)
            loss = F.mse_loss(pred_scores * mask, teacher_scores * mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {epoch_loss / len(dataloader):.4f}")

# ------------------------------
# 5. Inference with Metadata
# ------------------------------

def rerank_inference(model, query, document, max_k=3, tau=0.5):
    sentences = sent_tokenize(document)
    metadata = [{"sentence_index": i, "document": document} for i in range(len(sentences))]
    model.eval()
    with torch.no_grad():
        scores = model(query, sentences)
        mask = differentiable_top_k(scores, max_k, tau)
        _, indices = torch.topk(mask, max_k)
        selected_sents = [sentences[i] for i in indices.tolist()]
        selected_meta = [metadata[i] for i in indices.tolist()]
    return selected_sents, selected_meta

# ------------------------------
# 6. Run Example
# ------------------------------

if __name__ == "__main__":
    # Dummy example data
    example_data = [
        {
            "query": "What are hybrid retrieval systems?",
            "document": ("Hybrid systems combine sparse and dense retrieval. "
                         "They are useful in large-scale QA. "
                         "Some models use rerankers to select top documents."),
            "teacher_scores": [0.8, 0.6, 0.9]
        },
        {
            "query": "What is Gumbel softmax?",
            "document": ("Gumbel softmax allows differentiable sampling. "
                         "It helps in training discrete selections. "
                         "Widely used in end-to-end reranking tasks."),
            "teacher_scores": [0.7, 0.85, 0.75]
        }
    ]

    dataset = RerankerDataset(example_data)
    model = MixedBreadReranker()
    train_reranker(model, dataset, max_k=2, tau=0.7, epochs=3, batch_size=4, lr=2e-5, device='cuda' if torch.cuda.is_available() else 'cpu')

    print("\n--- Inference Example ---")
    q = "Explain Gumbel softmax in reranking."
    d = ("Gumbel softmax allows smooth approximations of argmax. "
         "This is crucial for end-to-end training. "
         "It is used in models like Gumbel Rerank.")
    sents, meta = rerank_inference(model, q, d)
    print("Selected Sentences:")
    for s in sents:
        print("•", s)
    print("\nMetadata:")
    for m in meta:
        print(m)



# Need to select the model for reranking (Small model = Finetune)
# mxbai-rerank-large-v2
# Need to select model for generating relavance score (Big Model = Inference)

# reranker is giving a score to each sentence.
# so for loss we can use this score or we may use the last generation of the model??
# which one should we use ??
