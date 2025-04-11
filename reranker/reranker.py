import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer
import random
from nltk.tokenize import sent_tokenize


# ------------------------------
# 1. Reranker Model Architecture
# ------------------------------

class RerankerModel(nn.Module):
    def __init__(self, pretrained_model_name='distilbert-base-uncased'):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(pretrained_model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size
        # A simple scoring head: maps the [CLS] embedding to a scalar score.
        self.scorer = nn.Linear(hidden_size, 1)

    def forward(self, query, sentences):
        """
        query: a string (e.g., "what is ...")
        sentences: a list of strings; each corresponds to one candidate sentence.
        Returns:
            scores: Tensor of shape (N,) where N is the number of sentences.
        """
        # Concatenate query with each candidate sentence.
        # We use a simple "[SEP]" token to separate query and sentence.
        # (Feel free to experiment with different input formats.)
        joint_inputs = [query + " [SEP] " + s for s in sentences]
        inputs = self.tokenizer(joint_inputs, padding=True, truncation=True, return_tensors="pt")
        outputs = self.encoder(**inputs)
        # We take the embedding corresponding to the first token ([CLS]-like token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # shape: (N, hidden_size)
        scores = self.scorer(cls_embeddings).squeeze(1)  # shape: (N,)
        return scores


# ------------------------------
# 2. Differentiable Gumbel-based Top-k Selector
# ------------------------------

def differentiable_top_k(scores, max_k, tau=0.5):
    """
    Given a score vector, apply gumbel_softmax to get a differentiable selection mask.
    For demonstration, we use the soft probabilities directly to weight scores.

    Arguments:
        scores: Tensor of shape (N,)
        max_k: Number of sentences to (approximately) select.
        tau: Temperature for gumbel_softmax.

    Returns:
        soft_mask: Tensor of shape (N,) that approximates a top-k selection.
    """
    # Get a soft probability distribution over the N sentences.
    prob_dist = F.gumbel_softmax(scores, tau=tau, hard=False)
    # For an approximate Top-k, we can zero out all but the top max_k probabilities.
    sorted_probs, sorted_indices = torch.sort(prob_dist, descending=True)
    soft_mask = torch.zeros_like(prob_dist)
    # Instead of a hard binary mask, we keep the original soft probabilities for top-k.
    for i in range(max_k):
        soft_mask[sorted_indices[i]] = sorted_probs[i]
    # Normalize the mask so that the sum is 1 (optional; depends on your loss design)
    soft_mask = soft_mask / (soft_mask.sum() + 1e-9)
    return soft_mask


# ------------------------------
# 3. Custom Dataset (with metadata)
# ------------------------------

class RerankerDataset(Dataset):
    """
    Each training sample in the dataset is assumed to be a dictionary with:
      - "query": a query string,
      - "document": a text string from which we derive sentences,
      - "teacher_scores": a list of floats (combined teacher scores) for each sentence,

      For demonstration, we split the document into sentences and track metadata.
    """

    def __init__(self, data_samples):
        """
        data_samples: list of dicts, each containing keys "query", "document", and "teacher_scores"
        """
        self.samples = []
        for sample in data_samples:
            query = sample["query"]
            doc = sample["document"]
            # Split document into sentences.
            sents = sent_tokenize(doc)
            teacher_scores = sample["teacher_scores"]
            # Assume teacher_scores are precomputed for each sentence (for simplicity, len(teacher_scores)==len(sents))
            # Track metadata for each sentence. (e.g., sentence index, original doc)
            for idx, sent in enumerate(sents):
                self.samples.append({
                    "query": query,
                    "sentence": sent,
                    "teacher_score": teacher_scores[idx] if idx < len(teacher_scores) else random.uniform(0, 1),
                    "metadata": {
                        "sentence_index": idx,
                        "document": doc
                    }
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ------------------------------
# 4. Training Code
# ------------------------------

def collate_fn(batch):
    """
    Since each sample is independent (a query with a single sentence),
    collate the batch into lists.
    """
    queries = [item["query"] for item in batch]
    sentences = [item["sentence"] for item in batch]
    teacher_scores = torch.tensor([item["teacher_score"] for item in batch], dtype=torch.float32)
    metadata = [item["metadata"] for item in batch]
    return {"queries": queries, "sentences": sentences, "teacher_scores": teacher_scores, "metadata": metadata}


def train_reranker(model, dataset, max_k=3, tau=0.5, epochs=3, batch_size=8, lr=1e-5, device='cpu'):
    """
    Train the reranker using teacher combined scores as the target.

    The loss here compares the predicted scores (optionally weighted with a differentiable top-k mask)
    to the teacher scores using MSE.

    Note: In this simple example, each batch is treated independently.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            # For simplicity, assume all queries in the batch are identical or process each individually.
            # You can alternatively handle heterogeneous queries by iterating within the batch.
            # Here, we process one query-sentence pair per sample.
            queries = batch["queries"]
            sentences = batch["sentences"]
            teacher_scores = batch["teacher_scores"].to(device)  # shape: (batch_size,)

            # We can process each sample independently (or concatenate by assuming they share the same query).
            # For demonstration, here we use the first query in the batch.
            # In practice, you may combine queries and sentences if multiple sentences belong to a single query.
            query = queries[0]

            optimizer.zero_grad()

            # Forward pass: compute predicted relevance scores.
            predicted_scores = model(query, sentences)  # shape: (batch_size,)

            # Apply differentiable top-k selection (for instance, if you wish to emphasize only the top-k)
            # Note: when batch_size is small or each sample is from a different query, this might be optional.
            selection_mask = differentiable_top_k(predicted_scores, max_k=max_k, tau=tau)

            # Option 1: Loss on weighted scores (focus on high-scoring sentences)
            loss = F.mse_loss(predicted_scores * selection_mask, teacher_scores * selection_mask)

            # Option 2 (alternative): Plain MSE loss over all sentences:
            # loss = F.mse_loss(predicted_scores, teacher_scores)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")


# ------------------------------
# 5. Example Data and Running Training
# ------------------------------

if __name__ == "__main__":
    # Example training data.
    # In practice, the teacher scores for each sentence should come from your DataMorgana + LLM combination.
    example_data = [
        {
            "query": "What are the recent trends in hybrid retrieval models?",
            "document": ("Hybrid retrieval systems combine sparse and dense methods. "
                         "They have shown promising results in retrieval-augmented generation. "
                         "This approach can improve both precision and recall."),
            # For demonstration, we assign random teacher scores to each sentence.
            "teacher_scores": [0.8, 0.9, 0.7]
        },
        {
            "query": "How does Gumbel Reranking improve retrieval?",
            "document": ("Gumbel Reranking introduces a differentiable subset selection method. "
                         "It uses the Gumbel softmax to approximate top-k selection. "
                         "This makes training end-to-end possible."),
            "teacher_scores": [0.7, 0.85, 0.75]
        }
    ]

    dataset = RerankerDataset(example_data)
    reranker_model = RerankerModel()

    # Train the reranker model.
    train_reranker(reranker_model, dataset, max_k=2, tau=0.5, epochs=5, batch_size=4, lr=1e-5, device='cpu')


    # ------------------------------
    # Inference Example: (Tracking metadata)
    # ------------------------------
    def infer_query(model, query, document, max_k=2, tau=0.5):
        """Perform inference on a new query & document while preserving sentence metadata."""
        sents = sent_tokenize(document)
        metadata = [{"sentence_index": i, "document": document} for i in range(len(sents))]

        model.eval()
        with torch.no_grad():
            scores = model(query, sents)
            selection_mask = differentiable_top_k(scores, max_k=max_k, tau=tau)
            # For a hard decision, pick the top-k indices.
            _, topk_indices = torch.topk(selection_mask, max_k)
            selected_sentences = [sents[i] for i in topk_indices.tolist()]
            selected_metadata = [metadata[i] for i in topk_indices.tolist()]
        return selected_sentences, selected_metadata


    # Example inference.
    test_query = "Explain the benefits of differentiable reranking."
    test_doc = ("Differentiable reranking allows gradients to flow from the language model back to "
                "the retrieval module. This leads to improved joint optimization. "
                "Moreover, using techniques like Gumbel softmax further aligns the training and inference stages.")

    selected_sents, selected_meta = infer_query(reranker_model, test_query, test_doc, max_k=2, tau=0.5)

    print("\nSelected Sentences (Inference):")
    for sent in selected_sents:
        print("-", sent)

    print("\nAssociated Metadata:")
    for meta in selected_meta:
        print(meta)


# Need to select the model for reranking (Small model = Finetune)
# mxbai-rerank-large-v2
# Need to select model for generating relavance score (Big Model = Inference)

# reranker is giving a score to each sentence.
# so for loss we can use this score or we may use the last generation of the model??
# which one should we use ??
