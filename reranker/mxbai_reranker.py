import torch
from transformers import AutoModel, AutoTokenizer

from reranker.nvidia_reranker import NvidiaReranker
from topic.topic_segmenter import get_topic_text_with_info


class MxbaiReranker:
    def __init__(self, model_name="mixedbread-ai/mxbai-rerank-large-v1"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def score_documents(self, query, documents):
        """
        Scores a list of document dictionaries based on their relevance to the query
        and adds mxbai_score to each document.

        Args:
            query (str): The query string
            documents (list): List of document dictionaries containing 'text' and 'uid' fields

        Returns:
            list: The original documents with added 'mxbai_score' field
        """
        # Extract text passages while keeping track of original indices
        passages = [doc['passage'] for doc in documents]

        # Format inputs for the model
        inputs = [f"query: {query} passage: {passage}" for passage in passages]

        # Tokenize inputs
        tokenized = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**tokenized)

        # Get the [CLS] token embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        # Apply a linear layer to get the score
        # Using the model's first hidden dimension as weights for simplicity
        weights = torch.nn.Linear(cls_embeddings.size(1), 1).to(self.device)
        scores = weights(cls_embeddings).squeeze().cpu().tolist()

        # Add scores to the original documents
        scored_documents = documents.copy()
        for i, score in enumerate(scores):
            scored_documents[i]['mxbai_score'] = float(score)

        return scored_documents


# Example usage
if __name__ == "__main__":
    # Example query
    query = "What is Left and Right Brain technology?"

    # Nvidia Reranker start
    MODEL_NAME = "nvidia/nv-rerankqa-mistral-4b-v3"
    API_KEY = "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"  # Replace with your actual API key

    # Sample data
    input_documents = get_topic_text_with_info()

    # Process the documents
    nvidia_reranker = NvidiaReranker(model=MODEL_NAME, api_key=API_KEY)
    reranked_docs = nvidia_reranker.rerank_documents(query=query, documents=input_documents)

    # Initialize mxbai reranker
    mxbai_reranker = MxbaiReranker()
    # Score documents
    scored_documents = mxbai_reranker.score_documents(query, reranked_docs)

    print(scored_documents)