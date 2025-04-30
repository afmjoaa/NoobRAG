import numpy as np
from typing import List, Dict, Any

from retrievers.dense_retriever import DenseRetriever
from retrievers.sparse_retriever import SparseRetriever


def softmax_score(scores: np.ndarray) -> np.ndarray:
    """Apply softmax normalization to scores."""
    exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
    return exp_scores / np.sum(exp_scores)


class CombinedRetriever:
    def __init__(
            self,
            dense_retriever: DenseRetriever,
            sparse_retriever: SparseRetriever,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

    def retrieve(self, query: str, top_k: int = 5, max_docs: int = 6) -> List[Dict[str, Any]]:
        num_dense = int(np.ceil(0.6 * max_docs))
        num_sparse = max_docs - num_dense  # Remaining from sparse

        seen_ids = set()
        docs = []

        # --- Dense Retrieval ---
        dense_docs = []
        dense_results = self.dense_retriever.query(query, top_k=top_k)
        for match in dense_results.get("matches", []):
            doc_id = match["id"]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                dense_docs.append({
                    "doc_id": match["metadata"].get("doc_id", ""),
                    "text": match["metadata"].get("text", ""),
                    "score": match["score"],
                    "source": "dense"
                })
        dense_docs.sort(key=lambda d: d["score"], reverse=True)
        docs.extend(dense_docs[:num_dense])

        # --- Sparse Retrieval ---
        sparse_docs = []
        sparse_results = self.sparse_retriever.query(query, top_k=top_k)
        hits = sparse_results.get("hits", {}).get("hits", [])
        scores = np.array([hit["_score"] for hit in hits])
        normalized_scores = softmax_score(scores)

        for idx, hit in enumerate(hits):
            doc_id = hit["_id"]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                sparse_docs.append({
                    "doc_id": hit["_source"].get("doc_id", ""),
                    "text": hit["_source"].get("text", ""),
                    "score": float(normalized_scores[idx]),
                    "source": "sparse"
                })
        sparse_docs.sort(key=lambda d: d["score"], reverse=True)
        docs.extend(sparse_docs[:num_sparse])

        # Optional: re-sort combined docs by score (or keep dense/sparse ordering)
        docs.sort(key=lambda d: d["score"], reverse=True)

        return docs

    @staticmethod
    def show_results(docs):
        print("Combined docs :\n", docs)


# Example usage:
if __name__ == "__main__":
    dense_retriever = DenseRetriever()
    sparse_retriever = SparseRetriever()

    combined_retriever = CombinedRetriever(dense_retriever, sparse_retriever)

    query = "What is a second brain?"
    docs = combined_retriever.retrieve(query, top_k=5, max_docs=6)
    print("Combined docs :\n", docs)

