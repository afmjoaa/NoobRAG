import numpy as np
from typing import List, Dict, Any, Optional

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

    def retrieve(
        self,
        query: Optional[str] = None,
        top_k: int = 5,
        max_docs: int = 6,
        embedded_query_vector: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        num_dense = int(np.ceil(0.6 * max_docs))
        num_sparse = max_docs - num_dense
        seen_ids = set()
        docs = []

        # --- Dense Retrieval ---
        dense_docs = []
        if embedded_query_vector is not None:
            dense_results = self.dense_retriever.query_by_vector(embedded_query_vector, top_k=top_k)
        else:
            dense_results = self.dense_retriever.query(query, top_k=top_k)

        for match in dense_results.get("matches", []):
            doc_id = match["id"]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                dense_docs.append({
                    "text": match["metadata"].get("text", ""),
                    "score": match["score"],
                    "source": "dense",
                    "id": doc_id,
                })
        dense_docs.sort(key=lambda d: d["score"], reverse=True)
        docs.extend(dense_docs[:num_dense])

        # --- Sparse Retrieval ---
        sparse_docs = []
        sparse_results = self.sparse_retriever.query(query, top_k=top_k)
        hits = sparse_results.get("hits", {}).get("hits", [])
        scores = np.array([hit["_score"] for hit in hits]) if hits else np.array([])
        normalized_scores = softmax_score(scores) if len(scores) > 0 else []

        for idx, hit in enumerate(hits):
            doc_id = hit["_id"]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                sparse_docs.append({
                    "text": hit["_source"].get("text", ""),
                    "score": float(normalized_scores[idx]) if idx < len(normalized_scores) else 0.0,
                    "source": "sparse",
                    "id": doc_id,
                })
        sparse_docs.sort(key=lambda d: d["score"], reverse=True)
        docs.extend(sparse_docs[:num_sparse])

        # Final sort
        docs.sort(key=lambda d: d["score"], reverse=True)
        return docs

    def merge_results(
        self,
        dense_results: Dict[str, Any],
        sparse_results: Dict[str, Any],
        max_docs: int = 6
    ) -> Dict[str, Any]:
        """
        Merge Pinecone-style results from dense and sparse.
        """
        merged = {}
        def add_matches(source_results, weight=1.0):
            for match in source_results.get("matches", []):
                doc_id = match["id"]
                score = match["score"] * weight
                if doc_id in merged:
                    merged[doc_id]["score"] += score
                    merged[doc_id]["count"] += 1
                else:
                    merged[doc_id] = {
                        "id": doc_id,
                        "score": score,
                        "metadata": match["metadata"],
                        "count": 1
                    }

        add_matches(dense_results, weight=1.0)
        add_matches(sparse_results, weight=0.7)

        merged_docs = list(merged.values())
        for doc in merged_docs:
            doc["score"] /= doc["count"]

        top_docs = sorted(merged_docs, key=lambda x: x["score"], reverse=True)[:max_docs]
        return {"matches": top_docs}

    @staticmethod
    def show_results(docs: List[Dict[str, Any]]):
        print("\n[Combined Retrieved Documents]")
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i + 1} ---")
            print(f"Source: {doc.get('source', 'unknown')}, Score: {doc['score']:.4f}")
            print(doc["text"][:400])