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

    def batch_retrieve(
            self,
            queries: List[str],
            top_k: int = 5,
            max_docs: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a batch of queries by combining results from dense and sparse retrievers.

        Args:
            queries: List of query strings.
            top_k: Number of top documents to retrieve from each retriever per query.
            max_docs: Maximum combined documents to return per query.

        Returns:
            List of document lists, one per query.
        """
        # Get batch results from both retrievers
        dense_batch_results = self.dense_retriever.batch_query(queries, top_k=top_k)
        sparse_batch_results = self.sparse_retriever.batch_query(queries, top_k=top_k)
        sparse_batch_results = sparse_batch_results.get("responses", [])
        print(f"sparse_batch_results:\n {sparse_batch_results}")

        all_queries_docs = []

        # Process each query's results
        for dense_results, sparse_results in zip(dense_batch_results, sparse_batch_results):
            num_dense = int(np.ceil(0.6 * max_docs))
            num_sparse = max_docs - num_dense
            seen_ids = set()
            combined_docs = []

            # Process dense results
            dense_docs = []
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
            # Take top dense docs
            dense_docs.sort(key=lambda x: x["score"], reverse=True)
            combined_docs.extend(dense_docs[:num_dense])

            # Process sparse results
            sparse_docs = []
            hits = sparse_results.get("hits", {}).get("hits", [])
            if hits:
                # Normalize scores using softmax
                scores = np.array([hit["_score"] for hit in hits])
                norm_scores = softmax_score(scores)

                for idx, hit in enumerate(hits):
                    doc_id = hit["_id"]
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        sparse_docs.append({
                            "doc_id": hit["_source"].get("doc_id", ""),
                            "text": hit["_source"].get("text", ""),
                            "score": float(norm_scores[idx]),
                            "source": "sparse"
                        })
            # Take top sparse docs
            sparse_docs.sort(key=lambda x: x["score"], reverse=True)
            combined_docs.extend(sparse_docs[:num_sparse])

            # Final combined results
            combined_docs.sort(key=lambda x: x["score"], reverse=True)
            all_queries_docs.extend(combined_docs[:max_docs])  # Ensure max_docs limit

        return all_queries_docs


    @staticmethod
    def show_results(docs):
        print("Combined docs :\n", docs)


# Example usage:
if __name__ == "__main__":
    dense_retriever = DenseRetriever()
    sparse_retriever = SparseRetriever()

    combined_retriever = CombinedRetriever(dense_retriever, sparse_retriever)

    query = "What is a second brain?"
    docs = combined_retriever.batch_retrieve([query * 10], top_k=5, max_docs=6)
    print("Combined docs :\n", docs)

# EXAMPLE OUTPUT
# [
#    {
#       "doc_id":"<urn:uuid:0cf75b43-d690-4aa6-b3ca-f488ceb28ed9>",
#       "text":"What will the age of Aquarius be like. What is second level consciousness and are we still there? Left and Right Brain technology what is it? part 1\nSorry we couldn't complete your registration. Please try again.\nYou must accept the Terms and conditions to register",
#       "score":0.835362315,
#       "source":"dense"
#    },
#    {
#       "doc_id":"<urn:uuid:b6541cf6-d442-454a-9e6f-45eaaa237424>",
#       "text":"Title: Left-Brain/Right-Brain Functions\nPreview: We have two eyes, two ears, two hands, and two minds. Our left brain thinks in terms of words and symbols while our right brain thinks in terms of images. The left side is the side used more by writiers, mathematicians, and scientists; the right side by artists, craftspeople, and musicians. Remembering a persons name is a function of the left-brain memory while rembering their face is a function .......\nBy aterry (adrienne)\non September 27, 2012.",
#       "score":0.823291481,
#       "source":"dense"
#    },
#    {
#       "doc_id":"<urn:uuid:e4bf2415-2032-4a8a-9c18-715cf2d5f91f>",
#       "text":"The legal term for this compensation is “damages.” Exactly what damages you can recover varies from state to state, but you can usually recover:\n- Past and future medical expenses\n- Future lost wages (if the injury limits your ability to work in the future)\n- Property damages\n- Pain and suffering\n- Emotional distress\nReady to contact a lawyer about a possible second impact syndrome case? Use our free online directory to schedule your initial consultation today.\n- Guide to traumatic brain injuries\n- Resources to help after a brain injury\n- How to recognize a brain injury and what you should do about it\n- Concussions and auto accidents\n- Rehabilitation and therapy after a brain injury\n- Second impact syndrome and sports injury lawsuits\n- Legal guide to brain death\n- What is CTE?\n- A loss of oxygen can lead to an anoxic brain injury\n- Can you recover costs for the accident that caused a brain bleed?\n- What is the Traumatic Brain Injury Act?\n- Understanding the Hidden Challenges of Mild Traumatic Brain Injury\n- What is the Glasgow Coma Scale?",
#       "score":0.24633245645981225,
#       "source":"sparse"
#    }
# ]