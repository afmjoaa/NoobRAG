from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document
from typing import List, Dict
import uuid
from itertools import cycle
from multiprocessing.pool import ThreadPool

from topic.topic_segmenter import get_topic_text_with_info  # Keep if used elsewhere


class NvidiaReranker:
    def __init__(self, model: str, api_keys: List[str]):
        """Initialize with API key rotation support."""
        self.model = model
        self.api_key_cycle = cycle(api_keys)  # Create infinite key iterator

    def _add_uid(self, documents: List[Dict]) -> List[Dict]:
        """Add unique IDs to documents if missing."""
        for doc in documents:
            doc.setdefault('uid', str(uuid.uuid4()))
        return documents

    def rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Rerank documents with API key rotation."""
        # Add UIDs and create mapping
        documents = self._add_uid(documents)
        uid_map = {doc['uid']: doc for doc in documents}

        # Create client with rotated API key
        client = NVIDIARerank(
            model=self.model,
            api_key=next(self.api_key_cycle)  # Get next key in rotation
        )

        # Convert to LangChain format and rerank
        lc_docs = [
            Document(
                page_content=self.truncate_text_estimate(doc['passage']),
                metadata={'uid': doc['uid']}
            ) for doc in documents
        ]
        reranked = client.compress_documents(query=query, documents=lc_docs)

        # Update scores and sort
        return sorted(
            [
                {**uid_map[doc.metadata['uid']], 'score': doc.metadata['relevance_score']}
                for doc in reranked
            ],
            key=lambda x: x['score'],
            reverse=True
        )

    def batch_rerank(self, queries: List[str], batch_documents: List[List[Dict]], n_parallel: int = 10) -> List[List[Dict]]:
        """Batch rerank documents for multiple queries using parallel threads."""
        with ThreadPool(n_parallel) as pool:
            results = pool.starmap(self.rerank_documents, zip(queries, batch_documents))
        return results

    @staticmethod
    def truncate_text_estimate(text: str, max_tokens: int = 300) -> str:
        """Smart text truncation for model input."""
        return text[:max_tokens * 4]  # Simple char-based truncation


if __name__ == "__main__":
    # Example usage
    reranker = NvidiaReranker(
        model="nvidia/nv-rerankqa-mistral-4b-v3",
        api_keys=["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"]  # Add more keys
    )

    results = reranker.rerank_documents(
        query="Sample query",
        documents=[{"passage": "Sample content..."}]  # Add your documents
    )
    print(results)