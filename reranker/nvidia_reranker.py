import json
import os

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document
from typing import List, Dict
import uuid

from topic.topic_segmenter import get_topic_text_with_info


class NvidiaReranker:
    def __init__(self, model: str, api_key: str):
        """Initialize the NVIDIA Reranker client."""
        self.client = NVIDIARerank(model=model, api_key=api_key)

    def _add_uid(self, documents: List[Dict]) -> List[Dict]:
        """Add unique identifier to each document if not present."""
        for doc in documents:
            if 'uid' not in doc:
                doc['uid'] = str(uuid.uuid4())
        return documents

    def rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Rerank documents and return them in the original format with updated scores.

        Args:
            query: The search query
            documents: List of documents in dictionary format with 'text' field

        Returns:
            List of documents in original format with updated relevance scores,
            maintaining all original fields and order by new score
        """
        # Add unique identifiers if not present
        documents = self._add_uid(documents)

        # Create mapping of UID to original document
        uid_to_doc = {doc['uid']: doc for doc in documents}

        # Convert to LangChain Document format with UID in metadata
        lc_documents = [
            Document(
                page_content=self.truncate_text_estimate(doc['passage']),
                metadata={'uid': doc['uid']}
            ) for doc in documents
        ]

        # Rerank documents
        reranked_docs = self.client.compress_documents(query=query, documents=lc_documents)

        # Update original documents with new scores using UID mapping
        result_docs = []
        for reranked_doc in reranked_docs:
            uid = reranked_doc.metadata['uid']
            original_doc = uid_to_doc[uid]
            updated_doc = original_doc.copy()
            updated_doc['score'] = reranked_doc.metadata['relevance_score']
            result_docs.append(updated_doc)

        # Sort by new score in descending order
        return sorted(result_docs, key=lambda x: x['score'], reverse=True)

    # TODO rather truncating ca we summarize using a model within 512 token
    @staticmethod
    def truncate_text_estimate(text, max_tokens=300):
        approx_chars_per_token = 4
        max_chars = max_tokens * approx_chars_per_token
        return text[:max_chars]


if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "nvidia/nv-rerankqa-mistral-4b-v3"
    load_dotenv()
    nvidia_api_keys = json.loads(os.getenv("NVIDIA_API_KEYS", "[]"))

    # Sample data
    query = "What is Left and Right Brain technology?"
    input_documents = get_topic_text_with_info()

    # Process the documents
    reranker = NvidiaReranker(model=MODEL_NAME, api_key=nvidia_api_keys[0])
    reranked_docs = reranker.rerank_documents(query=query, documents=input_documents)

    print(reranked_docs)