import random
import time
import threading
from collections import deque
from typing import List, Dict
from multiprocessing.pool import ThreadPool
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document
import uuid
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


class NvidiaReranker:
    def __init__(self, model: str, api_keys: List[str], rpm_limit: int = 40):
        self.model = model
        self.api_keys = api_keys.copy()
        self.rpm_limit = rpm_limit
        self.key_data = {
            key: {'timestamps': deque(maxlen=rpm_limit), 'lock': threading.Lock()}
            for key in self.api_keys
        }
        self.key_index = 0
        self.tokenizer = MistralTokenizer.v1()
        self.key_lock = threading.Lock()  # For thread-safe key rotation

    def _get_next_key(self):
        with self.key_lock:
            key = self.api_keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(self.api_keys)
            return key

    def _wait_for_key_capacity(self, key):
        with self.key_data[key]['lock']:
            now = time.time()
            # Remove timestamps older than 60 seconds
            while self.key_data[key]['timestamps'] and (now - self.key_data[key]['timestamps'][0] > 60):
                self.key_data[key]['timestamps'].popleft()
            if len(self.key_data[key]['timestamps']) >= self.rpm_limit:
                # Wait until the oldest timestamp is older than 60 seconds
                oldest = self.key_data[key]['timestamps'][0]
                wait_time = oldest + 60 - now
                if wait_time > 0:
                    time.sleep(wait_time)
                # Re-check after waiting
                now = time.time()
                while self.key_data[key]['timestamps'] and (now - self.key_data[key]['timestamps'][0] > 60):
                    self.key_data[key]['timestamps'].popleft()
            # Add the current timestamp
            self.key_data[key]['timestamps'].append(now)

    def _add_uid(self, documents: List[Dict]) -> List[Dict]:
        for doc in documents:
            doc.setdefault('uid', str(uuid.uuid4()))
        return documents

    def rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        max_retries = 30
        retry_delay = 30  # seconds
        for attempt in range(max_retries):
            try:
                key = self._get_next_key()
                self._wait_for_key_capacity(key)
                documents = self._add_uid(documents)
                uid_map = {doc['uid']: doc for doc in documents}
                client = NVIDIARerank(model=self.model, api_key=key)
                lc_docs = [
                    Document(
                        page_content=self.truncate_text_to_max_tokens(doc['passage']),
                        metadata={'uid': doc['uid']}
                    ) for doc in documents
                ]
                reranked = client.compress_documents(query=query, documents=lc_docs)
                sorted_docs = sorted(
                    [
                        {**uid_map[doc.metadata['uid']], 'score': doc.metadata['relevance_score']}
                        for doc in reranked
                    ],
                    key=lambda x: x['score'],
                    reverse=True
                )
                return sorted_docs
            except Exception as e:
                if self._is_ratelimit_exception(e) and attempt < max_retries - 1:
                    # Add jitter (randomness) to avoid thundering herd problem
                    jitter = random.uniform(0.5, 1.5)
                    actual_delay = retry_delay * jitter
                    print(f"Rate limit exceeded, retrying in {actual_delay} seconds...")
                    time.sleep(actual_delay)
        return []  # Fallback return

    def _is_ratelimit_exception(self, e):
        # Check if the exception is a 429 error
        if hasattr(e, 'status'):
            return e.status == 429
        elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            return e.response.status_code == 429
        return False

    def batch_rerank(self, queries: List[str], batch_documents: List[List[Dict]], n_parallel: int = 10) -> List[
        List[Dict]]:
        with ThreadPool(n_parallel) as pool:
            results = pool.starmap(self.rerank_documents, zip(queries, batch_documents))
        return results

    @staticmethod
    def truncate_text_estimate(text: str, max_tokens: int = 300) -> str:
        return text[:max_tokens * 4]

    def truncate_text_to_max_tokens(self, text: str, max_tokens: int = 400) -> str:
        # Binary search for truncation point
        low, high = 0, len(text)
        best_fit = ""
        while low <= high:
            mid = (low + high) // 2
            trial_text = text[:mid]

            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=trial_text)]
            )
            token_count = len(self.tokenizer.encode_chat_completion(completion_request).tokens)

            if token_count <= max_tokens:
                best_fit = trial_text
                low = mid + 1
            else:
                high = mid - 1
        return best_fit

