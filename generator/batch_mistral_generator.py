from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from collections import deque
from together import Together


class MistralGenerator:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
                 api_keys: List[str] = ["59457d628_____"], rpm_limit: int = 40):
        self.model_name = model_name
        self.api_keys = api_keys.copy()
        self.rpm_limit = rpm_limit

        # Initialize key tracking data structures
        self.key_data = {
            key: {'timestamps': deque(maxlen=rpm_limit), 'lock': threading.Lock()}
            for key in self.api_keys
        }
        self.key_index = 0
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

    def generate_answer(self, prompt: str, n_retries: int = 5) -> str:
        """Generates an answer with API key rotation and RPM limiting"""
        retries = 0
        while True:
            key = self._get_next_key()
            try:
                self._wait_for_key_capacity(key)
                client = Together(api_key=key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                retries += 1
                if retries > n_retries:
                    raise e

                if self._is_ratelimit_exception(e):
                    print(f"Rate limit hit on key, rotating... (retry {retries}/{n_retries})")
                else:
                    print(f"Retrying for the {retries} time(s)... (error: {e})")

                time.sleep(retries * 2)  # Exponential backoff

    def _is_ratelimit_exception(self, e):
        """Check if exception is a rate limit error (429 status code)"""
        if hasattr(e, 'status_code'):
            return e.status_code == 429
        if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            return e.response.status_code == 429
        return False

    def batch_generate_answer(self, prompts: List[str], queries: List[str], n_parallel: int = 10, n_retries: int = 8) -> List[str]:
        """Batch generate with thread pooling and key rotation"""
        results = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=n_parallel) as executor:
            future_to_index = {
                executor.submit(self.generate_answer, prompt, n_retries): index
                for index, prompt in enumerate(prompts)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    print(f"Request failed for prompt at index {index}: {e}")
                    results[index] = queries[index]

        return results

if __name__ == "__main__":
    # Initialize with multiple API keys
    summary_generator = MistralGenerator(
        api_keys=["59457d62865a1e3f69ae32e7f42148fafb7a2ea972e2b1438f749514892b8c8c", "2a6714a7f23ea83446e29cd1ac8c5fb4906aa720035c61fb779c92987db0b8aa"],  # Add your actual API keys
        rpm_limit=59  # Per-key RPM limit
    )

    # Batch processing with parallel requests and key management
    prompts = ["Tell me a joke", "Explain quantum computing briefly"]
    results = summary_generator.batch_generate_answer(prompts)
    print(results)
    
    
    
