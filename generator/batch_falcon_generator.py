from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
from ai71 import AI71


class FalconGenerator:
    __API_KEY = "ai71-api-b1e07fa1-d007-41cd-8306-85fc952e12a6"

    def __init__(self, model_name: str = "tiiuae/falcon3-10b-instruct"):
        self.client = AI71(self.__API_KEY)
        self.model_name = model_name

    def generate_answer(self, prompt: str, n_retries: int = 10) -> str:
        """Generates an answer for a single prompt with exponential backoff retry logic."""
        retries = 0
        base_delay = 1  # Starting delay in seconds
        max_delay = 60  # Maximum delay in seconds

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a knowledgeable assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                retries += 1
                if retries > n_retries:
                    raise e

                # Calculate exponential backoff with jitter
                delay = min(max_delay, base_delay * (2 ** (retries - 1)))
                # Add jitter (randomness) to avoid thundering herd problem
                jitter = random.uniform(0.5, 1.5)
                actual_delay = delay * jitter

                print(f"Retry {retries}/{n_retries} after {actual_delay:.2f}s... (error: {e})")
                time.sleep(actual_delay)

    def batch_generate_answer(self, prompts: List[str], n_parallel: int = 10, n_retries: int = 30) -> List[str]:
        """Generates answers for multiple prompts in parallel, preserving order."""
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=n_parallel) as executor:
            # Map each future to its index in the prompts list
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
                    results[index] = None
        return results

