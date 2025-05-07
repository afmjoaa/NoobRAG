from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from ai71 import AI71


class FalconGenerator:
    __API_KEY = "ai71-api-b1e07fa1-d007-41cd-8306-85fc952e12a6"

    def __init__(self, model_name: str = "tiiuae/falcon3-10b-instruct"):
        self.client = AI71(self.__API_KEY)
        self.model_name = model_name

    def generate_answer(self, prompt: str, n_retries: int = 5) -> str:
        """Generates an answer for a single prompt with retry logic."""
        retries = 0
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
                print(f"Retrying for the {retries} time(s)... (error: {e})")
                time.sleep(retries)

    def batch_generate_answer(self, prompts: List[str], n_parallel: int = 10, n_retries: int = 5) -> List[str]:
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

