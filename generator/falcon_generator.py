from ai71 import AI71
from dotenv import load_dotenv
import os


class FalconGenerator:
    def __init__(self, model_name: str = "tiiuae/falcon3-10b-instruct"):
        load_dotenv()
        self.__API_KEY = os.getenv("AI71_API_KEY")
        self.client = AI71(self.__API_KEY)
        self.model_name = model_name

    def generate_answer(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

