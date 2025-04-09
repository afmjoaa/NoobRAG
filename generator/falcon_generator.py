from ai71 import AI71


class FalconGenerator:
    __API_KEY = "ai71-api-b1e07fa1-d007-41cd-8306-85fc952e12a6"

    def __init__(self, model_name: str = "tiiuae/falcon3-10b-instruct"):
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

