from ai71 import AI71


class FalconGenerator:
    def __init__(self, api_key):
        self.client = AI71(api_key)
        self.model = "tiiuae/falcon-3-10b-instruct"

    def generate(self, prompt, stream=False):
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream
        )
        content = ""
        for chunk in response:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                content += delta_content
        return content
