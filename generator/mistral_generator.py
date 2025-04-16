from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class MistralGenerator:
    def __init__(self, model_name: str = "NousResearch/Nous-Hermes-2-Mistral-7B", use_gpu: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() and use_gpu else torch.float32,
            device_map="auto" if use_gpu else {"": "cpu"}
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.3,
        )

    def generate_answer(self, prompt: str) -> str:
        output = self.pipeline(prompt)[0]["generated_text"]
        return self._postprocess(output, prompt)

    def _postprocess(self, output: str, prompt: str) -> str:
        return output.replace(prompt, "").strip()