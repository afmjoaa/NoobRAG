from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load FLAN-T5 model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
coref_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def resolve_coreference(text):
    prompt = f"Resolve all pronouns in this paragraph: {text}"
    result = coref_pipeline(prompt, max_length=512)[0]['generated_text']
    return result

# Example text
text = "John went to the store. He bought a book. Sarah loves painting. She often paints landscapes."
resolved_text = resolve_coreference(text)

print("Resolved Text:")
print(resolved_text)