

def build_prompt(question, contexts):
    context_str = "\n\n".join(contexts)
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context_str}

Question: {question}
Answer:"""
    return prompt
