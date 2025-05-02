from typing import List, Dict, Any


def build_prompt(query: str, docs: List[Dict[str, Any]], max_docs: int = 5) -> str:
    """Construct a prompt from retrieved docs."""
    selected_docs = docs[:max_docs]
    context_parts = [
        f"Context Document {i + 1}: {doc['passage']}"
        for i, doc in enumerate(selected_docs)
    ]
    context_block = "\n\n".join(context_parts)

    return (
        "You are a knowledgeable assistant. Answer the following question using only the context provided below.\n\n"
        "----\n"
        f"{context_block}\n"
        "----\n"
        f"Question: {query}\n"
        "Answer:"
    )


def get_refine_query(query: str) -> str:
    prompt = f"""
You are a helpful assistant. Your task is to rewrite the following query to make it more specific, clear, and well-structured, while preserving its original intent.

Original Query:
"{query}"

Refined Query:
""".strip()
    return prompt


def get_hypothetical_answer(query: str) -> str:
    prompt = f"""
You are a knowledgeable assistant. Using general domain expertise and commonly accepted facts, craft a concise and informative 3-line answer to the question below.
Ensure the response is accurate, relevant, and uses appropriate technical or domain-specific language where applicable.

Question: {query}

Hypothetical Answer:
""".strip()
    return prompt




# result = falcon_generator.generate_answer(prompt).strip()
#     print("\n[Refined Query]")
#     print(result)

# A strict length limit of 200 tokens