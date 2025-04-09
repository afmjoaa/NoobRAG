from typing import List, Dict, Any


def build_prompt(query: str, docs: List[Dict[str, Any]], max_docs: int = 5) -> str:
    """Construct a prompt from retrieved docs."""
    selected_docs = docs[:max_docs]
    context_parts = [
        # f"Context Document {i + 1} (source: {doc['source']} | score: {doc['score']:.2f}):\n{doc['text']}"
        f"Context Document {i + 1} (source: {doc['source']}):\n{doc['text']}"
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
