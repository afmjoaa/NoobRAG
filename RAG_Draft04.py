import numpy as np
from generator.mistral_generator import MistralGenerator
from generator.falcon_generator import FalconGenerator
from retrievers.dense_retriever import DenseRetriever
from retrievers.sparse_retriever import SparseRetriever
from utils.combined_retriever import CombinedRetriever
from utils.prompt_template import build_prompt
from utils.save_to_file import save_to_csv
from multiprocessing.pool import ThreadPool

def decompose_query(query: str) -> str:
    generator = MistralGenerator(model_name="mistralai/Mistral-7B-Instruct-v0.3")
    prompt = f"""
You are a helpful assistant. Your task is to decompose the following query into 3 smaller, clear, and meaningful sub-questions that together help answer the original query.

Original Query:
"{query}"

Decomposed Sub-Questions:
1.
""".strip()
    result = generator.generate_answer(prompt).strip()
    print("\n[Decomposed Query]")
    print(result)
    return result


def generate_hypothetical_answer(query: str) -> str:
    generator = MistralGenerator(model_name="mistralai/Mistral-7B-Instruct-v0.3")
    prompt = f"""
You are a helpful assistant. Based on common knowledge and likely sources, generate a plausible, informative, and domain-relevant 3-line answer to the question below.
Focus on clear, factual detail using terminology relevant to the topic.

Question: {query}

Hypothetical Answer:
""".strip()
    answer = generator.generate_answer(prompt).strip()
    print("\n[HyDE] Hypothetical Answer:")
    print(answer)
    return answer

def getCombinedContext(
    query: str = "What is a second brain?",
    top_k: int = 10,
    max_docs: int = 10,
    will_print: bool = False,
    use_hyde: bool = True,
    return_separate: bool = False
):
    dense_retriever = DenseRetriever()
    sparse_retriever = SparseRetriever()
    combined_retriever = CombinedRetriever(dense_retriever, sparse_retriever)

    if use_hyde:
        decomposed_query = decompose_query(query)
        hypothetical_answer = generate_hypothetical_answer(query)

        decomposed_list = decomposed_query.strip().split("\n")
        decomposed_questions = [q.split('.', 1)[-1].strip() for q in decomposed_list if q.strip()]

        query_vec = dense_retriever.embed_text(decomposed_query)
        hyde_vec = dense_retriever.embed_text(hypothetical_answer)
        combined_vec = 0.7 * query_vec + 0.3 * hyde_vec
        sparse_query_text = decomposed_query.replace('\n', ' ').replace('2.', '').replace('3.', '').strip()

        inputs = [
            ("Original Query", {"query": query}),
            ("Hypothetical Answer", {"query": hypothetical_answer}),
            ("Combined Vector", {"embedded_query_vector": combined_vec, "query": sparse_query_text}),
        ] + [(f"Decomposed Q{i+1}", {"query": dq}) for i, dq in enumerate(decomposed_questions)]

        def run_retrieval(named_input):
            label, kwargs = named_input
            results = combined_retriever.retrieve(top_k=top_k, max_docs=max_docs, **kwargs)
            return label, results

        with ThreadPool(len(inputs)) as pool:
            combined_results_list = pool.map(run_retrieval, inputs)

        grouped_docs = dict(combined_results_list)

        if will_print:
            print("\n[Retrieved Documents by Input]")
            for label, docs in grouped_docs.items():
                print(f"\n## {label} (Retrieved {len(docs)} documents)")
                for idx, doc in enumerate(docs, start=1):
                    print(f"\nDocument {idx} (source: {doc.get('source', 'unknown')}, score: {doc['score']:.4f})")
                    print(doc["text"][:500])

        if return_separate:
            return grouped_docs
        else:
            all_docs = []
            seen_ids = set()
            for docs in grouped_docs.values():
                for doc in docs:
                    if doc["id"] not in seen_ids:
                        seen_ids.add(doc["id"])
                        all_docs.append(doc)
            all_docs.sort(key=lambda d: d["score"], reverse=True)
            return all_docs

    else:
        docs = combined_retriever.retrieve(
            query=query,
            top_k=top_k,
            max_docs=max_docs
        )
        if will_print:
            combined_retriever.show_results(docs)
        return docs


def getPrompt(query: str = "What is a second brain?", top_k: int = 5, max_docs: int = 6, will_print: bool = True, use_hyde: bool = True):
    combined_result = getCombinedContext(query=query, top_k=top_k, max_docs=max_docs, will_print=False, use_hyde=use_hyde)
    prompt = build_prompt(query, combined_result, max_docs=max_docs)
    if will_print:
        print("\n[Generated Prompt]:")
        print(prompt)
    return prompt


def getGeneratedAnswer(query: str, top_k: int, max_docs: int, use_hyde: bool = True):
    prompt = getPrompt(query=query, top_k=top_k, max_docs=max_docs, will_print=False, use_hyde=use_hyde)
    generator = FalconGenerator()
    answer = generator.generate_answer(prompt)
    return prompt, answer


if __name__ == "__main__":
    question = "What is a second brain?"
    top_k = 10
    max_docs = 10

    # 1. Retrieve grouped documents
    grouped_docs = getCombinedContext(query=question, top_k=top_k, max_docs=max_docs, use_hyde=True, return_separate=True)

    print("\n[Original Query]")
    print(question)

    print("\n[Grouped Retrieved Documents]")
    for label, docs in grouped_docs.items():
        print(f"\n## {label} (Retrieved {len(docs)} documents)")
        for idx, doc in enumerate(docs, start=1):
            print(f"\nDocument {idx} (source: {doc.get('source', 'unknown')}, score: {doc['score']:.4f})")
            print(doc["text"][:500])

    # 2. Flatten all documents
    all_docs = []
    seen_ids = set()
    for docs in grouped_docs.values():
        for doc in docs:
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                all_docs.append(doc)
    all_docs.sort(key=lambda d: d["score"], reverse=True)

    # 3. Build prompt from top 5 documents
    final_top_docs = all_docs[:5]
    prompt = build_prompt(question, final_top_docs, max_docs=5)

    # 4. Generate final answer
    generator = FalconGenerator()
    answer = generator.generate_answer(prompt)

    print("\n[Final Prompt]")
    print(prompt)

    print("\n[Final Answer]")
    print(answer)