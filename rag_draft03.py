import numpy as np
from generator.mistral_generator import MistralGenerator
from generator.falcon_generator import FalconGenerator
from retrievers.dense_retriever import DenseRetriever
from retrievers.sparse_retriever import SparseRetriever
from utils.combined_retriever import CombinedRetriever
from utils.prompt_template import build_prompt
from utils.save_to_file import save_to_csv


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


def getCombinedContext(query: str = "What is a second brain?", top_k: int = 5, max_docs: int = 6, will_print: bool = True, use_hyde: bool = True):
    dense_retriever = DenseRetriever()
    sparse_retriever = SparseRetriever()
    combined_retriever = CombinedRetriever(dense_retriever, sparse_retriever)

    if use_hyde:
        # 1. Decompose query
        decomposed_query = decompose_query(query)

        # 2. Generate HyDE answer
        hypothetical_answer = generate_hypothetical_answer(query)

        # 3. Get embeddings
        query_vec = dense_retriever.embed_text(decomposed_query)
        hyde_vec = dense_retriever.embed_text(hypothetical_answer)

        # 4. Combine embeddings (90% decomposed + 10% HyDE)
        combined_vec = 0.7 * query_vec + 0.3 * hyde_vec

        # 5. Clean decomposed query for sparse search
        sparse_query_text = decomposed_query.replace('\n', ' ').replace('2.', '').replace('3.', '').replace('3.', '').strip()

        # 6. Use vector-based retrieval (dense) + string query (sparse)
        combined_results = combined_retriever.retrieve(
            embedded_query_vector=combined_vec,
            query=sparse_query_text,
            top_k=top_k,
            max_docs=max_docs
        )
    else:
        # Use original query string
        combined_results = combined_retriever.retrieve(
            query=query,
            top_k=top_k,
            max_docs=max_docs
        )

    if will_print:
        combined_retriever.show_results(combined_results)

    return combined_results


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
    top_k = 4
    max_docs = 5

    # Generate everything
    prompt, answer = getGeneratedAnswer(query=question, top_k=top_k, max_docs=max_docs, use_hyde=True)

    print("\n[Original Query]")
    print(question)

    print("\n[Final Prompt]")
    print(prompt)

    print("\n[Final Answer]")
    print(answer)

    # Optionally save
    # save_to_csv(prompt, answer, filename="./data/generated_answers.csv")