from generator.falcon_generator import FalconGenerator
from generator.mistral_generator import MistralGenerator
from retrievers.dense_retriever import DenseRetriever
from retrievers.sparse_retriever import SparseRetriever
from utils.combined_retriever import CombinedRetriever
from utils.prompt_template import build_prompt
from utils.save_to_file import save_to_csv
import sys
import os

# Add the project root to Python's path (optional)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def get_generator(model_name: str = "mistral"):
    if model_name == "falcon":
        return FalconGenerator()
    elif model_name == "mistral":
        return MistralGenerator(model_name="mistralai/Mistral-7B-Instruct-v0.3")
    else:
        raise ValueError(f"Unknown model: {model_name}")

def getDenseContext(query: str = "What is a second brain?", top_k: int = 5):
    dense_retriever = DenseRetriever()
    dense_results = dense_retriever.query(query=query, top_k=top_k)
    dense_retriever.show_results(dense_results)

def getSparseContext(query: str = "What is a second brain?", top_k: int = 5):
    sparse_retriever = SparseRetriever()
    sparse_results = sparse_retriever.query(query, top_k)
    sparse_retriever.show_results(sparse_results)

def getCombinedContext(query: str = "What is a second brain?", top_k: int = 5, max_docs: int = 6, will_print: bool = True):
    dense_retriever = DenseRetriever()
    sparse_retriever = SparseRetriever()
    combined_retriever = CombinedRetriever(dense_retriever, sparse_retriever)

    combined_results = combined_retriever.retrieve(query, top_k=top_k, max_docs=max_docs)
    if will_print:
        combined_retriever.show_results(combined_results)
    return combined_results

def getPrompt(query: str = "What is a second brain?", top_k: int = 5, max_docs: int = 6, will_print: bool = True):
    combined_result = getCombinedContext(query=query, top_k=top_k, max_docs=max_docs, will_print=False)
    prompt = build_prompt(query, combined_result, max_docs=max_docs)
    if will_print:
        print("Generated Prompt:\n", prompt)
    return prompt

def getGeneratedAnswer(query: str, top_k: int, max_docs: int):
    prompt = getPrompt(query=query, top_k=top_k, max_docs=max_docs, will_print=False)
    generator = get_generator("falcon")
    answer = generator.generate_answer(prompt)
    return prompt, answer

def decompose_query(query: str):
    """
    Uses selected LLM to break a query into smaller, clearer sub-questions.
    """
    generator = get_generator("falcon")
    decomposition_prompt = f"""
You are a helpful assistant. Your task is to decompose the following query into smaller, clear, and meaningful sub-questions that together help answer the original query.

Original Query:
"{query}"

Decomposed Sub-Questions:
1.
"""
    result = generator.generate_answer(decomposition_prompt)
    print("Decomposed Sub-Questions:")
    print(result.strip())
    return result

if __name__ == "__main__":
    question = "What is a second brain?"
    top_k = 4
    max_docs = 5

    # Step 1: Decompose the query
    decomposed = decompose_query(question)

    # Step 2: Generate prompt and answer for the main query
    prompt, answer = getGeneratedAnswer(query=question, top_k=top_k, max_docs=max_docs)

    # Output
    print("\nOriginal Query:")
    print(question)

    print("\nGenerated Prompt:")
    print(prompt)

    print("\nAnswer:")
    print(answer)

    # Optional: Save to file
    # save_to_csv(prompt, answer, filename="./data/generated_answers.csv")
