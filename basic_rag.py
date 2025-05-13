from generator.falcon_generator import FalconGenerator
from retrievers.dense_retriever import DenseRetriever
from retrievers.sparse_retriever import SparseRetriever
from utils.combined_retriever import CombinedRetriever
from utils.prompt_template import build_prompt


def getDenseContext(query: str = "What is a second brain?", top_k: int = 5):
    # Dense Search Retriever
    dense_retriever = DenseRetriever()
    dense_results = dense_retriever.query(query=query, top_k=top_k)
    dense_retriever.show_results(dense_results)


def getSparseContext(query: str = "What is a second brain?", top_k: int = 5):
    # Sparse Search Retriever
    sparse_retriever = SparseRetriever()
    sparse_results = sparse_retriever.query(query=query, top_k=top_k)
    sparse_retriever.show_results(sparse_results)


def getCombinedContext(query: str = "What is a second brain?", top_k: int = 5, max_docs: int = 6,
                       will_print: bool = True):
    dense_retriever = DenseRetriever()
    sparse_retriever = SparseRetriever()
    combined_retriever = CombinedRetriever(dense_retriever, sparse_retriever)

    combined_results = combined_retriever.retrieve(query, top_k=top_k, max_docs=max_docs)
    if will_print:
        combined_retriever.show_results(combined_results)
    return combined_results


def getPrompt(query: str = "What is a second brain?", top_k: int = 5, max_docs: int = 6, will_print: bool = True):
    combined_result = getCombinedContext(query=query, top_k=top_k, max_docs=max_docs, will_print=False)
    prompt = build_prompt(query, combined_result, max_docs=max_docs, context_in="text")
    if will_print:
        print("Generated Prompt:\n", prompt)
    return prompt


def getGeneratedAnswer(query: str, top_k: int, max_docs: int):
    prompt = getPrompt(query=query, top_k=top_k, max_docs=max_docs, will_print=False)
    falcon_generator = FalconGenerator()
    answer = falcon_generator.generate_answer(prompt)
    return prompt, answer


def getResult(user_input):
    question = user_input
    top_k = 5
    max_docs = 5
    prompt, answer = getGeneratedAnswer(query=question, top_k=top_k, max_docs=max_docs)
    print(f"Prompt\n {prompt}\n\n")
    return answer


if __name__ == "__main__":
    query = "What is a second brain?"
    print(f"Basic RAG Question: {query}\n"
          f"Answer: {getResult(query)}")
