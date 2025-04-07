from generator.falcon_generator import FalconGenerator
from retrievers.bm25_opensearch import BM25Retriever
from retrievers.dense_pinecode import DenseRetriever
from utils.prompt_template import build_prompt


def main():
    question = "What is the capital of France?"

    # Initialize components
    api_key = "ai71-api-b1e07fa1-d007-41cd-8306-85fc952e12a6"
    generator = FalconGenerator(api_key)
    sparse = BM25Retriever()
    dense = DenseRetriever(index_name="fineweb-dense-index")

    # Retrieve documents
    sparse_results = sparse.retrieve(question, top_k=3)
    dense_results = dense.retrieve(question, top_k=3)

    # Merge contexts
    merged_context = sparse_results + dense_results

    # Build prompt
    prompt = build_prompt(question, merged_context)

    # Generate answer
    answer = generator.generate(prompt)
    print(f"Q: {question}\nA: {answer}")


if __name__ == "__main__":
    main()
