from retrievers.dense_retriever import DenseRetriever
from retrievers.sparse_retriever import SparseRetriever


def main():
    question = "What is a second brain?"

    # Dense Search Retriever
    dense_retriever = DenseRetriever()
    dense_results = dense_retriever.query(query=question, top_k=2)
    dense_retriever.show_results(dense_results)

    # Sparse Search Retriever
    sparse_retriever = SparseRetriever()
    sparse_results = sparse_retriever.query(query=question, top_k=2)
    sparse_retriever.show_results(sparse_results)

    # Merge contexts
    merged_context = sparse_results + dense_results

    # # Initialize components
    # api_key = "ai71-api-b1e07fa1-d007-41cd-8306-85fc952e12a6"
    # generator = FalconGenerator(api_key)
    # sparse = BM25Retriever()
    # dense = DenseRetriever(index_name="fineweb-dense-index")
    #
    # # Retrieve documents
    # sparse_results = sparse.retrieve(question, top_k=3)
    # dense_results = dense.retrieve(question, top_k=3)
    #
    # # Merge contexts
    # merged_context = sparse_results + dense_results
    #
    # # Build prompt
    # prompt = build_prompt(question, merged_context)
    #
    # # Generate answer
    # answer = generator.generate(prompt)
    # print(f"Q: {question}\nA: {answer}")


if __name__ == "__main__":
    main()
