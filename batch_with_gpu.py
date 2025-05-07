from coref.coref_batch import CorefResolver
from generator.batch_falcon_generator import FalconGenerator
from reranker.mxbai_reranker import MxbaiReranker
from reranker.test import NvidiaReranker
from retrievers.dense_retriever import DenseRetriever
from retrievers.sparse_retriever import SparseRetriever
from topic.topic_batch import TextTopicAnalyzer
from utils.combined_retriever import CombinedRetriever
from utils.prompt_template import get_refine_query, get_hypothetical_answer, build_batch_prompt
from utils.save_to_file import save_to_jsonl
from utils.utils import merge_scores_and_keep_positive_batch
import json
import time

# Global initialization of heavy components
dense_retriever = DenseRetriever()
sparse_retriever = SparseRetriever()
combined_retriever = CombinedRetriever(dense_retriever, sparse_retriever)

resolver = CorefResolver()
topic_analyzer = TextTopicAnalyzer(verbose=False)
falcon_generator = FalconGenerator()
mxbai_reranker = MxbaiReranker()
MODEL_NAME = "nvidia/nv-rerankqa-mistral-4b-v3"
API_KEY_ONE = "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"
API_KEY_TWO = "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"
API_KEY_THREE = "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"
nvidia_reranker = NvidiaReranker(model=MODEL_NAME, api_keys=[API_KEY_ONE, API_KEY_TWO, API_KEY_THREE])

question_path = "./data/question/questions.jsonl"
answer_path = "./data/answer/answers.jsonl"


def get_batch_retrieved_chunks(batch_queries):
    # Chunk are returned in a flat array, 20 item for each query
    return combined_retriever.batch_retrieve(queries=batch_queries, top_k=13, max_docs=20)


def get_batch_rerank_documents(batch_queries, retrieved_batch_chunks, use_mxbai_reranker=True, batch_size=20):
    coref_resolved_docs = resolver.resolve_batch_documents(retrieved_batch_chunks, batch_size=batch_size)

    # need a for loop here
    batch_combine_docs = []
    for i in range(0, len(coref_resolved_docs), 20): # Please note this 20 can't be changed because it is the retrieved chunk count
        current_batch = coref_resolved_docs[i:i + batch_size]
        current_combined_docs = topic_analyzer.resolve_topics(current_batch)
        batch_combine_docs.append(current_combined_docs)

    # Optional summarization here

    nvidia_reranked_docs = nvidia_reranker.batch_rerank(queries=batch_queries, batch_documents=batch_combine_docs)

    reranked_docs = (
        mxbai_reranker.score_batch_documents(batch_queries, nvidia_reranked_docs)
        if use_mxbai_reranker else nvidia_reranked_docs
    )

    print(f"reranked documents for {len(reranked_docs)} queries")
    print(f"first reranked docs size: {len(reranked_docs[0])}")
    return merge_scores_and_keep_positive_batch(reranked_docs)


# For query_refinement == Ture, instead of 20 use 40,
# obviously in that case u need to change get_batch_retrieved_chunks this function
def run_single_batch(questions, use_mxbai_reranker=True, use_query_refinement=True):
    batch_queries = [q["question"] for q in questions]
    batch_retrieved_chunks = get_batch_retrieved_chunks(batch_queries)

    batch_reranked_docs = get_batch_rerank_documents(batch_queries, batch_retrieved_chunks, use_mxbai_reranker)

    batch_prompts = build_batch_prompt(batch_queries, batch_reranked_docs, max_docs=10)
    batch_answer = falcon_generator.batch_generate_answer(batch_prompts, n_parallel=10)

    for reranked_docs, question, query, prompt, answer in zip(batch_reranked_docs, questions, batch_queries, batch_prompts, batch_answer):
        for doc in reranked_docs:
            # TODO Remove score in final, uncomment next line
            # doc.pop('score', None)
            doc.pop('source', None)

        final_output = {
            "id": question["id"],
            "question": query,
            "passages": reranked_docs[:10],
            "final_prompt": prompt,
            "answer": answer
        }

        save_to_jsonl(final_output, file_path=answer_path)
        print(final_output)
        print("\n")


# def run_pipeline(questions_path, use_mxbai_reranker=True):
#     with open(questions_path, 'r', encoding='utf-8') as f:
#         questions = [json.loads(line) for line in f]
#
#     for question in questions:
#         run_single(question, use_topic_combiner, use_mxbai_reranker)


if __name__ == "__main__":
    start_time = time.time()
    with open(question_path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]

    run_single_batch(questions)
    total_time = time.time() - start_time
    print(f"Processed {len(questions)} questions in {total_time:.2f} seconds")
    print(f"Took {(total_time/len(questions)):.2f} seconds each question")


