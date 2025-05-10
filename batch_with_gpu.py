from coref.coref_batch import CorefResolver
from generator.batch_falcon_generator import FalconGenerator
from generator.batch_mistral_generator import MistralGenerator
from reranker.mxbai_reranker import MxbaiReranker
from reranker.rerank_batch import NvidiaReranker
from retrievers.dense_retriever import DenseRetriever
from retrievers.sparse_retriever import SparseRetriever
from topic.topic_batch import TextTopicAnalyzer
from utils.combined_retriever import CombinedRetriever
from utils.prompt_template import build_batch_prompt, build_batch_summary_flat_prompt
from utils.save_to_file import save_to_jsonl
from utils.utils import merge_scores_and_keep_positive_batch
import json
import time
from typing import List, Dict, Any


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
API_KEY_FOUR = "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"
nvidia_reranker = NvidiaReranker(model=MODEL_NAME, api_keys=[API_KEY_ONE, API_KEY_TWO, API_KEY_THREE, API_KEY_FOUR])

question_path = "./data/question/questions.jsonl"
answer_path = "./data/answer/answers.jsonl"
refine_items_path = "./data/refine/refine_items.jsonl"

summary_generator = MistralGenerator(
        api_keys=["59457d62865a1e3f69ae32e7f42148fafb7a2ea972e2b1438f749514892b8c8c",
                  "2a6714a7f23ea83446e29cd1ac8c5fb4906aa720035c61fb779c92987db0b8aa",
                  "e38a2f42303bd976b218d8904116a473f78d1467b36015e730471d36a8c78d48",
                  "583c84df297987f1c992c063ce2a291deb9a8dcd3781764344021c82286c4eeb"
                  ],  # Add your actual API keys
        rpm_limit=59  # Per-key RPM limit
    )


def update_passages_with_summaries(
        batch_combine_docs: List[List[Dict[str, Any]]],
        batch_queries: List[str]
) -> List[List[Dict[str, Any]]]:
    # Generate prompts in flat structure (List[str])
    flat_prompts = build_batch_summary_flat_prompt(batch_queries, batch_combine_docs)

    # Generate summaries in one batch call
    flat_summaries = summary_generator.batch_generate_answer(flat_prompts)

    # Reshape summaries to match original nested structure
    nested_summaries = []
    cursor = 0
    for doc_group in batch_combine_docs:
        num_docs = len(doc_group)
        nested_summaries.append(flat_summaries[cursor:cursor + num_docs])
        cursor += num_docs

    # Create updated document structure with new passages
    updated_docs = []
    for query_idx, doc_group in enumerate(batch_combine_docs):
        new_group = []
        for doc_idx, doc in enumerate(doc_group):
            # Preserve metadata while updating passage
            updated_doc = {
                **doc,
                'passage': nested_summaries[query_idx][doc_idx]
            }
            new_group.append(updated_doc)
        updated_docs.append(new_group)

    return updated_docs


def get_batch_retrieved_chunks(batch_queries):
    # Chunk are returned in a flat array, 20 item for each query
    return combined_retriever.batch_retrieve(queries=batch_queries, top_k=13, max_docs=20)


def get_batch_rerank_documents(batch_queries, retrieved_batch_chunks, use_mxbai_reranker=True):
    coref_batch_size = 100
    coref_resolved_docs = resolver.resolve_batch_documents(retrieved_batch_chunks, batch_size=coref_batch_size)

    # need a for loop here
    batch_combine_docs = []
    for i in range(0, len(coref_resolved_docs), 20): # Please note this 20 can't be changed because it is the retrieved chunk count
        current_batch = coref_resolved_docs[i:i + 20]
        current_combined_docs = topic_analyzer.resolve_topics(current_batch)
        batch_combine_docs.append(current_combined_docs)

    # Optional summarization here
    summary_batch_combine_docs = update_passages_with_summaries(
        batch_combine_docs=batch_combine_docs,
        batch_queries=batch_queries,
    )

    nvidia_batch_reranked_docs = nvidia_reranker.batch_rerank(queries=batch_queries, batch_documents=summary_batch_combine_docs)

    batch_reranked_docs = (
        mxbai_reranker.score_batch_documents(batch_queries, nvidia_batch_reranked_docs)
        if use_mxbai_reranker else nvidia_batch_reranked_docs
    )

    # TODO remove the print statement
    print(f"reranked documents for {len(batch_reranked_docs)} queries")
    for j, rerank_docs in enumerate(batch_reranked_docs):
        print(f"{j} reranked docs size: {len(rerank_docs)}")

    return merge_scores_and_keep_positive_batch(batch_reranked_docs)


# For query_refinement == Ture, instead of 20 use 40,
# obviously in that case u need to change get_batch_retrieved_chunks this function
def run_single_batch(questions, use_mxbai_reranker=True, use_query_refinement=True):
    batch_queries = [q["question"] for q in questions]
    batch_retrieved_chunks = get_batch_retrieved_chunks(batch_queries)

    batch_reranked_docs = get_batch_rerank_documents(batch_queries, batch_retrieved_chunks, use_mxbai_reranker)

    indices_to_delete = []

    for k, rerank_docs in enumerate(batch_reranked_docs):
        # TODO remove the print statement
        print(f"{k} reranked docs have {len(rerank_docs)} positive docs")
        if len(rerank_docs) == 0 or (len(rerank_docs) > 0 and rerank_docs[0]["score"] <= 10):
            # Create and save the refine item
            refine_item = {
                "id": questions[k]["id"],
                "question": questions[k]["question"],
                "rerank_docs": rerank_docs,
                "retrieved_chunks": batch_retrieved_chunks[k * 20: (k + 1) * 20]
            }
            save_to_jsonl(refine_item, file_path=refine_items_path)

            # Mark index for deletion
            indices_to_delete.append(k)

    # Delete in reverse to avoid index shifting issues
    for idx in reversed(indices_to_delete):
        questions.pop(idx)
        batch_queries.pop(idx)
        batch_reranked_docs.pop(idx)

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
        # print(final_output)
        # print("\n")


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


