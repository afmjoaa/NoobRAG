from coref.coref_batch import CorefResolver
from generator.batch_falcon_generator import FalconGenerator
from generator.batch_mistral_generator import MistralGenerator
from reranker.mxbai_reranker import MxbaiReranker
from reranker.rerank_batch import NvidiaReranker
from retrievers.dense_retriever import DenseRetriever
from retrievers.sparse_retriever import SparseRetriever
from topic.topic_batch import TextTopicAnalyzer
from utils.combined_retriever import CombinedRetriever
from utils.prompt_template import build_batch_prompt, build_batch_summary_flat_prompt, get_batch_refine_query, \
    get_batch_hypothetical_answer
from utils.save_to_file import save_to_jsonl
from utils.utils import merge_scores_and_keep_positive_batch
import json
import time
from typing import List, Dict, Any
from itertools import chain
import argparse

start_time = time.time()
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

summary_generator = MistralGenerator(
        api_keys=["59457d62865a1e3f69ae32e7f42148fafb7a2ea972e2b1438f749514892b8c8c",
                  "2a6714a7f23ea83446e29cd1ac8c5fb4906aa720035c61fb779c92987db0b8aa",
                  "e38a2f42303bd976b218d8904116a473f78d1467b36015e730471d36a8c78d48",
                  "583c84df297987f1c992c063ce2a291deb9a8dcd3781764344021c82286c4eeb",
                  "6d3a4f46600c7c305f4dd4c5e99830da893515d3272cf868e87efae61af72b89"
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
    flat_summaries = falcon_generator.batch_generate_answer(flat_prompts)

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


def get_batch_retrieved_chunks(batch_queries) -> tuple[List[Dict], List[int]]:
    # Chunk are returned in a flat array, 20 item for each query
    batch_combine_retrieved_docs = combined_retriever.batch_retrieve(queries=batch_queries, top_k=20, max_docs=20, isFlat=False)
    # Get the lengths of each sublist
    batch_combine_retrieved_docs_count = [len(sublist) for sublist in batch_combine_retrieved_docs]

    # Flatten the list of lists
    flat_batch_combine_retrieved_docs = list(chain.from_iterable(batch_combine_retrieved_docs))
    return flat_batch_combine_retrieved_docs, batch_combine_retrieved_docs_count


def get_batch_refine_retrieved_chunks(batch_queries: List[str], batch_selected_refine_candidates: List[Dict]) -> tuple[List[Dict], List[int]]:
    # Chunk are returned in a flat array, 30 item for each query
    batch_previous_docs = [candidate["retrieved_chunks"] for candidate in batch_selected_refine_candidates]

    # get the refined question
    batch_refine_query_prompt = get_batch_refine_query(batch_queries)
    batch_refine_query = summary_generator.batch_generate_answer(batch_refine_query_prompt, batch_queries)
    batch_refine_query_docs = combined_retriever.batch_retrieve(queries=batch_refine_query, top_k=29, max_docs=15, batch_previous_docs=batch_previous_docs, isFlat=False)

    # get the refined answer
    current_batch_previous_docs = [a + b for a, b in zip(batch_previous_docs, batch_refine_query_docs)]
    batch_hypothetical_answer_prompt = get_batch_hypothetical_answer(batch_refine_query)
    batch_hypothetical_answer = summary_generator.batch_generate_answer(batch_hypothetical_answer_prompt, batch_queries)
    batch_hypothetical_answer_docs = combined_retriever.batch_retrieve(queries=batch_hypothetical_answer, top_k=38, max_docs=15, batch_previous_docs=current_batch_previous_docs, isFlat=False)

    print(f"\nOriginal query: {batch_queries[0]}\n"
          f"refined query: {batch_refine_query[0]}\n"
          f"hypothetical answer: {batch_hypothetical_answer[0]}\n")

    batch_combine_retrieved_docs = list(chain.from_iterable(
        a + b for a, b in zip(batch_refine_query_docs, batch_hypothetical_answer_docs)
    ))

    # Lengths of each combined a + b
    batch_combine_retrieved_docs_count = [len(a + b) for a, b in zip(batch_refine_query_docs, batch_hypothetical_answer_docs)]

    print(f"refine question count: {len(batch_queries)}")
    return batch_combine_retrieved_docs, batch_combine_retrieved_docs_count


def get_batch_rerank_documents(batch_queries, retrieved_batch_chunks, retrieved_batch_chunks_count, use_mxbai_reranker=True):
    coref_batch_size = 200
    print(f"retrieved_batch_chunks: {len(retrieved_batch_chunks)}")
    coref_resolved_docs = resolver.resolve_batch_documents(retrieved_batch_chunks, batch_size=coref_batch_size)

    # need a for loop here
    batch_combine_docs = []
    start_idx = 0

    for chunk_size in retrieved_batch_chunks_count:
        end_idx = start_idx + chunk_size
        current_batch = coref_resolved_docs[start_idx:end_idx]
        current_combined_docs = topic_analyzer.resolve_topics(current_batch)
        batch_combine_docs.append(current_combined_docs)
        start_idx = end_idx
    print(f"Processed {start_idx} documents from retrieved_batch_chunks\n")

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


def run_single_batch(questions, use_mxbai_reranker=True):
    batch_queries = [q["question"] for q in questions]
    batch_retrieved_chunks, batch_retrieved_chunks_count = get_batch_retrieved_chunks(batch_queries)

    batch_reranked_docs = get_batch_rerank_documents(
        batch_queries=batch_queries,
        retrieved_batch_chunks=batch_retrieved_chunks,
        retrieved_batch_chunks_count=batch_retrieved_chunks_count,
        use_mxbai_reranker=use_mxbai_reranker)

    indices_to_delete = []
    for k, rerank_docs in enumerate(batch_reranked_docs):
        print(f"{k} reranked docs have {len(rerank_docs)} positive docs")
        if len(rerank_docs) == 0 or (len(rerank_docs) > 0 and rerank_docs[0]["score"] <= 10):
            # Create and save the refine item
            refine_item = {
                "id": questions[k]["id"],
                "question": questions[k]["question"],
                "passages": rerank_docs,
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
        # print(f"{final_output}\n")


def run_refine_single_batch(batch_selected_refine_candidates: List[Dict], use_mxbai_reranker=True):
    batch_queries = [candidate["question"] for candidate in batch_selected_refine_candidates]
    batch_refined_retrieved_chunks, batch_refined_retrieved_chunks_count = get_batch_refine_retrieved_chunks(
        batch_queries=batch_queries,
        batch_selected_refine_candidates=batch_selected_refine_candidates)

    batch_refined_reranked_docs = get_batch_rerank_documents(
        batch_queries=batch_queries,
        retrieved_batch_chunks=batch_refined_retrieved_chunks,
        retrieved_batch_chunks_count=batch_refined_retrieved_chunks_count,
        use_mxbai_reranker=use_mxbai_reranker)

    batch_combine_reranked_docs = [a + b["passages"] for a, b in zip(batch_refined_reranked_docs, batch_selected_refine_candidates)]

    for k, entries in enumerate(batch_combine_reranked_docs):
        print(f"{k} reranked docs have {len(entries)} positive docs")
        entries.sort(key=lambda x: x['score'], reverse=True)

    batch_prompts = build_batch_prompt(batch_queries, batch_combine_reranked_docs, max_docs=10)
    batch_answer = falcon_generator.batch_generate_answer(batch_prompts, n_parallel=10)

    for reranked_docs, question, query, prompt, answer in zip(batch_combine_reranked_docs, batch_selected_refine_candidates, batch_queries,
                                                              batch_prompts, batch_answer):
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


# Maximum 100 questions at a time
if __name__ == "__main__":
    # start_time = time.time()
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--job-id', type=str, required=True, help='Slurm Job ID')
    args = parser.parse_args()
    print(f"JOBID: {args.job_id}\n")

    question_path = "./data/question/current_questions.jsonl"
    answer_path = f"./data/answer/answers_{args.job_id}.jsonl"
    refine_items_path = f"./data/refine/refine_items_{args.job_id}.jsonl"

    with open(question_path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]
    run_single_batch(questions)

    with open(refine_items_path, 'r', encoding='utf-8') as f:
        refine_items = [json.loads(line) for line in f]
    if len(refine_items) > 0:
        run_refine_single_batch(refine_items)

    total_time = time.time() - start_time
    total_question = len(questions) + len(refine_items)
    print(f"\nProcessed {total_question} questions in {total_time:.2f} seconds")
    print(f"Took {(total_time/total_question):.2f} seconds each question")


# Implement MPI here
# def run_pipeline(questions_path, use_mxbai_reranker=True):
#     with open(questions_path, 'r', encoding='utf-8') as f:
#         questions = [json.loads(line) for line in f]
#
#     for question in questions:
#         run_single(question, use_topic_combiner, use_mxbai_reranker)


