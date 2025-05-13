from dotenv import load_dotenv

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
import os, json


start_time = time.time()
load_dotenv()

# Global initialization of heavy components
dense_retriever = DenseRetriever()
sparse_retriever = SparseRetriever()
combined_retriever = CombinedRetriever(dense_retriever, sparse_retriever)

resolver = CorefResolver()
topic_analyzer = TextTopicAnalyzer(verbose=False)
falcon_generator = FalconGenerator()
mxbai_reranker = MxbaiReranker()
MODEL_NAME = "nvidia/nv-rerankqa-mistral-4b-v3"
nvidia_api_keys = json.loads(os.getenv("NVIDIA_API_KEYS", "[]"))
nvidia_reranker = NvidiaReranker(model=MODEL_NAME, api_keys=nvidia_api_keys)

mistral_api_keys = json.loads(os.getenv("TOGETHER_AI_API_KEYS", "[]"))
summary_generator = MistralGenerator(
        api_keys=mistral_api_keys,  # Add your actual API keys
        rpm_limit=59  # Per-key RPM limit
    )

refine_item_list = []


def update_passages_with_summaries(
        batch_combine_docs: List[List[Dict[str, Any]]],
        batch_queries: List[str]
) -> List[List[Dict[str, Any]]]:
    # Generate prompts in flat structure (List[str])
    flat_prompts = build_batch_summary_flat_prompt(batch_queries, batch_combine_docs)

    # Generate summaries in one batch call
    # flat_summaries = summary_generator.batch_generate_answer(flat_prompts)
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
    # batch_refine_query = falcon_generator.batch_generate_answer(batch_refine_query_prompt)
    batch_refine_query_docs = combined_retriever.batch_retrieve(queries=batch_refine_query, top_k=29, max_docs=15, batch_previous_docs=batch_previous_docs, isFlat=False)

    # get the refined answer
    current_batch_previous_docs = [a + b for a, b in zip(batch_previous_docs, batch_refine_query_docs)]
    batch_hypothetical_answer_prompt = get_batch_hypothetical_answer(batch_refine_query)
    batch_hypothetical_answer = summary_generator.batch_generate_answer(batch_hypothetical_answer_prompt, batch_queries)
    # batch_hypothetical_answer = falcon_generator.batch_generate_answer(batch_hypothetical_answer_prompt)
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

    try:
        nvidia_batch_reranked_docs = nvidia_reranker.batch_rerank(
            queries=batch_queries,
            batch_documents=summary_batch_combine_docs)
    except Exception:
        nvidia_batch_reranked_docs = []

    if len(nvidia_batch_reranked_docs) == 0:
        nvidia_batch_reranked_docs = summary_batch_combine_docs

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
            refine_item_list.append(refine_item)
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
            doc.pop('score', None)
            doc.pop('source', None)

        final_output = {
            "id": question["id"],
            "question": query,
            "passages": reranked_docs[:10],
            "final_prompt": prompt,
            "answer": answer
        }

        save_to_jsonl(final_output, file_path=answer_path)


if __name__ == "__main__":
    # start_time = time.time()
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--job-id', type=str, required=True, help='Slurm Job ID')
    parser.add_argument('--task-id', type=int, required=True, help='Task ID')
    args = parser.parse_args()
    print(f"JOBID: {args.job_id}\n")

    # question_path = "./data/question/test_questions.jsonl"
    question_path = "./data/question/challenge_questions.jsonl"
    answer_path = f"./data/answer/answers_{args.job_id}_{args.task_id}.jsonl"
    refine_items_path = f"./data/refine/refine_items_{args.job_id}_{args.task_id}.jsonl"
    n_parallel = 5

    with open(question_path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]

    question_chunks = [questions[i::n_parallel] for i in range(n_parallel)]
    run_single_batch(question_chunks[args.task_id])

    # with open(refine_items_path, 'r', encoding='utf-8') as f:
    #     refine_items = [json.loads(line) for line in f]
    print(f"Refine item length is {len(refine_item_list)}")
    if len(refine_item_list) > 0:
        run_refine_single_batch(refine_item_list)
        total_question = len(question_chunks[args.task_id]) + len(refine_item_list)
    else:
        total_question = len(question_chunks[args.task_id])

    total_time = time.time() - start_time
    print(f"\nTaskID: {args.task_id} Processed {total_question} questions in {total_time:.2f} seconds")
    print(f"TaskID: {args.task_id} Took {(total_time/total_question):.2f} seconds each question")