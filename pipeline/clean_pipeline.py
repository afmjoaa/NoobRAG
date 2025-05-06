from coref.coref_resolver import CorefResolver
from generator.falcon_generator import FalconGenerator
from main import getCombinedContext
from reranker.mxbai_reranker import MxbaiReranker
from reranker.nvidia_reranker import NvidiaReranker
from retrievers.dense_retriever import DenseRetriever
from retrievers.sparse_retriever import SparseRetriever
from topic.combine_topics import cluster_documents_with_berttopic
from topic.topic_segmenter import TextTopicAnalyzer
from utils.combined_retriever import CombinedRetriever
from utils.prompt_template import build_prompt, get_refine_query, get_hypothetical_answer
from utils.save_to_file import save_to_jsonl
from utils.utils import test_question_path, merge_scores_and_keep_positive
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
# TODO: API KEY ROTATION NEED TO BE IMPLEMENTED
MODEL_NAME = "nvidia/nv-rerankqa-mistral-4b-v3"
API_KEY = "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"
nvidia_reranker = NvidiaReranker(model=MODEL_NAME, api_key=API_KEY)


def get_all_retrieved_chunks(query):
    return combined_retriever.retrieve(query=query, top_k=13, max_docs=20)


def refined_retrieval(query, classic_retrieved):
    # TODO: Change the model to mistral, and run refined_retrieval in parallel

    # get the refined question
    refine_query_prompt = get_refine_query(query)
    refine_query = falcon_generator.generate_answer(refine_query_prompt)
    print(f"refined query: {refine_query}")
    rq_docs = combined_retriever.retrieve(query=refine_query, top_k=12, max_docs=20)

    # get the refined answer
    hypothetical_answer_prompt = get_hypothetical_answer(refine_query)
    hypothetical_answer = falcon_generator.generate_answer(hypothetical_answer_prompt)
    print(f"hypothetical answer: {hypothetical_answer}")
    ha_docs = combined_retriever.retrieve(query=hypothetical_answer, top_k=12, max_docs=20)

    # remove duplicate retrieval and create refined retrieval
    unique_docs = {}

    for doc in rq_docs:
        unique_docs[doc["doc_id"]] = doc

    for doc in ha_docs:
        if doc["doc_id"] not in unique_docs:
            unique_docs[doc["doc_id"]] = doc

    # Step 2: Build a set of doc_ids from classic_retrieved
    classic_doc_ids = {doc["doc_id"] for doc in classic_retrieved}

    # Step 3: Filter out documents present in classic_retrieved
    filtered_docs = [doc for doc_id, doc in unique_docs.items() if doc_id not in classic_doc_ids]

    # print(len(classic_retrieved), classic_retrieved)
    # print(len(filtered_docs), filtered_docs)
    return filtered_docs[:20], refine_query


def get_rerank_documents(query, retrieved_chunks, use_topic_combiner=True, use_mxbai_reranker=True):
    coref_resolved_docs = resolver.resolve_documents(retrieved_chunks)
    segmented_docs = topic_analyzer.resolve_topic(coref_resolved_docs)

    combined_docs = (
        cluster_documents_with_berttopic(segmented_docs)
        if len(segmented_docs) > 10 and use_topic_combiner
        else segmented_docs
    )

    nvidia_reranked_docs = nvidia_reranker.rerank_documents(query=query, documents=combined_docs)

    reranked_docs = (
        mxbai_reranker.score_documents(query, nvidia_reranked_docs)
        if use_mxbai_reranker else nvidia_reranked_docs
    )

    print(f"reranked docs size: {len(reranked_docs)}")
    return merge_scores_and_keep_positive(reranked_docs)


def run_single(question, use_topic_combiner=True, use_mxbai_reranker=True, use_query_refinement=True):
    query = question["question"]
    retrieved_chunks = get_all_retrieved_chunks(query)
    reranked_docs = get_rerank_documents(query, retrieved_chunks, use_topic_combiner, use_mxbai_reranker)

    # Optional query refinement steps
    if use_query_refinement:
        refine_retrieved_chunks, refine_query = refined_retrieval(query, retrieved_chunks)
        refined_reranked_docs = get_rerank_documents(refine_query, refine_retrieved_chunks, use_topic_combiner, use_mxbai_reranker)
        reranked_docs.extend(refined_reranked_docs)
        reranked_docs.sort(key=lambda x: x['score'], reverse=True)

    print(f"len(reranked_docs): {len(reranked_docs)}, {reranked_docs}")
    prompt = build_prompt(query, reranked_docs, max_docs=10)
    answer = falcon_generator.generate_answer(prompt)

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

    save_to_jsonl(final_output)
    return final_output


def run_pipeline(questions_path, use_topic_combiner=True, use_mxbai_reranker=True):
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]

    for question in questions:
        run_single(question, use_topic_combiner, use_mxbai_reranker)


if __name__ == "__main__":
    # Run the full pipeline
    # run_pipeline(test_question_path)

    # # Test refined retrieval
    # query = "What is the difference between the default sampling rate used in audio CDs versus the recommended sampling rate for improving amp sim sound quality?"
    # retrieved_chunks = get_all_retrieved_chunks(query)
    # refined_retrieval(query, retrieved_chunks)

    # For single query test
    start_time = time.time()
    query = "What is the difference between the default sampling rate used in audio CDs versus the recommended sampling rate for improving amp sim sound quality?"
    question = {"id": 11, "question": query}
    print(run_single(question, use_query_refinement=False))
    total_time = time.time() - start_time
    print(f"Processed 1 questions in {total_time:.2f} seconds without query refinement")

    # # get rerank document test
    # retrieved_chunks = get_all_retrieved_chunks(query)
    # print(get_rerank_documents(query, retrieved_chunks))
