from mpi4py import MPI
import json
from coref.coref_resolver import CorefResolver
from generator.falcon_generator import FalconGenerator
from basic_rag import getCombinedContext
from reranker.mxbai_reranker import MxbaiReranker
from reranker.nvidia_reranker import NvidiaReranker
from topic.combine_topics import cluster_documents_with_berttopic
from topic.topic_segmenter import TextTopicAnalyzer
from utils.prompt_template import build_prompt
from utils.save_to_file import save_to_jsonl
from utils.utils import merge_scores_and_keep_positive

# Pre-initialize reusable components
coref_resolver = CorefResolver()
topic_analyzer = TextTopicAnalyzer(verbose=False)
falcon_generator = FalconGenerator()

# You could also rotate keys with rank if you have a pool of API keys
MODEL_NAME = "nvidia/nv-rerankqa-mistral-4b-v3"
API_KEY = "nvapi----" # TODO: Replace with your actual API key

nvidia_reranker = NvidiaReranker(model=MODEL_NAME, api_key=API_KEY)
mxbai_reranker = MxbaiReranker()


def get_all_retrieved_chunks(query):
    return getCombinedContext(query=query, top_k=6, max_docs=10, will_print=False)


def get_rerank_documents(query, retrieved_chunks, use_topic_combiner=True, use_mxbai_reranker=True):
    coref_resolved_docs = coref_resolver.resolve_documents(retrieved_chunks)
    segmented_docs = topic_analyzer.resolve_topic(coref_resolved_docs)

    if len(segmented_docs) > 10 and use_topic_combiner:
        combined_docs = cluster_documents_with_berttopic(segmented_docs)
    else:
        combined_docs = segmented_docs

    nvida_reranked_docs = nvidia_reranker.rerank_documents(query=query, documents=combined_docs)

    if use_mxbai_reranker:
        mxbai_reranked_documents = mxbai_reranker.score_documents(query, nvida_reranked_docs)
    else:
        mxbai_reranked_documents = nvida_reranked_docs

    return merge_scores_and_keep_positive(mxbai_reranked_documents)


def run_single(question, use_topic_combiner=True, use_mxbai_reranker=True):
    query = question["question"]
    retrieved_chunks = get_all_retrieved_chunks(query)
    reranked_docs = get_rerank_documents(query, retrieved_chunks, use_topic_combiner, use_mxbai_reranker)
    prompt = build_prompt(query, reranked_docs, max_docs=10)
    answer = falcon_generator.generate_answer(prompt)

    for doc in reranked_docs:
        doc.pop('score', None)
        doc.pop('source', None)

    final_output = {
        "id": question["id"],
        "question": query,
        "passages": reranked_docs,
        "final_prompt": prompt,
        "answer": answer
    }

    save_to_jsonl(final_output)
    return final_output


def run_pipeline(questions_path, use_topic_combiner=True, use_mxbai_reranker=True):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Load and scatter the questions
        with open(questions_path, 'r') as f:
            questions = [json.loads(line) for line in f]
        chunks = [questions[i::size] for i in range(size)]
    else:
        chunks = None

    # Scatter the question chunks to each process
    local_questions = comm.scatter(chunks, root=0)

    # Process locally assigned questions
    local_results = []
    for question in local_questions:
        try:
            result = run_single(question, use_topic_combiner, use_mxbai_reranker)
            local_results.append(result)
        except Exception as e:
            print(f"[Rank {rank}] Error processing question {question['id']}: {e}")

    # Gather results from all ranks
    all_results = comm.gather(local_results, root=0)

    # Combine and optionally save all results at rank 0
    if rank == 0:
        flat_results = [item for sublist in all_results for item in sublist]
        print(f"Total processed: {len(flat_results)}")
        return flat_results
