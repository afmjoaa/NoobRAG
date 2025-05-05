import asyncio
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

# Import your components
from coref.coref_resolver import CorefResolver
from generator.falcon_generator import FalconGenerator
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants - limiting to 4 cores
MAX_WORKERS = 4  # Using exactly 4 cores as requested
MAX_THREAD_WORKERS = 16  # For I/O bound tasks
MODEL_NAME = "nvidia/nv-rerankqa-mistral-4b-v3"
API_KEY = "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"  # TODO: Replace with your actual API key

# Resource management
torch.set_num_threads(2)  # Lower thread count per process


# Component initialization - lazy loading with caching
class ComponentCache:
    """Singleton to manage expensive component instances."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ComponentCache, cls).__new__(cls)
            cls._instance._components = {}
        return cls._instance

    def get_combined_retriever(self):
        if 'combined_retriever' not in self._components:
            dense_retriever = DenseRetriever()
            sparse_retriever = SparseRetriever()
            self._components['combined_retriever'] = CombinedRetriever(dense_retriever, sparse_retriever)
        return self._components['combined_retriever']

    def get_resolver(self):
        if 'resolver' not in self._components:
            self._components['resolver'] = CorefResolver()
        return self._components['resolver']

    def get_topic_analyzer(self):
        if 'topic_analyzer' not in self._components:
            self._components['topic_analyzer'] = TextTopicAnalyzer(verbose=False)
        return self._components['topic_analyzer']

    def get_generator(self):
        if 'generator' not in self._components:
            self._components['generator'] = FalconGenerator()
        return self._components['generator']

    def get_mxbai_reranker(self):
        if 'mxbai_reranker' not in self._components:
            self._components['mxbai_reranker'] = MxbaiReranker()
        return self._components['mxbai_reranker']

    def get_nvidia_reranker(self):
        if 'nvidia_reranker' not in self._components:
            self._components['nvidia_reranker'] = NvidiaReranker(model=MODEL_NAME, api_key=API_KEY)
        return self._components['nvidia_reranker']


# Main processing functions
async def get_all_retrieved_chunks(query: str, cache: ComponentCache) -> List[Dict]:
    """Retrieve chunks for a given query using the combined retriever."""
    combined_retriever = cache.get_combined_retriever()
    return combined_retriever.retrieve(query=query, top_k=13, max_docs=20)


async def refined_retrieval(query: str, classic_retrieved: List[Dict], cache: ComponentCache) -> Tuple[List[Dict], str]:
    """Refine retrieval using query reformulation and hypothetical answers."""
    combined_retriever = cache.get_combined_retriever()
    generator = cache.get_generator()

    # Run these tasks concurrently
    refine_query_task = asyncio.create_task(
        generate_refined_query(query, generator)
    )
    hypothetical_answer_task = asyncio.create_task(
        generate_hypothetical_answer(query, generator)
    )

    # Wait for both tasks to complete
    refine_query, hypothetical_answer = await asyncio.gather(
        refine_query_task, hypothetical_answer_task
    )

    # Retrieve documents concurrently
    rq_docs_task = asyncio.create_task(
        retrieve_docs(refine_query, combined_retriever)
    )
    ha_docs_task = asyncio.create_task(
        retrieve_docs(hypothetical_answer, combined_retriever)
    )

    rq_docs, ha_docs = await asyncio.gather(rq_docs_task, ha_docs_task)

    # Process results efficiently
    unique_docs = {}
    classic_doc_ids = {doc["doc_id"] for doc in classic_retrieved}

    # Deduplicate and combine results
    for doc in rq_docs + ha_docs:
        if doc["doc_id"] not in unique_docs and doc["doc_id"] not in classic_doc_ids:
            unique_docs[doc["doc_id"]] = doc

    filtered_docs = list(unique_docs.values())

    return filtered_docs[:20], refine_query


async def generate_refined_query(query: str, generator) -> str:
    """Generate a refined query."""
    refine_query_prompt = get_refine_query(query)
    refine_query = generator.generate_answer(refine_query_prompt)
    logger.debug(f"Refined query: {refine_query}")
    return refine_query


async def generate_hypothetical_answer(query: str, generator) -> str:
    """Generate a hypothetical answer."""
    hypothetical_answer_prompt = get_hypothetical_answer(query)
    hypothetical_answer = generator.generate_answer(hypothetical_answer_prompt)
    logger.debug(f"Hypothetical answer: {hypothetical_answer}")
    return hypothetical_answer


async def retrieve_docs(query: str, retriever) -> List[Dict]:
    """Retrieve documents for a given query."""
    return retriever.retrieve(query=query, top_k=12, max_docs=20)


async def get_rerank_documents(
        query: str,
        retrieved_chunks: List[Dict],
        use_topic_combiner: bool = True,
        use_mxbai_reranker: bool = True,
        cache: ComponentCache = None
) -> List[Dict]:
    """Rerank documents based on query relevance."""
    if cache is None:
        cache = ComponentCache()

    resolver = cache.get_resolver()
    topic_analyzer = cache.get_topic_analyzer()
    nvidia_reranker = cache.get_nvidia_reranker()
    mxbai_reranker = cache.get_mxbai_reranker()

    # Coreference resolution
    coref_resolved_docs = resolver.resolve_documents(retrieved_chunks)

    # Topic segmentation
    segmented_docs = topic_analyzer.resolve_topic(coref_resolved_docs)

    # Conditionally cluster documents
    if len(segmented_docs) > 10 and use_topic_combiner:
        combined_docs = cluster_documents_with_berttopic(segmented_docs)
    else:
        combined_docs = segmented_docs

    # Primary reranking
    nvidia_reranked_docs = nvidia_reranker.rerank_documents(
        query=query, documents=combined_docs
    )

    # Optional secondary reranking
    if use_mxbai_reranker:
        reranked_docs = mxbai_reranker.score_documents(query, nvidia_reranked_docs)
    else:
        reranked_docs = nvidia_reranked_docs

    logger.debug(f"Reranked docs size: {len(reranked_docs)}")
    return merge_scores_and_keep_positive(reranked_docs)


async def process_question(
        question: Dict,
        use_topic_combiner: bool = True,
        use_mxbai_reranker: bool = True,
        use_query_refinement: bool = True
) -> Dict:
    """Process a single question through the entire pipeline."""
    cache = ComponentCache()
    generator = cache.get_generator()

    query = question["question"]
    start_time = time.time()

    try:
        # Initial retrieval
        retrieved_chunks = await get_all_retrieved_chunks(query, cache)

        # Reranking
        reranked_docs = await get_rerank_documents(
            query, retrieved_chunks, use_topic_combiner, use_mxbai_reranker, cache
        )

        # Optional query refinement
        if use_query_refinement:
            refined_chunks, refine_query = await refined_retrieval(query, retrieved_chunks, cache)
            refined_reranked_docs = await get_rerank_documents(
                refine_query, refined_chunks, use_topic_combiner, use_mxbai_reranker, cache
            )

            # Combine and sort results
            reranked_docs.extend(refined_reranked_docs)
            reranked_docs.sort(key=lambda x: x['score'], reverse=True)

        # Build prompt and generate answer
        prompt = build_prompt(query, reranked_docs, max_docs=10)
        answer = generator.generate_answer(prompt)

        # Clean up document data
        for doc in reranked_docs:
            doc.pop('source', None)

        result = {
            "id": question["id"],
            "question": query,
            "passages": reranked_docs[:10],
            "final_prompt": prompt,
            "answer": answer,
            "processing_time": round(time.time() - start_time, 2)
        }

        # Save results asynchronously
        save_to_jsonl(result)

        return result

    except Exception as e:
        logger.error(f"Error processing question {question['id']}: {str(e)}")
        return {
            "id": question["id"],
            "question": query,
            "error": str(e),
            "processing_time": round(time.time() - start_time, 2)
        }


def process_in_worker_pool(
        question: Dict,
        use_topic_combiner: bool = True,
        use_mxbai_reranker: bool = True,
        use_query_refinement: bool = True
) -> Dict:
    """Process a single question (worker pool wrapper)."""
    return asyncio.run(process_question(
        question, use_topic_combiner, use_mxbai_reranker, use_query_refinement
    ))


def run_pipeline(
        questions_path: str,
        use_topic_combiner: bool = True,
        use_mxbai_reranker: bool = True,
        use_query_refinement: bool = True,
        output_path: str = None
) -> List[Dict]:
    """Run the complete pipeline using process pool for parallelism."""
    start_time = time.time()

    # Load questions
    logger.info(f"Loading questions from {questions_path}")
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]

    # Process questions in parallel using process pool with 4 workers
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        process_func = partial(
            process_in_worker_pool,
            use_topic_combiner=use_topic_combiner,
            use_mxbai_reranker=use_mxbai_reranker,
            use_query_refinement=use_query_refinement
        )

        for result in tqdm(
                executor.map(process_func, questions),
                total=len(questions),
                desc="Processing questions"
        ):
            results.append(result)

    # Sort by original ID
    results.sort(key=lambda x: x["id"])

    # Save to output file if specified
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    total_time = time.time() - start_time
    logger.info(f"Processed {len(results)} questions in {total_time:.2f} seconds")
    logger.info(f"Average time per question: {total_time / len(results):.2f} seconds")

    return results


def run_single(
        question: Dict,
        use_topic_combiner: bool = True,
        use_mxbai_reranker: bool = True,
        use_query_refinement: bool = True
) -> Dict:
    """Process a single question (for API or individual testing)."""
    return asyncio.run(process_question(
        question, use_topic_combiner, use_mxbai_reranker, use_query_refinement
    ))


if __name__ == "__main__":
    # Run the pipeline with 4 cores
    run_pipeline(
        test_question_path,
        use_topic_combiner=True,
        use_mxbai_reranker=True,
        use_query_refinement=True,
        output_path="../data/answer/answers.jsonl"
    )