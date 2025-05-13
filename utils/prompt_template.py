from typing import List, Dict, Any
from multiprocessing.pool import ThreadPool


def build_prompt(query: str, docs: List[Dict[str, Any]], max_docs: int = 10, context_in: str = 'passage') -> str:
    """Construct a prompt from retrieved docs."""
    selected_docs = docs[:max_docs]
    context_parts = [
        f"Context Document {i + 1}: {doc[context_in]}"
        for i, doc in enumerate(selected_docs)
    ]
    context_block = "\n\n".join(context_parts)

    prompt = f"""
    Role: You are an information retrieval system that answers exclusively from provided context.

    \nContext:
    {context_block}\n

    Strict Instructions:
    1. Respond to the question using ONLY information present in the context
    2. If answer requires information not in context, respond EXACTLY: "No information available"
    3. Never:
       - Use prior knowledge
       - Speculate or make assumptions
    4. Preserve numerical values, technical terms and named entities exactly
    5. Make proper step by step calculations if formula and values are given
    6. Connecting statements if needed to answer the question clearly.
    7. Maximum length: 200 tokens
    
    \n
    Question: {query}\n
    Answer:""".strip()

    return prompt


def build_batch_prompt(queries: List[str], batch_docs: List[List[Dict]], max_docs: int = 10) -> List[str]:
    """Construct multiple prompts from batches of retrieved docs using parallel processing."""
    with ThreadPool() as pool:
        # Pair each query with its corresponding docs and apply build_prompt in parallel
        prompts = pool.starmap(
            build_prompt,
            [(query, docs, max_docs) for query, docs in zip(queries, batch_docs)]
        )
    return prompts


def get_refine_query(query: str) -> str:
    prompt = f"""
    You are an expert search query optimizer. Improve the following search query for a RAG system by following these strict instructions:

1. Identify the core information need and explicit/implicit requirements
2. Resolve ambiguous pronouns/nouns and replace vague terms with specific technical language
3. Make the query more specific, clear, and well-structured, while preserving its original intent
4. Output ONLY the final refined query without commentary or explanation
5. Query should be a single sentence, or two sentences at most.

Original Query:
"{query}"

Refined Query:
""".strip()
    return prompt


def get_batch_refine_query(queries: List[str]) -> List[str]:
    with ThreadPool() as pool:
        prompts = pool.starmap(
            get_refine_query,
            [query for query in zip(queries)]
        )
    return prompts


def get_hypothetical_answer(query: str) -> str:
    prompt = f"""
    You are a a domain expert, generate a concise and informative 2-3 line answer to the question below for retrieval.
    
Your answer must:
1. Contain essential factual claims and technical terminology
2. Include specific named entities (people, organizations, theories, dates)
3. Be exactly 2-3 sentences, optimized for maximum term overlap with relevant documents
4. Focus strictly on verifiable information; avoid speculation, opinions.
4. Output ONLY the Answer without commentary

Question: {query}

Answer:
""".strip()
    return prompt


def get_batch_hypothetical_answer(queries: List[str]) -> List[str]:
    with ThreadPool() as pool:
        prompts = pool.starmap(
            get_hypothetical_answer,
            [query for query in zip(queries)]
        )
    return prompts


def build_summary_prompt(query: str, context_paragraph: str) -> str:
    prompt = f"""
Instruction:
Create a concise summary containing ALL key information from the context paragraph below. Follow these strict rules:

1. Extract and list every factual element from the context
2. Preserve exact technical terms, measurements, relationships and named entities
3. Never add explanations, comparisons, or information not explicitly stated
4. The summary should be comprehensive, yet concise.
5. Strict maximum: 400 tokens

Context Paragraph:
{context_paragraph}
    """.strip()

    return prompt


def build_batch_summary_flat_prompt(queries: List[str], batch_docs: List[List[Dict]]) -> List[str]:
    # Flatten the list for parallel processing
    flat_args = [
        (queries[i], doc['passage'])
        for i, docs in enumerate(batch_docs)
        for doc in docs
    ]

    # Process in parallel
    with ThreadPool() as pool:
        flat_prompts = pool.starmap(build_summary_prompt, flat_args)

    # # Reshape prompts into the original structure (List[List[str]])
    # result = []
    # idx = 0
    # for docs in batch_docs:
    #     num_docs = len(docs)
    #     result.append(flat_prompts[idx:idx + num_docs])
    #     idx += num_docs

    return flat_prompts

# result = falcon_generator.generate_answer(prompt).strip()
#     print("\n[Refined Query]")
#     print(result)

# A strict length limit of 200 tokens
#
#     prompt = f"""
# Query: {query}
#
# Context Paragraph:
# {context_paragraph}
#
# Instruction:
# Summarize the context paragraph in relation to the query above. Focus only on the information relevant to answering the query. The summary should be comprehensive, yet concise, and must not exceed 450 tokens. Avoid repeating the query or including irrelevant details. Do not add any information that is not in the context paragraph.
# If the context paragraph does not directly provide information about the query, then just say "No information available"
# """.strip()

# (
#         "You are a knowledgeable assistant. Answer the following question using only the context provided below.\n\n"
#         "----\n"
#         f"{context_block}\n"
#         "----\n"
#         f"Question: {query}\n"
#         "Answer:"
#     )
