import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict


class MxbaiReranker:
    def __init__(self, model_name="mixedbread-ai/mxbai-rerank-large-v1"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def score_documents(self, query, documents):
        """
        Scores a list of document dictionaries based on their relevance to the query
        and adds mxbai_score to each document.

        Args:
            query (str): The query string
            documents (list): List of document dictionaries containing 'text' and 'uid' fields

        Returns:
            list: The original documents with added 'mxbai_score' field
        """
        # Extract text passages while keeping track of original indices
        passages = [doc['passage'] for doc in documents]

        # Format inputs for the model
        inputs = [f"query: {query} passage: {passage}" for passage in passages]

        # Tokenize inputs
        tokenized = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**tokenized)

        # Get the [CLS] token embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        # Apply a linear layer to get the score
        # Using the model's first hidden dimension as weights for simplicity
        weights = torch.nn.Linear(cls_embeddings.size(1), 1).to(self.device)
        scores = weights(cls_embeddings).squeeze().cpu().tolist()

        # Add scores to the original documents
        scored_documents = documents.copy()
        for i, score in enumerate(scores):
            scored_documents[i]['mxbai_score'] = float(score)

        return scored_documents

    def score_batch_documents(self, queries: List[str], batch_documents: List[List[Dict]]) -> List[List[Dict]]:
        """
        Scores multiple batches of documents for corresponding queries using batch inference,
        adding mxbai_score to each document.

        Args:
            queries (List[str]): List of query strings
            batch_documents (List[List[Dict]]): List of document batches, where each batch corresponds to a query
                                                and contains dictionaries with 'text' and 'uid' fields

        Returns:
            List[List[Dict]]: The original document batches with added 'mxbai_score' field for each document
        """
        # Validate input lengths
        if len(queries) != len(batch_documents):
            print(f"length of queries: {len(queries)} and length of batch_documents: {len(batch_documents)}")
            raise ValueError("Number of queries must match number of document batches")

        # Prepare input strings and track document counts per query
        input_strings = []
        lengths = []
        for query, docs in zip(queries, batch_documents):
            current_length = len(docs)
            lengths.append(current_length)
            for doc in docs:
                input_str = f"query: {query} passage: {doc['passage']}"
                input_strings.append(input_str)

        # Handle empty input case
        if not input_strings:
            return [docs.copy() for docs in batch_documents]

        # Batch tokenization
        tokenized = self.tokenizer(
            input_strings,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Model inference
        with torch.no_grad():
            outputs = self.model(**tokenized)

        # Extract CLS embeddings and compute scores
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        linear_layer = torch.nn.Linear(cls_embeddings.size(-1), 1).to(self.device)
        scores = linear_layer(cls_embeddings).squeeze(-1).cpu().tolist()

        # Split scores into original batches and add to documents
        scored_batches = []
        start_idx = 0
        for i, length in enumerate(lengths):
            end_idx = start_idx + length
            batch_scores = scores[start_idx:end_idx]

            # Create new documents with scores
            original_docs = batch_documents[i]
            scored_docs = []
            for doc, score in zip(original_docs, batch_scores):
                # Create copy to avoid modifying original input
                new_doc = doc.copy()
                new_doc['mxbai_score'] = float(score)
                scored_docs.append(new_doc)

            scored_batches.append(scored_docs)
            start_idx = end_idx

        return scored_batches


if __name__ == "__main__":
    # Single query example using batch interface
    query = "What is a second brain?"
    input_documents = [
        {'doc_id': '<urn:uuid:0cf75b43-d690-4aa6-b3ca-f488ceb28ed9>', 'text': "What will the age of Aquarius be like. What is second level consciousness and are we still there? Left and Right Brain technology what is Left and Right Brain technology? part 1\nSorry we couldn't complete your registration.", 'score': 0.835362315, 'source': 'dense'},
        {'doc_id': '<urn:uuid:0cf75b43-d690-4aa6-b3ca-f488ceb28ed9>', 'text': 'Please try again. You must accept the Terms and conditions to register.', 'score': 0.835362315, 'source': 'dense'},
        {'doc_id': '<urn:uuid:b6541cf6-d442-454a-9e6f-45eaaa237424>', 'text': "Title: Left-Brain/Right-Brain Functions\nPreview: We have two eyes, two ears, two hands, and two minds. Remembering a persons name is a function of the left-brain memory while rembering a persons's face is a function. By aterry (adrienne)\non September 27, 2012.", 'score': 0.823291481, 'source': 'dense'},
        {'doc_id': '<urn:uuid:b6541cf6-d442-454a-9e6f-45eaaa237424>', 'text': 'Our left brain thinks in terms of words and symbols while our right brain thinks in terms of images. Our left brain is the side used more by writiers, mathematicians, and scientists; the right side by artists, craftspeople, and musicians.', 'score': 0.823291481, 'source': 'dense'},
        {'doc_id': '<urn:uuid:e4bf2415-2032-4a8a-9c18-715cf2d5f91f>', 'text': 'The legal term for this compensation is “damages. ” Exactly what damages you can recover varies from state to state, but you can usually recover:\n- Past and future medical expenses\n- Future lost wages (if the injury limits your ability to work in the future)\n- Property damages\n- Pain and suffering\n- Emotional distress\nReady to contact a lawyer about a possible second impact syndrome case? Use our free online directory to schedule your initial consultation today. - Guide to traumatic brain injuries\n- Resources to help after a brain injury\n- How to recognize a brain injury and what you should do about a brain injury\n- Concussions and auto accidents\n- Rehabilitation and therapy after a brain injury\n- Second impact syndrome and sports injury lawsuits\n- Legal guide to brain death\n- What is CTE?\n- A loss of oxygen can lead to an anoxic brain injury\n- Can you recover costs for the accident that caused a brain bleed?\n- What is the Traumatic Brain Injury Act?\n- Understanding the Hidden Challenges of Mild Traumatic Brain Injury\n- What is the Glasgow Coma Scale?.', 'score': 0.24633245645981225, 'source': 'sparse'}]

    mxbai_reranker = MxbaiReranker()
    scored_batches = mxbai_reranker.score_batch_documents([query, query], [input_documents, input_documents])

    print(scored_batches[0])
    print(scored_batches)