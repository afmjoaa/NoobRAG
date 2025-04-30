import torch
from transformers import AutoModel, AutoTokenizer


class SimpleReranker:
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
        passages = [doc['text'] for doc in documents]

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


# Example usage
if __name__ == "__main__":
    # Initialize reranker
    reranker = SimpleReranker()

    # Example query
    query = "What is Left and Right Brain technology?"

    # Example documents in the specified format
    documents = [
        {'doc_id': '<urn:uuid:0cf75b43-d690-4aa6-b3ca-f488ceb28ed9>',
         'text': "What will the age of Aquarius be like. What is second level consciousness and are we still there? Left and Right Brain technology what is Left and Right Brain technology? part 1\nSorry we couldn't complete your registration. Please try again. You must accept the Terms and conditions to register.",
         'score': 13.5859375,
         'source': 'dense',
         'uid': 'bd4777f1-e752-4cee-99b9-3be6ab9a7429'},

        {'doc_id': '<urn:uuid:b6541cf6-d442-454a-9e6f-45eaaa237424>',
         'text': 'Title: Left-Brain/Right-Brain Functions\nPreview: We have two eyes, two ears, two hands, and two minds. Our left brain is the side used more by writiers, mathematicians, and scientists; the right side by artists, craftspeople, and musicians. By aterry (adrienne)\non September 27, 2012.',
         'score': -2.0390625,
         'source': 'dense',
         'uid': 'd7a8041f-91a4-4220-8ac1-5c644f7d611f'},

        {'doc_id': '<urn:uuid:b6541cf6-d442-454a-9e6f-45eaaa237424>',
         'text': "Our left brain thinks in terms of words and symbols while our right brain thinks in terms of images. Remembering a persons name is a function of the left-brain memory while rembering a persons's face is a function.",
         'score': -4.41796875,
         'source': 'dense',
         'uid': '5cca9cdf-1644-44f5-a154-4e3269c2e591'},

        {'doc_id': '<urn:uuid:e4bf2415-2032-4a8a-9c18-715cf2d5f91f>',
         'text': 'The legal term for this compensation is "damages. " Exactly what damages you can recover varies from state to state, but you can usually recover:\n- Past and future medical expenses\n- Future lost wages (if the injury limits your ability to work in the future)\n- Property damages\n- Pain and suffering\n- Emotional distress\nReady to contact a lawyer about a possible second impact syndrome case? Use our free online directory to schedule your initial consultation today. - Guide to traumatic brain injuries\n- Resources to help after a brain injury\n- How to recognize a brain injury and what you should do about a brain injury\n- Concussions and auto accidents\n- Rehabilitation and therapy after a brain injury\n- Second impact syndrome and sports injury lawsuits\n- Legal guide to brain death\n- What is CTE?\n- A loss of oxygen can lead to an anoxic brain injury\n- Can you recover costs for the accident that caused a brain bleed?\n- What is the Traumatic Brain Injury Act?\n- Understanding the Hidden Challenges of Mild Traumatic Brain Injury\n- What is the Glasgow Coma Scale?.',
         'score': -17.671875,
         'source': 'sparse',
         'uid': '503c8248-a468-4f79-81f6-faed77356cca'}
    ]

    # Score documents
    scored_documents = reranker.score_documents(query, documents)

    print(scored_documents)

    # Print results
    # print("Documents with MixedBread Scores:")
    # for i, doc in enumerate(scored_documents):
    #     print(f"{i + 1}. UID: {doc['uid']}")
    #     print(f"   Original score: {doc['score']}")
    #     print(f"   MixedBread score: {doc['mxbai_score']:.4f}")
    #     print(f"   Text: {doc['text'][:100]}...\n")
    #
    # # Sort by MixedBread score
    # ranked_documents = sorted(scored_documents, key=lambda x: x['mxbai_score'], reverse=True)
    #
    # print("\nRanked Documents by MixedBread Score:")
    # for i, doc in enumerate(ranked_documents):
    #     print(f"{i + 1}. UID: {doc['uid']} - Score: {doc['mxbai_score']:.4f}")