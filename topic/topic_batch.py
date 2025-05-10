import time
from typing import List, Dict, Any
import torch
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN


class TextTopicAnalyzer:
    """
    A class for analyzing multiple text documents using BERTopic and returning
    topic-based paragraphs with associated metadata following clean architecture principles.
    Uses GPU acceleration when available.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize models
        embedding_model, vectorizer_model, umap_model, hdbscan_model = self._initialize_models(100)
        self.topic_model = self._create_topic_model(embedding_model, vectorizer_model, umap_model, hdbscan_model)
        if self.verbose:
            print(f"Using device: {self.device}")

    def _log(self, message):
        if self.verbose:
            print(message)

    def _initialize_models(self, sentence_count):
        embedding_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)
        vectorizer_model = CountVectorizer(stop_words="english", min_df=1)

        n_components = min(3, sentence_count - 1) if sentence_count > 1 else 1
        umap_model = UMAP(
            n_neighbors=min(15, sentence_count - 1) if sentence_count > 1 else 2,
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=min(2, sentence_count),
            min_samples=1,
            metric='euclidean',
            prediction_data=True
        )

        return embedding_model, vectorizer_model, umap_model, hdbscan_model

    def _create_topic_model(self, embedding_model, vectorizer_model, umap_model, hdbscan_model):
        return BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            language='multilingual',
            calculate_probabilities=True,
            verbose=self.verbose,
            nr_topics="auto",
            min_topic_size=1,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model
        )

    def _preprocess_text(self, text_chunk):
        return [sentence.strip() for sentence in text_chunk.split('.') if sentence.strip()]

    def resolve_topics(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple documents to create topic-based paragraphs with document associations.

        Args:
            documents: List of documents with 'text', 'doc_id', 'score', and 'source'

        Returns:
            List of topic dictionaries with aggregated passages and document associations
        """
        # Collect sentences and track their origins
        all_sentences = []
        doc_origins = []
        for doc in documents:
            sentences = self._preprocess_text(doc.get("text", ""))
            all_sentences.extend(sentences)
            doc_origins.extend([doc["doc_id"]] * len(sentences))

        # Handle insufficient data
        if len(all_sentences) < 3:
            return [{
                'passage': ' '.join(f"{s}." for s in all_sentences),
                'score': 0,
                'source': 'basic',
                'doc_IDs': list(set(doc_origins))
            }]

        try:


            # Perform topic modeling
            topics, _ = self.topic_model.fit_transform(all_sentences)

            # Organize results by topic
            topic_groups = {}
            for sentence, topic, doc_id in zip(all_sentences, topics, doc_origins):
                if topic not in topic_groups:
                    topic_groups[topic] = {
                        'sentences': [],
                        'doc_ids': set()
                    }
                topic_groups[topic]['sentences'].append(sentence)
                topic_groups[topic]['doc_ids'].add(doc_id)

            # Format output
            return [{
                'passage': ' '.join(f"{s}." for s in group['sentences']),
                'score': 0,
                'source': 'basic',
                'doc_IDs': list(group['doc_ids'])
            } for group in topic_groups.values()]

        except Exception as e:
            self._log(f"Topic modeling failed: {str(e)}")
            return [{
                'passage': ' '.join(f"{s}." for s in all_sentences),
                'score': 0,
                'source': 'basic',
                'doc_IDs': list(set(doc_origins))
            }]


# Usage Example
if __name__ == "__main__":
    start_time = time.time()
    analyzer = TextTopicAnalyzer(verbose=True)

    sample_documents = [
   {
      "doc_id":"<urn:uuid:0cf75b43-d690-4aa6-b3ca-f488ceb28ed9>",
      "text":"What will the age of Aquarius be like. What is second level consciousness and are we still there? Left and Right Brain technology what is it? part 1\nSorry we couldn't complete your registration. Please try again.\nYou must accept the Terms and conditions to register",
      "score":0.835362315,
      "source":"dense"
   },
   {
      "doc_id":"<urn:uuid:b6541cf6-d442-454a-9e6f-45eaaa237424>",
      "text":"Title: Left-Brain/Right-Brain Functions\nPreview: We have two eyes, two ears, two hands, and two minds. Our left brain thinks in terms of words and symbols while our right brain thinks in terms of images. The left side is the side used more by writiers, mathematicians, and scientists; the right side by artists, craftspeople, and musicians. Remembering a persons name is a function of the left-brain memory while rembering their face is a function .......\nBy aterry (adrienne)\non September 27, 2012.",
      "score":0.823291481,
      "source":"dense"
   },
   {
      "doc_id":"<urn:uuid:e4bf2415-2032-4a8a-9c18-715cf2d5f91f>",
      "text":"The legal term for this compensation is “damages.” Exactly what damages you can recover varies from state to state, but you can usually recover:\n- Past and future medical expenses\n- Future lost wages (if the injury limits your ability to work in the future)\n- Property damages\n- Pain and suffering\n- Emotional distress\nReady to contact a lawyer about a possible second impact syndrome case? Use our free online directory to schedule your initial consultation today.\n- Guide to traumatic brain injuries\n- Resources to help after a brain injury\n- How to recognize a brain injury and what you should do about it\n- Concussions and auto accidents\n- Rehabilitation and therapy after a brain injury\n- Second impact syndrome and sports injury lawsuits\n- Legal guide to brain death\n- What is CTE?\n- A loss of oxygen can lead to an anoxic brain injury\n- Can you recover costs for the accident that caused a brain bleed?\n- What is the Traumatic Brain Injury Act?\n- Understanding the Hidden Challenges of Mild Traumatic Brain Injury\n- What is the Glasgow Coma Scale?",
      "score":0.24633245645981225,
      "source":"sparse"
   }
]

    results = analyzer.resolve_topics(sample_documents)
    print(f"Results: \n {results}")

    print("\nFinal Topics:")
    for topic in results:
        print(f"Passage: {topic['passage'][:100]}...")
        print(f"Associated Doc IDs: {topic['doc_IDs']}\n")

    print(f"Processed in {time.time() - start_time:.2f} seconds")