from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from typing import List, Dict, Any

from coref.coref_resolver import CorefResolver
from main import getCombinedContext


class TextTopicAnalyzer:
    """
    A class for analyzing small text chunks using BERTopic and returning
    topic-based paragraphs following clean architecture principles.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def _log(self, message):
        if self.verbose:
            print(message)

    def _initialize_models(self, sentence_count):
        """
        Initialize the required models with settings optimized for small text.

        Args:
            sentence_count (int): Number of sentences in the input text

        Returns:
            tuple: Initialized models (embedding_model, vectorizer_model, umap_model, hdbscan_model)
        """
        # Initialize embedding model
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize vectorizer
        vectorizer_model = CountVectorizer(stop_words="english", min_df=1)

        # Configure UMAP for small datasets
        n_components = min(2, sentence_count - 1)
        umap_model = UMAP(
            n_neighbors=min(5, sentence_count - 1),
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )

        # Configure HDBSCAN for small datasets
        hdbscan_model = HDBSCAN(
            min_cluster_size=min(2, sentence_count),
            min_samples=1,
            metric='euclidean',
            prediction_data=True
        )

        return embedding_model, vectorizer_model, umap_model, hdbscan_model

    def _create_topic_model(self, embedding_model, vectorizer_model, umap_model, hdbscan_model):
        """
        Create the BERTopic model with the configured components.

        Args:
            embedding_model: The sentence transformer model
            vectorizer_model: The count vectorizer model
            umap_model: The UMAP dimensionality reduction model
            hdbscan_model: The HDBSCAN clustering model

        Returns:
            BERTopic: Configured BERTopic model
        """
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
        """
        Preprocess the input text into sentences.

        Args:
            text_chunk (str): Input text to be processed

        Returns:
            list: List of sentence strings
        """
        sentences = [sentence.strip() for sentence in text_chunk.split('.') if sentence.strip()]
        self._log(f"Processing {len(sentences)} sentences")
        return sentences

    def _group_sentences_by_topic(self, sentences, topics):
        """
        Group sentences by their assigned topics.

        Args:
            sentences (list): List of sentences
            topics (list): List of topic IDs corresponding to each sentence

        Returns:
            dict: Dictionary mapping topic IDs to lists of sentences
        """
        topic_sentences = {}
        for sentence, topic in zip(sentences, topics):
            if topic not in topic_sentences:
                topic_sentences[topic] = []
            topic_sentences[topic].append(sentence)
        return topic_sentences

    def _format_topic_paragraphs(self, topic_sentences):
        """
        Format topic sentences into paragraphs, one paragraph per topic.

        Args:
            topic_sentences (dict): Dictionary mapping topic IDs to lists of sentences

        Returns:
            dict: Dictionary mapping topic IDs to paragraph strings
        """
        topic_paragraphs = {}
        for topic_num, sentences in topic_sentences.items():
            paragraph = " ".join([sentence + "." for sentence in sentences])
            topic_paragraphs[topic_num] = paragraph
        return topic_paragraphs

    def _print_topic_results(self, topic_sentences, topic_model, topics):
        """
        Print detailed results about topics and their sentences.

        Args:
            topic_sentences (dict): Dictionary mapping topic IDs to lists of sentences
            topic_model (BERTopic): The trained topic model
            topics (list): List of topic IDs
        """
        # Print discovered topics
        self._log("\nDiscovered Topics:\n")
        for topic_num in sorted(topic_sentences.keys()):
            if topic_num == -1:
                self._log("Uncategorized sentences:")
            else:
                self._log(f"Topic {topic_num}:")

            for sentence in topic_sentences[topic_num]:
                self._log(f" - {sentence}.")
            self._log("")

        # Get and print topic info if there are actual topics
        if -1 not in topics:
            try:
                topic_info = topic_model.get_topic_info()
                self._log("\nTopic Information:")
                self._log(topic_info)
            except Exception as e:
                self._log(f"Could not get topic info: {str(e)}")

    def analyze(self, text_chunk):
        """
        Analyze the input text chunk and extract topics as paragraphs.

        Args:
            text_chunk (str): Text to analyze

        Returns:
            dict: Dictionary mapping topic IDs to paragraph strings
        """
        # Preprocess text
        sentences = self._preprocess_text(text_chunk)

        # Check if we have enough sentences to proceed
        if len(sentences) < 3:
            self._log("Warning: Text has too few sentences for meaningful topic modeling.")
            self._log("All sentences assigned to a single topic due to small sample size.")
            return {0: " ".join([sentence + "." for sentence in sentences])}

        try:
            # Initialize models
            embedding_model, vectorizer_model, umap_model, hdbscan_model = self._initialize_models(len(sentences))

            # Create and fit topic model
            topic_model = self._create_topic_model(embedding_model, vectorizer_model, umap_model, hdbscan_model)
            topics, probs = topic_model.fit_transform(sentences)

            # Group sentences by topics
            topic_sentences = self._group_sentences_by_topic(sentences, topics)

            # Print detailed results if verbose
            if self.verbose:
                self._print_topic_results(topic_sentences, topic_model, topics)

            # Format topics as paragraphs
            topic_paragraphs = self._format_topic_paragraphs(topic_sentences)

            return topic_paragraphs

        except Exception as e:
            # self._log(f"Error in topic modeling: {str(e)}")
            self._log("Falling back to basic sentence grouping...")

            # Simple fallback: group all sentences into one topic
            combined_text = " ".join([sentence + "." for sentence in sentences])
            return {0: combined_text}


def get_topic_text_with_info():
    # Sample text chunk
    sample_documents = getCombinedContext(max_docs=3)
    # Coref resolution
    resolver = CorefResolver()
    resolved_docs = resolver.resolve_documents(sample_documents)
    # Initialize the result list
    expanded_results = []
    for document in resolved_docs:
        text_chunk = document.get("text", "")
        # Create analyzer with verbose output
        analyzer = TextTopicAnalyzer(verbose=False)

        # Process the text and get topic paragraphs
        topic_paragraphs = analyzer.analyze(text_chunk)

        # print(topic_paragraphs)
        # print(len(topic_paragraphs))

        # Creating new list
        doc_id = document.get('doc_id', '')
        score = document.get('score', 0.0)
        source = document.get('source', '')

        for _, value in topic_paragraphs.items():
            new_entry = {
                'doc_id': doc_id,
                'text': value,
                'score': score,
                'source': source,
            }
            expanded_results.append(new_entry)
    return expanded_results


# Example usage
if __name__ == "__main__":
    print(get_topic_text_with_info())



# text_chunk = """
#     Environment in Focus: Green Revolution
#     This content is not assigned to a topic
#     TOPIC OF THE WEEK
#     ENVIRONMENT IN FOCUS:
#     Green Revolution is a public relations term, probably coined in the 1960s by William Gaud, then Director of the U.S. Agency for international Development, that symbolized the modernization of agriculture in less industrialized countries by means of technological change rather than violent "Red Revolution" (Communism).
#     The essence of Green Revolution was developing fertilizer-responsive varieties of wheat and rice that would increase national yields of wheat and rice.
#     With the increased national yields, promoters of the modernization of agriculture in less industrialized countries by means of technological change rather than violent "Red Revolution" (Communism) saw the potential for reducing hunger, poverty, misery, and the potential for violent social upheaval that would threaten the geopolitical interests of the United States and other western powers.
#     Scientific research that underlay Green Revolution is generally dated from the establishment of the Mexican Agricultural Program by the Rockefeller Foundation in 1943, and the successful technological changes were well in place by the 1970s in many parts of Asia and Latin America. Green revolution technology is now almost universally the "normal" way to do agriculture in all parts of the world except in parts of Africa.
#     For most of human evolutionary history, people were hunter-gatherers. people ate that which grew with little or no human management. Beginning about 40,000 years ago, and culminating about 10,000 years ago, however, people invented and increasingly used active management or "agriculture" to increase the quantity and security of food supplies, including grains, vegetables, and livestock. With this change, often called the "Neolithic Revolution," modern humans became more sedentary and lived in villages or cities. A farming class tilled the soil and created the potential for large stores of grain that supported an increasingly thriving urban culture. For most of the past 10,000 years, until about 250 years ago, most (probably over 90 percent) people were tillers of the soil, and the primary human occupation was "farmer."
#     With the exception of the last 100 years, the primary limitation on agricultural production was the supply of nitrogen to growing plants. The natural nitrogen cycle slowly takes the largely inert gas nitrogen (N2) from the atmosphere and fixes the largely inert gas nitrogen (N2) into soil in forms that can be utilized as nutrients by plants.
#     Human activities, including the production of nitrogen fertilizers, combustion of fossil fuels, and other activities, however, have substantially increased the amounts of nitrogen fixed per year to levels that rival rates of natural nitrogen fixation. Increased levels of nitrogen fixation have already had effects on the atmosphere, terrestrial ecosystems, and aquatic ecosystems. Most of effects on the atmosphere, terrestrial ecosystems, and aquatic ecosystems are troublesome, but the still increasing uses of nitrogen fertilizer clearly have made agricultural ecosystems more productive. Nitrogen fertilizer, in a word, was essential to the productivity gains of the green revolution. As human population continues to expand, expansion of the use of nitrogen fertilizer, too, will occur.
#     Before 1900, the only major interventions people could do to increase supplies of nitrogen for crop plants lay in using manure, crop rotations with leguminous crops like clover, and mining of salts like sodium nitrate and natural materials like bird guano. These extra supplies, however, were expensive and limited in size. People were surrounded by billions of tons of inert nitrogen in the air, but it was impossible to bring billions of tons of inert nitrogen in the air to service of increasing yields in agriculture.
#     RELATED NEWS IN FOCUS
#     """