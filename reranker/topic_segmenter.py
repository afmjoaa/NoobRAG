from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN


def bertopic_for_small_text(text_chunk):
    # Preprocess the text into sentences
    sentences = [sentence.strip() for sentence in text_chunk.split('.') if sentence.strip()]

    print(f"Processing {len(sentences)} sentences")

    # Initialize models with settings optimized for small text
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectorizer_model = CountVectorizer(stop_words="english", min_df=1)

    # Configure UMAP for small datasets - key fix is to set n_components to a small value
    # and ensure it's smaller than the number of sentences
    n_components = min(2, len(sentences) - 1)  # Ensure n_components is less than number of sentences

    umap_model = UMAP(
        n_neighbors=min(5, len(sentences) - 1),  # Adjust neighbors based on dataset size
        n_components=n_components,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    # Configure HDBSCAN for small datasets
    hdbscan_model = HDBSCAN(
        min_cluster_size=min(2, len(sentences)),  # Minimum size of 2 or the number of sentences if less
        min_samples=1,
        metric='euclidean',
        prediction_data=True
    )

    # Initialize BERTopic with our custom models
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        language='multilingual',
        calculate_probabilities=True,  # Enable probabilities
        verbose=True,
        nr_topics="auto",  # Let the algorithm decide
        min_topic_size=1,  # For very small datasets
        umap_model=umap_model,
        hdbscan_model=hdbscan_model
    )

    # Check if we have enough sentences to proceed
    if len(sentences) < 3:
        print("Warning: Text has too few sentences for meaningful topic modeling.")
        # Create a simple topic structure for very small texts
        topics = [0] * len(sentences)
        topic_sentences = {0: sentences}
        print("All sentences assigned to a single topic due to small sample size.")
        return topic_sentences

    # Fit the model
    try:
        topics, probs = topic_model.fit_transform(sentences)

        # Group sentences by their topics
        topic_sentences = {}
        for sentence, topic in zip(sentences, topics):
            if topic not in topic_sentences:
                topic_sentences[topic] = []
            topic_sentences[topic].append(sentence)

        # Print results
        print("Discovered Topics:\n")
        for topic_num in sorted(topic_sentences.keys()):
            if topic_num == -1:  # Outliers
                print("Uncategorized sentences:")
            else:
                print(f"Topic {topic_num}:")

            for sentence in topic_sentences[topic_num]:
                print(f" - {sentence}.")
            print()

        # Get topic info for better interpretation
        if -1 not in topics:  # Only if we have actual topics
            topic_info = topic_model.get_topic_info()
            print("\nTopic Information:")
            print(topic_info)

        return topic_sentences

    except Exception as e:
        print(f"Error in topic modeling: {str(e)}")
        print("Falling back to basic sentence grouping...")

        # Simple fallback: group all sentences into one topic
        return {0: sentences}


# Your text chunk
text_chunk = """
Environment in Focus: Green Revolution
This content is not assigned to a topic
TOPIC OF THE WEEK
ENVIRONMENT IN FOCUS:
Green Revolution is a public relations term, probably coined in the 1960s by William Gaud, then Director of the U.S. Agency for international Development, that symbolized the modernization of agriculture in less industrialized countries by means of technological change rather than violent “Red Revolution” (Communism).
The essence of Green Revolution was developing fertilizer-responsive varieties of wheat and rice that would increase national yields of wheat and rice.
With the increased national yields, promoters of the modernization of agriculture in less industrialized countries by means of technological change rather than violent “Red Revolution” (Communism) saw the potential for reducing hunger, poverty, misery, and the potential for violent social upheaval that would threaten the geopolitical interests of the United States and other western powers.
Scientific research that underlay Green Revolution is generally dated from the establishment of the Mexican Agricultural Program by the Rockefeller Foundation in 1943, and the successful technological changes were well in place by the 1970s in many parts of Asia and Latin America. Green revolution technology is now almost universally the “normal” way to do agriculture in all parts of the world except in parts of Africa.
For most of human evolutionary history, people were hunter-gatherers. people ate that which grew with little or no human management. Beginning about 40,000 years ago, and culminating about 10,000 years ago, however, people invented and increasingly used active management or “agriculture” to increase the quantity and security of food supplies, including grains, vegetables, and livestock. With this change, often called the “Neolithic Revolution,” modern humans became more sedentary and lived in villages or cities. A farming class tilled the soil and created the potential for large stores of grain that supported an increasingly thriving urban culture. For most of the past 10,000 years, until about 250 years ago, most (probably over 90 percent) people were tillers of the soil, and the primary human occupation was “farmer.”
With the exception of the last 100 years, the primary limitation on agricultural production was the supply of nitrogen to growing plants. The natural nitrogen cycle slowly takes the largely inert gas nitrogen (N2) from the atmosphere and fixes the largely inert gas nitrogen (N2) into soil in forms that can be utilized as nutrients by plants.
Human activities, including the production of nitrogen fertilizers, combustion of fossil fuels, and other activities, however, have substantially increased the amounts of nitrogen fixed per year to levels that rival rates of natural nitrogen fixation. Increased levels of nitrogen fixation have already had effects on the atmosphere, terrestrial ecosystems, and aquatic ecosystems. Most of effects on the atmosphere, terrestrial ecosystems, and aquatic ecosystems are troublesome, but the still increasing uses of nitrogen fertilizer clearly have made agricultural ecosystems more productive. Nitrogen fertilizer, in a word, was essential to the productivity gains of the green revolution. As human population continues to expand, expansion of the use of nitrogen fertilizer, too, will occur.
Before 1900, the only major interventions people could do to increase supplies of nitrogen for crop plants lay in using manure, crop rotations with leguminous crops like clover, and mining of salts like sodium nitrate and natural materials like bird guano. These extra supplies, however, were expensive and limited in size. People were surrounded by billions of tons of inert nitrogen in the air, but it was impossible to bring billions of tons of inert nitrogen in the air to service of increasing yields in agriculture.
RELATED NEWS IN FOCUS"
"""

# Run the topic modeling
bertopic_for_small_text(text_chunk)