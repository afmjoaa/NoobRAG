import json
from collections import defaultdict
import warnings
import numpy as np

from topic.topic_segmenter import get_topic_text_with_info

warnings.filterwarnings("ignore")

# Import BERTopic
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN


def cluster_documents_with_berttopic(documents, min_topic_size=2, nr_topics=None):
    """
    Uses the BERTopic library for advanced topic modeling and document clustering.

    Args:
        documents: List of document dictionaries with 'doc_id' and 'text' fields
        min_topic_size: Minimum number of documents to form a topic
        nr_topics: Number of topics to extract (or "auto" to determine automatically)

    Returns:
        List of clustered documents with combined text and possibly combined doc_ids
    """
    if not documents:
        return []

    # Extract text content
    texts = [doc['text'] for doc in documents]

    # For small datasets, we need to adjust UMAP and HDBSCAN parameters
    sentence_count = len(texts)

    print(f"sentence_count {sentence_count}")
    # Configure UMAP for small datasets

    n_components = min(3, sentence_count - 1)
    umap_model = UMAP(
        n_neighbors=min(6, sentence_count - 1),
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

    # Create BERTopic model with custom UMAP and HDBSCAN
    topic_model = BERTopic(
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        # umap_model=umap_model,
        # hdbscan_model=hdbscan_model
    )

    # Fit the model and transform documents
    topics, probs = topic_model.fit_transform(texts)

    # Group documents by topic
    topic_groups = defaultdict(list)
    for i, topic in enumerate(topics):
        topic_groups[topic].append(i)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    topic_names = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}

    # Create result clusters
    result_clusters = []

    for topic, indices in topic_groups.items():
        # Get documents for this topic
        doc_ids = {documents[idx]['doc_id'] for idx in indices}
        combined_text = "\n".join([documents[idx]['text'] for idx in indices])

        # Use the first document's score and source
        score = documents[indices[0]]['score']
        source = documents[indices[0]]['source']

        # Create the cluster
        cluster = {'passage': combined_text, 'score': score, 'source': source, 'doc_IDs': list(doc_ids)}
        result_clusters.append(cluster)

    return result_clusters


if __name__ == "__main__":
    # example_input = get_topic_text_with_info()
    example_input = example_input = [
    # Climate change documents
    {
        'doc_id': '<urn:uuid:1234abcd-5678-efgh-9101-ijklmnopqrst>',
        'text': 'Global warming is causing rising sea levels. Scientists predict that coastal cities may face severe flooding in the coming decades.',
        'score': 0.92,
        'source': 'dense'
    },
    {
        'doc_id': '<urn:uuid:2345bcde-6789-fghi-0112-jklmnopqrstu>',
        'text': 'Climate change impacts include extreme weather events, rising temperatures, and melting ice caps. These effects are becoming more pronounced each year.',
        'score': 0.89,
        'source': 'dense'
    },
    {
        'doc_id': '<urn:uuid:6789fghi-0123-jklm-5156-nopqrstuvwxy>',
        'text': 'Reducing carbon emissions is crucial to mitigating climate change. Many countries have pledged to become carbon neutral by 2050.',
        'score': 0.88,
        'source': 'dense'
    },

    # AI technology documents
    {
        'doc_id': '<urn:uuid:3456cdef-7890-ghij-2123-klmnopqrstuv>',
        'text': 'Machine learning algorithms are revolutionizing how we approach data analysis. Neural networks can identify patterns humans might miss.',
        'score': 0.78,
        'source': 'dense'
    },
    {
        'doc_id': '<urn:uuid:4567defg-8901-hijk-3134-lmnopqrstuvw>',
        'text': 'Artificial intelligence is making significant advances in natural language processing. New transformer models can understand context better than previous approaches.',
        'score': 0.81,
        'source': 'dense'
    },

    # Space exploration
    {
        'doc_id': '<urn:uuid:5678efgh-9012-ijkl-4145-mnopqrstuvwx>',
        'text': 'NASA plans to return humans to the Moon by 2025 with the Artemis program. This will establish a sustainable lunar presence before Mars missions.',
        'score': 0.85,
        'source': 'dense'
    },

    # Healthcare
    {
        'doc_id': '<urn:uuid:7890ghij-1234-klmn-6167-opqrstuvwxyz>',
        'text': 'Telemedicine has transformed healthcare delivery, enabling remote consultations and monitoring for patients with chronic conditions.',
        'score': 0.83,
        'source': 'dense'
    },

    # Finance
    {
        'doc_id': '<urn:uuid:8901hijk-2345-lmno-7178-pqrstuvwxyza>',
        'text': 'Cryptocurrency markets remain volatile, but institutional adoption of Bitcoin and Ethereum suggests a maturing financial instrument.',
        'score': 0.80,
        'source': 'dense'
    },

    # Education
    {
        'doc_id': '<urn:uuid:9012ijkl-3456-mnop-8189-qrstuvwxyzab>',
        'text': 'Online learning platforms have democratized education, making high-quality courses available to learners around the world.',
        'score': 0.84,
        'source': 'dense'
    },

    # Sports
    {
        'doc_id': '<urn:uuid:0123jklm-4567-nopq-9190-rstuvwxyzabc>',
        'text': 'The Olympics showcase elite athletic performance and international unity, drawing attention to emerging sports and youth talent.',
        'score': 0.77,
        'source': 'dense'
    },

    # Cybersecurity
    {
        'doc_id': '<urn:uuid:1234klmn-5678-opqr-0201-stuvwxyzabcd>',
        'text': 'Ransomware attacks are increasing, targeting critical infrastructure and demanding payment in cryptocurrency to avoid detection.',
        'score': 0.86,
        'source': 'dense'
    },

    # Philosophy
    {
        'doc_id': '<urn:uuid:2345lmno-6789-pqrs-1212-tuvwxyzabcde>',
        'text': 'Existentialism explores individual freedom, choice, and the search for meaning in a world perceived as indifferent or absurd.',
        'score': 0.79,
        'source': 'dense'
    }
]
    # Process the example input using BERTopic
    print("Using BERTopic for document clustering...")
    result = cluster_documents_with_berttopic(example_input)

    print(result)
    print(len(result))


