import pinecone
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, index_name, model_name="intfloat/e5-base", pinecone_env="us-west1-gcp"):
        pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment=pinecone_env)
        self.index = pinecone.Index(index_name)
        self.encoder = SentenceTransformer(model_name)

    def retrieve(self, query, top_k=5):
        embedding = self.encoder.encode(query).tolist()
        result = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return [match['metadata']['text'] for match in result['matches']]
