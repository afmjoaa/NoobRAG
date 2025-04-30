from typing import List, Literal, Optional, Dict, Any
from multiprocessing.pool import ThreadPool
from functools import cache

import torch
from transformers import AutoModel, AutoTokenizer
from pinecone import Pinecone

from retrievers.indices_config import IndicesConfig


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        index_name: str = "fineweb10bt-512-0w-e5-base-v2",
        namespace: str = "default",
        query_prefix: str = "query: ",
        pooling: Literal["cls", "avg"] = "avg",
        normalize: bool = True,
        pinecone_token_ssm_path: str = "/pinecone/ro_token",
        aws_profile: str = IndicesConfig.AWS_PROFILE_NAME,
        aws_region: str = IndicesConfig.AWS_REGION_NAME,
    ):
        self.model_name = model_name
        self.index_name = index_name
        self.namespace = namespace
        self.query_prefix = query_prefix
        self.pooling = pooling
        self.normalize = normalize
        self.pinecone_token_ssm_path = pinecone_token_ssm_path
        self.aws_profile = aws_profile
        self.aws_region = aws_region

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.index = self._load_pinecone_index()

    @staticmethod
    @cache
    def _has_mps():
        return torch.backends.mps.is_available()

    @staticmethod
    @cache
    def _has_cuda():
        return torch.cuda.is_available()

    @cache
    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    @cache
    def _load_model(self):
        model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        device = "mps" if self._has_mps() else "cuda" if self._has_cuda() else "cpu"
        return model.to(device)

    @cache
    def _load_pinecone_index(self):
        api_key = IndicesConfig.get_ssm_secret(
            self.pinecone_token_ssm_path,
            profile=self.aws_profile,
            region=self.aws_region,
        )
        pc = Pinecone(api_key=api_key)
        return pc.Index(name=self.index_name)

    def _average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        inputs = [f"{self.query_prefix} {text}" for text in texts]
        encoded = self.tokenizer(inputs, padding=True, return_tensors="pt", truncation="longest_first")
        encoded = encoded.to(self.model.device)

        with torch.no_grad():
            output = self.model(**encoded)
            if self.pooling == "cls":
                embeddings = output.last_hidden_state[:, 0]
            else:
                embeddings = self._average_pool(output.last_hidden_state, encoded["attention_mask"])

            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()

    def query(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        vector = self._embed([query])[0]
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            namespace=self.namespace,
        )

    def batch_query(self, queries: List[str], top_k: int = 10, n_parallel: int = 10) -> List[Dict[str, Any]]:
        vectors = self._embed(queries)
        pool = ThreadPool(n_parallel)
        return pool.map(
            lambda vec: self.index.query(
                vector=vec,
                top_k=top_k,
                include_values=False,
                include_metadata=True,
                namespace=self.namespace,
            ),
            vectors,
        )

    @staticmethod
    def show_results(results: Dict[str, Any]):
        for match in results.get("matches", []):
            print("chunk:", match["metadata"].get("doc_id", ""), "score:", match["score"])
            print(match["metadata"].get("text", ""))
            print()


# EXAMPLE OUTPUT
# {'id': 'doc-<urn:uuid:0cf75b43-d690-4aa6-b3ca-f488ceb28ed9>::chunk-0',
#  'metadata': {'chunk_order': 0.0,
#               'doc_id': '<urn:uuid:0cf75b43-d690-4aa6-b3ca-f488ceb28ed9>',
#               'is_first_chunk': True,
#               'is_last_chunk': True,
#               'text': 'What will the age of Aquarius be like. What is second '
#                       'level consciousness and are we still there? Left and '
#                       'Right Brain technology what is it? part 1\n'
#                       "Sorry we couldn't complete your registration. Please "
#                       'try again.\n'
#                       'You must accept the Terms and conditions to register',
#               'total_doc_chunks': 1.0},
#  'score': 0.835362315,
#  'values': []}