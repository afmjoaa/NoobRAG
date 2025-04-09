from functools import cache
from typing import List, Dict
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection
import boto3

from retrievers.indices_config import IndicesConfig


class SparseRetriever:
    def __init__(
        self,
        index_name: str = "fineweb10bt-512-0w-e5-base-v2",
        profile: str = IndicesConfig.AWS_PROFILE_NAME,
        region: str = IndicesConfig.AWS_REGION_NAME,
    ):
        self.index_name = index_name
        self.profile = profile
        self.region = region
        self.client = self._get_client()

    @cache
    def _get_client(self) -> OpenSearch:
        credentials = boto3.Session(profile_name=self.profile).get_credentials()
        auth = AWSV4SignerAuth(credentials, region=self.region)
        host = IndicesConfig.get_ssm_value("/opensearch/endpoint", profile=self.profile, region=self.region)

        return OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )

    def query(self, query: str, top_k: int = 10) -> Dict:
        body = {
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": top_k,
        }
        return self.client.search(index=self.index_name, body=body)

    def batch_query(self, queries: List[str], top_k: int = 10) -> List[Dict]:
        request = []
        for query in queries:
            request.append({"index": self.index_name})
            request.append({
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text"],
                    }
                },
                "size": top_k,
            })
        return self.client.msearch(body=request)

    @staticmethod
    def show_results(results: Dict):
        hits = results.get("hits", {}).get("hits", [])
        for hit in hits:
            print("chunk:", hit["_id"], "score:", hit["_score"])
            print(hit["_source"].get("text", ""))
            print()

