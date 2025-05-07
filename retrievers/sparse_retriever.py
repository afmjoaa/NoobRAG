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
            timeout=60,
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

    def batch_query(self, queries: list[str], top_k: int = 10, n_parallel: int = 10) -> Dict:
        """Sends a list of queries to OpenSearch and returns the results. Configuration of Connection Timeout might be needed for serving large batches of queries"""
        request = []
        for query in queries:
            req_head = {"index": self.index_name}
            req_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text"],
                    }
                },
                "size": top_k,
            }
            request.extend([req_head, req_body])

        return self.client.msearch(body=request)

    @staticmethod
    def show_results(results: Dict):
        hits = results.get("hits", {}).get("hits", [])
        for hit in hits:
            print("chunk:", hit["_source"].get("doc_id", ""), "score:", hit["_score"])
            print(hit["_source"].get("text", ""))
            print()


# EXAMPLE OUTPUT
# {
#    "_index":"sigir-512-0w-e5-base-v2",
#    "_id":"doc-<urn:uuid:3275960d-6467-4e26-b9e5-994945c7f377>::chunk-0",
#    "_score":14.364122,
#    "_source":{
#       "text":"Earlier today I spoke with some one at a brain injury association. During our conversation I was told that there was nothing special about Second Chance to Live. I was told that every one has a story and that they have heard the same thing that I share about on Second Chance to Live many, many times before. I was also told that Second Chance to Live presents too much information.\nI listened to what the person had to say and thanked them for their input. I then asked the individual to read about the impact that Second Chance to Live is having upon lives through my testimonials and endorsements. As to there being too much information I went on to share that Second Chance to Live addresses various topics and presents information to encourage, motivate, empower and provide hope to brain injury survivors, their families, and professionals as a traumatic brain injury survivor and as a professional.\nI went on to share with this individual that audience members — who have heard me speak — have told me that the information that I present can benefit anyone — regardless of whether they have experienced a brain injury or some other type of adversity. As we continued our conversation I shared with her that I had written a total of 1715 articles which can be used as a resource and as a tool to generate discussion in support groups — while honoring my copyright and using my Resource Box.\nAs we continued to talk I again got the impression from the director of this state brain injury association — with whom I was speaking — that there was nothing special about Second Chance to Live. After the phone call ended I processed some of my feelings. Initially I felt minimized, marginalized and some what discounted by what she shared during our conversation. As I examined why I felt those feelings I realized that her opinion was merely her opinion, not reality.\nAfter I owned and examined my feelings I remembered that I have experienced similar responses from many of the brain injury associations that I have contacted through out the United States. I am not quite sure why the message of encouragement, motivation, empowerment and hope that is presented through Second Chance to Live is not being utilized by the Brain Injury Association of America or the state brain injury associations. If any one can give me feedback as to the experience I am having with the Brain Injury Associations, please let me know. Thank you.\nCraig J. Phillips MRC, BA\nSecond Chance to Live\nOur circumstances are not meant to keep us down, but to build us up!",
#       "doc_id":"<urn:uuid:3275960d-6467-4e26-b9e5-994945c7f377>",
#       "chunk_order":0,
#       "total_doc_chunks":2,
#       "is_last_chunk":false,
#       "is_first_chunk":true
#    }
# }