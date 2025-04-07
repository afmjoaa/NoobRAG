from opensearchpy import OpenSearch


class BM25Retriever:
    def __init__(self, host='localhost', port=9200, index_name='fineweb-sparse'):
        self.client = OpenSearch([{'host': host, 'port': port}], http_compress=True)
        self.index = index_name

    def retrieve(self, query, top_k=5):
        response = self.client.search(
            index=self.index,
            body={
                "query": {
                    "match": {
                        "content": {
                            "query": query
                        }
                    }
                },
                "size": top_k
            }
        )
        hits = response['hits']['hits']
        return [hit['_source']['content'] for hit in hits]
