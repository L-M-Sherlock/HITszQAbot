from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from typing import List

from config import INDEX_NAME

HEADERS = {
    'content-type': 'application/json'
}


class SearchResult():
    """Represents a product returned from elasticsearch."""

    def __init__(self, id_, url, title):
        self.id = id_
        self.url = url
        self.title = title

    def from_doc(doc) -> 'SearchResult':
        return SearchResult(
            id_=doc.meta.id,
            url=doc.url,
            title=doc.title,
        )


def search(term: str, count: int) -> List[SearchResult]:
    client = Elasticsearch()

    # Elasticsearch 6 requires the content-type header to be set, and this is
    # not included by default in the current version of elasticsearch-py
    client.transport.connection_pool.connection.headers.update(HEADERS)

    s = Search(using=client, index=INDEX_NAME)

    name_query = {
        "dis_max": {
            "queries": [
                {
                    "match": {
                        "title": {
                            "query": term,
                            "operator": "and",
                            "fuzziness": "AUTO"
                        }
                    }
                },
                {
                    "match": {
                        "title.chinese_analyzed": {
                            "query": term,
                            "operator": "and"
                        }
                    }
                }
            ],
            "tie_breaker": 0.7
        }
    }

    docs = s.query(name_query)[:count].execute()

    return [SearchResult.from_doc(d) for d in docs]

if __name__ == "__main__":
    result = search("间隔重复", 10)
    print(result)
