from elasticsearch import Elasticsearch
from config import INDEX_NAME
import pandas as pd


def main():
    # Connect to localhost:9200 by default.
    es = Elasticsearch()
    es.indices.delete(index=INDEX_NAME, ignore=404)
    es.indices.create(
        index=INDEX_NAME,
        body={
            'mappings': {
                'properties': {  # Just a magic word.
                    'title': {  # The field we want to configure.
                        'type': 'text',  # The kind of data we’re working with.
                        'fields': {  # create an analyzed field.
                            'chinese_analyzed': {  # Name that field `name.english_analyzed`.
                                'type': 'text',  # It’s also text.
                                'analyzer': 'ik_max_word',  # And here’s the analyzer we want to use.
                                'search_analyzer': 'ik_smart'
                            }
                        }
                    }
                }
            },
            'settings': {},
        },
    )
    df = pd.read_csv('./index.csv', index_col=False, names=['title', 'url'])
    for idx, line in df.iterrows():
        try:
            es.create(
                index=INDEX_NAME,
                id=line['url'],
                body={
                    'title': line['title'],
                    'url': line['url']
                }
            )
        except:
            try:
                es.update(
                    index=INDEX_NAME,
                    id=line['url'],
                    body={
                        'title': line['title'],
                        'url': line['url']
                    }
                )
            except:
                pass
            pass


if __name__ == '__main__':
    main()
