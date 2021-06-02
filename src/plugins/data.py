import json
import os
import textwrap
_all_products = None


class ProductData():
    """
    Our product records. In this case they come from a json file, but you could
    just as easily load them from a database, or anywhere else.
    """

    def __init__(self, id_, url, title):
        self.id = id_
        self.url = url
        self.title = title

    def __str__(self):
        return textwrap.dedent("""\
            Id: {}
            url: {}
            title: {}
        """).format(self.id, self.url, self.title)


def all_products():
    """
    Returns a list of ~20,000 ProductData objects, loaded from
    searchapp/products.json
    """

    global _all_products

    if _all_products is None:
        _all_products = []

        # Load the product json from the same directory as this file.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        products_path = os.path.join(dir_path, 'news.json')
        with open(products_path, encoding='utf-8') as product_file:
            _all_products = [ProductData(idx + 1, **product) for idx, product in enumerate(json.load(product_file))]

    return _all_products
