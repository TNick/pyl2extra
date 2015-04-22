"""
Unit tests for ./yaml_parse.py

The script is also expected to pass all tests in
`pyl2extra/pyl2extra/config/tests`.

TODO: add tests for new functionality.
"""

from pyl2extra.config.yaml_parse import load, load_path, initialize

def test_original():
    """
    Runs original test suite from `pyl2extra/pyl2extra/config/tests`.
    """
    from pylearn2.config.tests import test_yaml_parse

    test_yaml_parse.load = load
    test_yaml_parse.load_path = load_path
    test_yaml_parse.initialize = initialize

    for k in dir(test_yaml_parse):
        if k.startswith('test_'):
            getattr(test_yaml_parse, k)()


if __name__ == "__main__":
    # runs all tests
    funcs = [k for k in locals().keys() if k.startswith('test_')]
    for k in funcs:
        locals()[k]()
