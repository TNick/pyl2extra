"""
Unit tests for pyl2extra/scripts/train_multi.py

TODO: add tests.
"""

from pyl2extra.scripts import train_multi




if __name__ == "__main__":
    # runs all tests
    funcs = [k for k in locals().keys() if k.startswith('test_')]
    for k in funcs:
        locals()[k]()
