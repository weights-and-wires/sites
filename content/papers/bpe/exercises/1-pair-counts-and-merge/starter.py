import collections
import re


def get_stats(vocab):
    """Count adjacent symbol pairs, weighted by word frequency.

    vocab maps "s y m b o l s </w>" to its frequency.
    Return a dict that maps (left, right) to a count.
    """
    raise NotImplementedError


def merge_vocab(pair, vocab):
    """Replace each whole-symbol occurrence of pair with the joined symbol.

    Return a new dict; keep each word's frequency.
    """
    raise NotImplementedError
