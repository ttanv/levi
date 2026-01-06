"""Trie-based prefix hit evaluation for LLM SQL problem."""

import pandas as pd
from typing import Tuple


class TrieNode:
    __slots__ = ['children', 'end_of_word']

    def __init__(self):
        self.children = {}
        self.end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.end_of_word = True

    def longest_common_prefix(self, word: str) -> int:
        node = self.root
        common_prefix_length = 0
        for char in word:
            if char in node.children:
                common_prefix_length += 1
                node = node.children[char]
            else:
                break
        return common_prefix_length


def evaluate_df_prefix_hit_cnt(df: pd.DataFrame) -> Tuple[int, float]:
    """
    Evaluate the prefix hit count of a DataFrame.

    Returns:
        (total_prefix_hit_count, hit_rate_percentage)
    """
    trie = Trie()
    total_prefix_hit_count = 0
    total_string_length = 0

    for _, row in df.iterrows():
        row_string = "".join(row.fillna("").astype(str).values)
        total_string_length += len(row_string)
        row_prefix_hit_count = min(len(row_string), trie.longest_common_prefix(row_string))
        trie.insert(row_string)
        total_prefix_hit_count += row_prefix_hit_count

    hit_rate = (total_prefix_hit_count / total_string_length * 100) if total_string_length > 0 else 0.0
    return total_prefix_hit_count, hit_rate
