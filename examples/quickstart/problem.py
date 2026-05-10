"""Toy bin-packing problem shared by the code-evolution quickstarts.

Small, deterministic, no external dependencies. The seed implementation is a
naive first-fit packer; the evolutionary loop has plenty of room to find a
better strategy (first-fit-decreasing, best-fit, small local search, etc.).
"""

from collections import Counter

ITEMS_LIST = [
    [4, 8, 1, 4, 2, 1, 3, 5, 7, 6],
    [9, 2, 3, 7, 8, 1, 4, 6, 5, 2, 3, 4],
    [5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 6, 6, 7, 7, 8, 8],
]
CAPACITY = 10

PROBLEM_DESCRIPTION = """\
# Bin Packing

Given a list of integer item sizes and an integer bin capacity, partition
the items into bins such that the contents of each bin sum to at most
`capacity`. Minimize the number of bins used (equivalently, minimize the
total wasted space).

Constraints:
- Every item must appear in exactly one bin.
- No bin's contents may sum to more than `capacity`.
- Return a list of lists of ints.

Optimal is not required; better packings score higher.
"""

FUNCTION_SIGNATURE = "def pack(items: list[int], capacity: int) -> list[list[int]]:"

SEED_PROGRAM = '''\
def pack(items, capacity):
    """Naive first-fit: place each item in the first bin that has room."""
    bins = []
    for item in items:
        for b in bins:
            if sum(b) + item <= capacity:
                b.append(item)
                break
        else:
            bins.append([item])
    return bins
'''


def score_fn(pack, inputs):
    """Average utilization across the input instances (max 100).

    A packing is invalid (score 0) if it overfills a bin or drops/duplicates
    items. Otherwise, score = 100 * total_item_size / (n_bins * capacity).
    """
    scores = []
    for items in inputs:
        try:
            bins = pack(list(items), CAPACITY)
        except Exception:
            return {"score": 0.0}
        if not isinstance(bins, list) or not all(isinstance(b, list) for b in bins):
            return {"score": 0.0}
        if any(sum(b) > CAPACITY for b in bins):
            return {"score": 0.0}
        if Counter(x for b in bins for x in b) != Counter(items):
            return {"score": 0.0}
        used = sum(sum(b) for b in bins)
        scores.append(100.0 * used / (len(bins) * CAPACITY))
    return {"score": sum(scores) / len(scores)}
