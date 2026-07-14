"""Classic retrieval-quality metrics over ranked result lists.

All functions take the retriever's ranked output (\`retrieved\`, best first)
and the set of known-relevant ids (\`relevant\`). Graded relevance is not
required: relevance is binary, which matches how RAG eval sets are usually
labeled in practice.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Set


def _as_set(relevant: Iterable[str]) -> Set[str]:
    s = set(relevant)
    if not s:
        raise ValueError("relevant set is empty — a case with no relevant docs is unmeasurable")
    return s


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if any relevant doc appears in the top-k, else 0.0."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    rel = _as_set(relevant)
    return 1.0 if any(doc in rel for doc in retrieved[:k]) else 0.0


def mrr(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant doc; 0.0 if none retrieved."""
    rel = _as_set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k.

    DCG uses the standard log2 discount. The ideal DCG assumes all relevant
    docs (up to k) ranked first, so the score is comparable across cases
    with different numbers of relevant docs.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    rel = _as_set(relevant)

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(retrieved[:k], start=1)
        if doc in rel
    )
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

