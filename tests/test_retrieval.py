import math

import pytest

from rag_eval import hit_at_k, mrr, ndcg_at_k


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], {"b"}, k=2) == 1.0
    assert hit_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0
    assert hit_at_k([], {"x"}, k=3) == 0.0


def test_mrr():
    assert mrr(["a", "b", "c"], {"b"}) == 0.5
    assert mrr(["x", "y"], {"z"}) == 0.0
    assert mrr(["z"], {"z"}) == 1.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(1.0)


def test_ndcg_late_hit_discounted():
    late = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
    assert late == pytest.approx((1 / math.log2(4)) / 1.0)
    assert late < ndcg_at_k(["a", "y", "x"], {"a"}, k=3)


def test_empty_relevant_raises():
    with pytest.raises(ValueError):
        mrr(["a"], set())
    with pytest.raises(ValueError):
        hit_at_k(["a"], [], k=1)


def test_bad_k_raises():
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a"}, k=0)

