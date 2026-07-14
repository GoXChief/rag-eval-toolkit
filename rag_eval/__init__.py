"""rag-eval-toolkit: retrieval metrics + LLM-as-judge scoring for RAG pipelines."""

from .retrieval import hit_at_k, mrr, ndcg_at_k
from .judge import Judge, Verdict

__all__ = ["hit_at_k", "mrr", "ndcg_at_k", "Judge", "Verdict"]
__version__ = "0.1.0"

