# rag-eval-toolkit

Small, dependency-free toolkit for evaluating RAG (Retrieval-Augmented Generation) pipelines:
**retrieval metrics** (hit@k, MRR, nDCG) + **LLM-as-judge answer scoring** (faithfulness, relevance, completeness) with a bring-your-own-model interface.

Built out of production need: shipping an LLM support agent taught me that without a regression eval set, every prompt or KB edit is a gamble. This is the distilled, generic version of that eval harness.

## Install

```bash
pip install -e .
```

No runtime dependencies. Any LLM client works — you pass a `complete(prompt) -> str` callable.

## Retrieval metrics

```python
from rag_eval import hit_at_k, mrr, ndcg_at_k

# ranked doc ids from your retriever vs. the known-relevant ids
retrieved = ["doc_7", "doc_2", "doc_9"]
relevant  = {"doc_2"}

hit_at_k(retrieved, relevant, k=3)   # 1.0
mrr(retrieved, relevant)             # 0.5
ndcg_at_k(retrieved, relevant, k=3)  # 0.63
```

## LLM-as-judge answer scoring

```python
from rag_eval import Judge

def complete(prompt: str) -> str:
    ...  # call your LLM of choice and return the text

judge = Judge(complete)
verdict = judge.score(
    question="How do I reset my device limit?",
    context="Device slots can be reset once every 30 days from the cabinet…",
    answer="You can reset device slots in the cabinet, once per 30 days.",
)
verdict.faithfulness   # 1.0  — every claim grounded in context
verdict.relevance      # 1.0  — answers the question asked
verdict.completeness   # 0.5..1.0
verdict.reasons        # judge's short rationale per dimension
```

The judge prompt forces **structured JSON output**, retries on malformed responses, and treats "I don't know" as faithful (refusing beats hallucinating).

## Regression runs

```bash
python -m rag_eval.cli run examples/dataset.jsonl --out report.json
```

`dataset.jsonl` — one case per line: `{"question", "context", "answer", "relevant_ids", "retrieved_ids"}`.
The CLI aggregates means per metric and exits non-zero if any metric drops below the thresholds in `--min-hit` / `--min-mrr`. Wire it into CI to catch prompt/KB regressions before deploy.

## Design notes

- **Judge bias controls**: scoring dimensions are judged independently (one prompt per dimension) — cheaper single-call scoring consistently inflated faithfulness on long answers.
- **No framework lock-in**: functions take plain lists/strings. Your retriever and generator stay yours.
- **Failure is data**: malformed judge output after retries is recorded as a `judge_error`, never silently dropped — silent gaps make weekly metric drift unreadable.

## License

MIT

