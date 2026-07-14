"""Regression runner: evaluate a JSONL dataset and gate on minimum thresholds.

Usage:
    python -m rag_eval.cli run dataset.jsonl --out report.json \\
        --min-hit 0.8 --min-mrr 0.6

Each dataset line: {"question", "context", "answer", "relevant_ids",
"retrieved_ids"}. Judge dimensions are only computed when the caller wires an
LLM in code (see README); the CLI itself covers retrieval metrics, which is
what CI usually gates on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .retrieval import hit_at_k, mrr, ndcg_at_k


def _load_cases(path: Path):
    cases = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.exit(f"{path}:{lineno}: invalid JSON: {exc}")
        missing = {"retrieved_ids", "relevant_ids"} - case.keys()
        if missing:
            sys.exit(f"{path}:{lineno}: missing fields: {sorted(missing)}")
        cases.append(case)
    if not cases:
        sys.exit(f"{path}: dataset is empty")
    return cases


def run(args: argparse.Namespace) -> int:
    cases = _load_cases(Path(args.dataset))

    per_case = []
    for case in cases:
        retrieved, relevant = case["retrieved_ids"], set(case["relevant_ids"])
        per_case.append(
            {
                "question": case.get("question", ""),
                "hit@k": hit_at_k(retrieved, relevant, k=args.k),
                "mrr": mrr(retrieved, relevant),
                "ndcg@k": ndcg_at_k(retrieved, relevant, k=args.k),
            }
        )

    n = len(per_case)
    summary = {
        "cases": n,
        "k": args.k,
        "hit@k": sum(c["hit@k"] for c in per_case) / n,
        "mrr": sum(c["mrr"] for c in per_case) / n,
        "ndcg@k": sum(c["ndcg@k"] for c in per_case) / n,
    }
    report = {"summary": summary, "cases": per_case}

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2))

    failed = []
    if summary["hit@k"] < args.min_hit:
        failed.append(f"hit@k {summary['hit@k']:.3f} < {args.min_hit}")
    if summary["mrr"] < args.min_mrr:
        failed.append(f"mrr {summary['mrr']:.3f} < {args.min_mrr}")
    if failed:
        print("THRESHOLDS FAILED: " + "; ".join(failed), file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="evaluate a JSONL dataset")
    p_run.add_argument("dataset")
    p_run.add_argument("--out", help="write full JSON report here")
    p_run.add_argument("--k", type=int, default=5)
    p_run.add_argument("--min-hit", type=float, default=0.0)
    p_run.add_argument("--min-mrr", type=float, default=0.0)
    p_run.set_defaults(func=run)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()

