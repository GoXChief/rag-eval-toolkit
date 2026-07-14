"""LLM-as-judge scoring of RAG answers.

Provider-agnostic: the caller supplies \`complete(prompt) -> str\`. Each quality
dimension is judged in its own call — joint single-call scoring measurably
inflates faithfulness on long answers, because the judge anchors on overall
impression instead of per-claim grounding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

CompleteFn = Callable[[str], str]

_DIMENSIONS: Dict[str, str] = {
    "faithfulness": (
        "Judge FAITHFULNESS only: is every factual claim in the answer supported "
        "by the context? Claims not present in the context count against the score "
        "even if they happen to be true. An explicit refusal or 'I don't know' is "
        "fully faithful (score 1.0)."
    ),
    "relevance": (
        "Judge RELEVANCE only: does the answer address the question that was "
        "actually asked? Ignore whether it is correct."
    ),
    "completeness": (
        "Judge COMPLETENESS only: how much of what the context offers for this "
        "question did the answer actually use? Missing caveats or conditions "
        "lower the score."
    ),
}

_PROMPT = """You are a strict evaluation judge for a RAG system.

{instruction}

Question:
{question}

Context (the only source of truth):
{context}

Answer under evaluation:
{answer}

Respond with ONLY a JSON object, no prose around it:
{{"score": <float 0.0-1.0>, "reason": "<one short sentence>"}}"""


@dataclass
class Verdict:
    faithfulness: Optional[float]
    relevance: Optional[float]
    completeness: Optional[float]
    reasons: Dict[str, str] = field(default_factory=dict)
    judge_errors: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when every dimension produced a score."""
        return not self.judge_errors


class Judge:
    def __init__(self, complete: CompleteFn, retries: int = 2):
        self._complete = complete
        self._retries = retries

    def score(self, question: str, context: str, answer: str) -> Verdict:
        scores: Dict[str, Optional[float]] = {}
        reasons: Dict[str, str] = {}
        errors: Dict[str, str] = {}

        for dim, instruction in _DIMENSIONS.items():
            prompt = _PROMPT.format(
                instruction=instruction,
                question=question,
                context=context,
                answer=answer,
            )
            try:
                score, reason = self._ask(prompt)
                scores[dim], reasons[dim] = score, reason
            except ValueError as exc:
                # Recorded, never silently dropped: gaps must be visible in reports.
                scores[dim] = None
                errors[dim] = str(exc)

        return Verdict(
            faithfulness=scores.get("faithfulness"),
            relevance=scores.get("relevance"),
            completeness=scores.get("completeness"),
            reasons=reasons,
            judge_errors=errors,
        )

    def _ask(self, prompt: str) -> "tuple[float, str]":
        last_err = "no attempts made"
        for _ in range(self._retries + 1):
            raw = self._complete(prompt)
            try:
                return self._parse(raw)
            except ValueError as exc:
                last_err = str(exc)
        raise ValueError(f"judge output unparseable after {self._retries + 1} attempts: {last_err}")

    @staticmethod
    def _parse(raw: str) -> "tuple[float, str]":
        # Models occasionally wrap JSON in code fences or prose; extract the
        # first object rather than failing on cosmetic noise.
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in output: {raw[:120]!r}")
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc

        score = obj.get("score")
        if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"score missing or out of [0, 1]: {score!r}")
        return float(score), str(obj.get("reason", ""))

