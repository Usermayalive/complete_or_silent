"""Q1 + Q2. Question class, target alpha, and the calibrated reading plan.

A reading plan says how many chapters the gate must open so that, if the answer is in the
document, it is among them with probability >= alpha. The number comes from this
document's own calibration (``canaries.py``), not from a fixed k.

Three bases, matching how the gate ranks:

    named    the question names a unit small enough to open whole (a contract): every
             chapter of it is opened; coverage then rests on routing, which is measured
             separately (routing correctness), not on a rank
    unit     the question names a unit with more chapters than the gate opens (a book):
             k from the calibration's "unit" table
    room     no unit named: k from the "room" table

``certified_alpha`` does the reverse after a search: given the chapters actually opened
and the lexical ranking, the highest alpha whose k chapters were all opened. That is the
level an absence certificate can claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .canaries import Calibration

DEFAULT_ALPHA = 0.98
NAMED_AS_FILTER = 6      # same as workers.orchestrator: a named unit with more chapters is ranked within

_EXISTENCE = re.compile(r"^\s*(is|are|does|do|has|have|was|were)\b.*\b(any|a|an|there|contain|include|exist)\b"
                        r"|\b(is there|are there|does .* (?:have|contain|include))\b", re.I)
_LIST = re.compile(r"\b(list|all|every|each|which (?:contracts|agreements|chapters|documents)|how many)\b", re.I)
_FACT = re.compile(r"\b(how (?:much|many|long|often)|what (?:amount|date|rate|price|number|percentage|fee|term)|"
                   r"when|rate|amount|price|fee|date|deadline|days|months|years|percent|usd|\$|€|£)\b", re.I)


def classify_question(question: str) -> str:
    """fact | clause | list | existence (Q1)."""
    if _LIST.search(question):
        return "list"
    if _EXISTENCE.search(question):
        return "existence"
    if _FACT.search(question):
        return "fact"
    return "clause"


# Question class -> canary class whose ranks stand in for it.
CANARY_CLASS = {"fact": ("fact",), "clause": ("type", "clause"), "existence": ("type", "clause"), "list": ("all",)}


@dataclass
class ReadingPlan:
    alpha: float
    question_class: str
    basis: str                    # named | unit | room | uncalibrated
    k: int                        # chapters the gate must open (0 = fixed default)
    chapters: List[str] = field(default_factory=list)   # named chapters, when basis == named
    note: str = ""
    threshold: Optional[float] = None   # basis "threshold": open every chapter scoring >= this share of the top


def reading_plan(calibration: Optional[Calibration], question: str, named: List[str],
                 alpha: float = DEFAULT_ALPHA, default_k: int = 3) -> ReadingPlan:
    qc = classify_question(question)
    if calibration is None:
        return ReadingPlan(alpha, qc, "uncalibrated", default_k, list(named), "no calibration for this document")
    cls = CANARY_CLASS[qc]
    if named and len(named) <= NAMED_AS_FILTER:
        return ReadingPlan(alpha, qc, "named", len(named), list(named),
                           "every chapter of the named unit is opened; coverage rests on routing")
    table = "unit" if named else "room"
    if qc == "list" and not named:     # "which contracts contain X?": every chapter above the threshold (U4.5)
        stats = calibration.stats(("type", "clause", "all"), "room")
        tau = stats.threshold.get(str(alpha))
        if tau is not None:
            return ReadingPlan(alpha, qc, "threshold", 0, [], f"open every chapter scoring >= {tau:.2f} of the top "
                               f"chapter (canary class n={stats.n})", threshold=tau)
    k = calibration.k(alpha, cls, table)
    return ReadingPlan(alpha, qc, table, k, list(named),
                       f"{table} table, class {'/'.join(cls)}: open the lexical top {k} chapters")


def above_threshold(scores: dict, tau: float, chapters: List[str]) -> List[str]:
    """Chapters whose relative score (best / top) is at least ``tau``; with tau 0, all."""
    top = max(scores.values(), default=0.0)
    if tau <= 0 or top <= 0:
        return list(chapters)
    return [c for c in chapters if scores.get(c, 0.0) / top >= tau]


def certified_alpha(calibration: Optional[Calibration], plan: ReadingPlan, opened: List[str],
                    ranking: List[str], scores: Optional[dict] = None) -> Optional[float]:
    """The highest alpha whose calibrated k chapters (by ``ranking``, the lexical order the
    canaries were ranked with, restricted to the unit when the basis is ``unit``, and padded
    with the chapters the words do not match) were all opened; None if none was reached or
    the plan has no calibration. ``opened`` must hold only chapters a worker actually read."""
    if calibration is None or plan.basis == "uncalibrated":
        return None
    opened_set = set(opened)
    if plan.basis == "named":
        return max(calibration.alphas()) if set(plan.chapters) <= opened_set else None
    if plan.basis == "threshold":
        if scores is None:
            return None
        best = None
        for a in calibration.alphas():
            tau = calibration.stats(("type", "clause", "all"), "room").threshold.get(str(a))
            if tau is not None and set(above_threshold(scores, tau, ranking)) <= opened_set:
                best = a if best is None or a > best else best
        return best
    cls = CANARY_CLASS[plan.question_class]
    best = None
    for a in calibration.alphas():
        k = calibration.k(a, cls, plan.basis)
        top = ranking[:k]
        # all k must exist and be opened: a ranking shorter than k (chapters the words do
        # not match are missing from a lexical ranking) never certifies
        if len(top) == k and set(top) <= opened_set and (best is None or a > best):
            best = a
    return best
