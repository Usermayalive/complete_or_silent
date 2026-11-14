"""Q6. Certificates: the three honest outcomes, readable and as JSON.

    closure      a "found" answer: the closure reached a fixpoint, every sweep candidate
                 was judged, precedence applied, no unresolved conflict
    absence      a "not found" answer: the calibrated chapters for alpha were all opened,
                 so the miss risk is the canaries' measured rate (with its upper bound)
    uncertified  anything else, with the reasons (budget, unverified sweep, conflict,
                 fewer chapters opened than the calibration asks, no calibration)

A certificate is never a statement that the answer is right. A closure certificate says
what was read around it; an absence certificate says how likely a miss is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .canaries import Calibration
from .closure import Closure
from .plan import CANARY_CLASS, ReadingPlan
from .precedence import Ordering

CLOSURE, ABSENCE, UNCERTIFIED = "closure", "absence", "uncertified"


@dataclass
class Certificate:
    kind: str
    level: Optional[float] = None           # absence: alpha reached
    complete: Optional[bool] = None         # closure: fixpoint reached
    read: Dict = field(default_factory=dict)       # chapters opened, share of chapters, share of tokens
    evidence: Dict = field(default_factory=dict)   # counts: items, forward, reverse, swept, ...
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    def text(self) -> str:
        r = self.read
        read = (f"read {r.get('chapters', 0)} of {r.get('of', 0)} chapters "
                f"({100 * r.get('token_share', 0):.1f}% of the text)")
        if self.kind == CLOSURE:
            e = self.evidence
            return (f"COMPLETE: every provision linked to the answer was read ({e.get('items', 0)} items: "
                    f"{e.get('forward', 0)} forward, {e.get('reverse', 0)} reverse, {e.get('accepted_sweep', 0)} from the "
                    f"sweep; {e.get('superseded', 0)} superseded); {read}.")
        if self.kind == ABSENCE:
            e = self.evidence
            if e.get("basis") == "named":
                return (f"NOT FOUND, certified at {100 * self.level:.0f}%: the named unit was opened whole ({read}); "
                        f"the risk is that the question was routed to the wrong unit (measured separately).")
            return (f"NOT FOUND, certified at {100 * self.level:.0f}%: {read}; on {e.get('canaries', 0)} planted canaries "
                    f"of this document the miss rate at this budget was {100 * e.get('miss', 0):.1f}% "
                    f"(upper bound {100 * e.get('miss_upper', 0):.1f}%).")
        return "UNCERTIFIED: " + "; ".join(self.reasons) + f" ({read})."


def _read(tree, opened: List[str]) -> Dict:
    chapters = [c for c in tree.chapters() if c.role != "contents"]
    total = sum(c.tokens for c in chapters) or 1
    opened_set = set(opened)
    return {"chapters": len(opened_set & {c.id for c in chapters}), "of": len(chapters),
            "token_share": round(sum(c.tokens for c in chapters if c.id in opened_set) / total, 4)}


def closure_certificate(tree, closure: Closure, ordering: Ordering, opened: List[str],
                        verification=None) -> Certificate:
    reasons = []
    if verification is not None:
        if not verification.parsed:
            reasons.append("the verifier's reply could not be read")
        elif verification.unsupported:
            reasons.append(f"{len(verification.unsupported)} of {verification.total} answer sentences are not "
                           f"supported by their quotes")
    if not closure.complete:
        reasons.append(f"closure stopped at the {closure.reason} after {closure.rounds} rounds")
    if closure.unverified:
        reasons.append(f"{closure.unverified} sweep candidates were not judged")
    reasons += ordering.conflicts
    evidence = {**closure.summary(), "accepted_sweep": sum(i.found_by == "sweep" for i in closure.items),
                "superseded": len(ordering.superseded()), "governing": len(ordering.governing()),
                "precedence_rule": ordering.precedence_rule,
                **({"verified": verification.summary()} if verification is not None else {})}
    kind = CLOSURE if not reasons else UNCERTIFIED
    return Certificate(kind, complete=closure.complete, read=_read(tree, opened), evidence=evidence, reasons=reasons)


def absence_certificate(tree, calibration: Optional[Calibration], plan: ReadingPlan, certified: Optional[float],
                        opened: List[str]) -> Certificate:
    read = _read(tree, opened)
    if calibration is None or plan.basis == "uncalibrated":
        return Certificate(UNCERTIFIED, read=read, reasons=["no calibration for this document"])
    if certified is None:
        return Certificate(UNCERTIFIED, read=read,
                           reasons=[f"fewer chapters opened than the calibration asks for ({plan.k} for "
                                    f"{100 * plan.alpha:.0f}%, basis {plan.basis})"])
    if plan.basis == "named":
        evidence = {"basis": "named", "note": "the named unit was opened whole; coverage rests on routing"}
        return Certificate(ABSENCE, level=certified, read=read, evidence=evidence)
    table = calibration.tables[plan.basis]
    prefs = CANARY_CLASS[plan.question_class]
    used = next((c for c in prefs if c in table), "all")
    stats = table[used]
    evidence = {"basis": plan.basis, "class": used, "k": stats.k[str(certified)],
                "canaries": stats.n, "miss": stats.miss_at_k[str(certified)], "miss_upper": stats.miss_upper[str(certified)]}
    return Certificate(ABSENCE, level=certified, read=read, evidence=evidence)
