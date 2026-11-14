"""Q5. Precedence: which closure item governs, which applies, which is superseded.

Rules, in order:

1. **Explicit precedence clause** of the contract ("in the event of any conflict, the
   Schedules shall prevail over the body"): items in the winning part outrank the rest.
2. **Later amendment wins.** An ``amends`` item supersedes the clause it targets. An
   amendment is taken to be later than what it amends unless both carry dates (in their
   chapter titles, "(2022-06-26)") and the amendment's is earlier: then nothing is
   superseded and a conflict is flagged.
3. **Specific over general.** An ``overrides`` item ("notwithstanding Section 12.3 …")
   applies within its own scope; the general clause still governs outside it.

Superseded text is kept and marked, never dropped: the answer says "24 months (the
12-month cap in 12.3 was replaced by Amendment 2)". Two governing items that amend the
same clause are a conflict the certificate must report.

Statuses:  governs | applies | superseded | rule | context
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..tree import LIFT_TO_CHAPTER
from .closure import Closure, ClosureItem
from .qualifiers import AMENDS, OVERRIDES, PRECEDENCE, QUALIFIES

GOVERNS, APPLIES, SUPERSEDED, RULE, CONTEXT = "governs", "applies", "superseded", "rule", "context"
_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_SCHEDULES_WIN = re.compile(r"\b(schedules?|exhibits?|annex(?:es)?|appendi(?:x|ces))\b[^.]{0,60}\b(prevail|control|govern|precedence)",
                            re.I)
_BODY_WINS = re.compile(r"\b(this agreement|the agreement|the body|main body)\b[^.]{0,60}\b(prevail|control|govern|precedence)",
                        re.I)


@dataclass
class Ruling:
    item: ClosureItem
    status: str
    rule: str = ""              # precedence clause | later amendment | specific over general | carve-out
    by: Optional[str] = None    # passage id of the item that outranks this one


@dataclass
class Ordering:
    rulings: List[Ruling] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    precedence_rule: str = ""   # schedules | body | ""

    def status_of(self, passage: str) -> Optional[str]:
        return next((r.status for r in self.rulings if r.item.passage == passage), None)

    def superseded(self) -> List[Ruling]:
        return [r for r in self.rulings if r.status == SUPERSEDED]

    def governing(self) -> List[Ruling]:
        return [r for r in self.rulings if r.status == GOVERNS]


def _chapter_title(tree, passage: str) -> str:
    node = tree.node(passage)
    while node.parent not in (None, "root"):
        node = tree.node(node.parent)
    return node.title or ""


def _date(tree, passage: str) -> Optional[tuple]:
    m = _DATE.search(_chapter_title(tree, passage))
    return tuple(int(x) for x in m.groups()) if m else None


_AMEND_NO = re.compile(r"\bamendment\s+(?:no\.?|number|#)\s*(\d+)", re.I)


def _amendment_number(tree, passage: str) -> Optional[int]:
    m = _AMEND_NO.search(_chapter_title(tree, passage))
    return int(m.group(1)) if m else None


def _in_schedule(tree, passage: str) -> bool:
    return bool(LIFT_TO_CHAPTER.match(_chapter_title(tree, passage)))


_PREVAIL = re.compile(r"\b(shall|will|should)?\s*(prevail|control|govern|take[s]?\s+precedence)\b", re.I)
_SCHEDULE_WORD = re.compile(r"\b(schedules?|exhibits?|annex(?:es)?|appendi(?:x|ces))\b", re.I)
_BODY_WORD = re.compile(r"\b(this agreement|the agreement|the body|main body|these terms)\b", re.I)


_PRECEDENCE_CUE = next(p for kind, p in __import__("gnn_pruner.certify.qualifiers", fromlist=["CUES"]).CUES
                       if kind == PRECEDENCE)


def precedence_winner(text: str) -> str:
    """"schedules" | "body" | "": the side named closest before the prevail verb (its
    subject). "In the event of a conflict between the Schedules and this Agreement, this
    Agreement shall prevail" -> "body"."""
    m = _PREVAIL.search(text)
    if not m:
        return ""
    before = text[:m.start()]
    last_s = max((x.end() for x in _SCHEDULE_WORD.finditer(before)), default=-1)
    last_b = max((x.end() for x in _BODY_WORD.finditer(before)), default=-1)
    if last_s < 0 and last_b < 0:
        return ""
    return "schedules" if last_s > last_b else "body"


def order(tree, closure: Closure, seeds: List[ClosureItem] = None) -> Ordering:
    """Rank the seeds and the closure items. ``seeds`` default to the closure's seed
    passages as answer items."""
    out = Ordering()
    seeds = seeds or [ClosureItem(p, [], "answer", "seed", "", "structural") for p in closure.seeds]
    rulings: Dict[str, Ruling] = {}
    for s in seeds:
        rulings[s.passage] = Ruling(s, GOVERNS)
    for item in closure.items:
        text = tree.quote(item.sentence_ids)
        # an order-of-precedence statement is the rule however it joined the closure (its
        # "shall prevail" also reads as an override cue, so the sweep may bring it first)
        if item.relation == PRECEDENCE or (_PRECEDENCE_CUE.search(text) and precedence_winner(text)):
            rule = precedence_winner(text)
            if rule:
                out.precedence_rule = rule
            rulings[item.passage] = Ruling(item, RULE, "precedence clause")
        elif item.relation == AMENDS:
            rulings[item.passage] = Ruling(item, GOVERNS, "later amendment")
        elif item.relation == OVERRIDES:
            rulings[item.passage] = Ruling(item, APPLIES, "specific over general")
        elif item.relation == QUALIFIES:
            rulings[item.passage] = Ruling(item, APPLIES, "carve-out")
        elif item.relation == "restates":
            rulings[item.passage] = Ruling(item, CONTEXT, "restatement")   # governs only through a precedence clause
        else:
            rulings[item.passage] = Ruling(item, CONTEXT)

    # 2. later amendment supersedes what it targets
    for item in closure.items:
        if item.relation != AMENDS:
            continue
        target = rulings.get(item.via)
        if target is None or target.status not in (GOVERNS, APPLIES):   # a deleted carve-out is superseded too
            continue
        d_amend, d_target = _date(tree, item.passage), _date(tree, item.via)
        if d_amend and d_target and d_amend < d_target:
            out.conflicts.append(f"amendment {item.passage} is dated earlier than {item.via}: not applied")
            rulings[item.passage].status = CONTEXT
            continue
        target.status, target.rule, target.by = SUPERSEDED, "later amendment", item.passage

    # 1. explicit precedence clause: the losing part cannot govern over the winning part
    if out.precedence_rule:
        for r in rulings.values():
            if r.status not in (GOVERNS, APPLIES):
                continue
            wins = _in_schedule(tree, r.item.passage) if out.precedence_rule == "schedules" \
                else not _in_schedule(tree, r.item.passage)
            if r.status == APPLIES and wins and not r.rule.startswith("precedence"):
                r.rule = "precedence clause"
        in_winning_part = lambda r: _in_schedule(tree, r.item.passage) == (out.precedence_rule == "schedules")
        # a restatement of the same matter in the winning part governs (a carve-out, which
        # is scoped, does not); governing statements in the losing part are superseded
        winners = [r for r in rulings.values() if r.status in (GOVERNS, APPLIES, CONTEXT) and in_winning_part(r)
                   and r.item.relation in ("restates", "answer")]
        losers = [r for r in rulings.values() if r.status == GOVERNS and not in_winning_part(r)]
        if winners and losers:
            for r in winners:
                if r.status != GOVERNS:
                    r.status, r.rule = GOVERNS, "precedence clause"
            for r in losers:
                r.status, r.rule, r.by = SUPERSEDED, "precedence clause", winners[0].item.passage

    # several amendments of one clause: the later one (by number, else by date) governs;
    # otherwise a conflict
    amendments = [i for i in closure.items if i.relation == AMENDS and rulings[i.passage].status == GOVERNS]
    by_target: Dict[str, List[ClosureItem]] = {}
    for a in amendments:
        by_target.setdefault(a.via, []).append(a)
    for target, items in by_target.items():
        if len(items) < 2:
            continue
        keyed = [(_amendment_number(tree, i.passage), _date(tree, i.passage), i) for i in items]
        if all(k[0] is not None for k in keyed) and len({k[0] for k in keyed}) == len(keyed):
            keyed.sort(key=lambda k: k[0])
        elif all(k[1] is not None for k in keyed) and len({k[1] for k in keyed}) == len(keyed):
            keyed.sort(key=lambda k: k[1])
        else:
            out.conflicts.append(f"{len(items)} amendments govern {target}: {', '.join(i.passage for i in items)}")
            continue
        latest = keyed[-1][2]
        for _, _, earlier in keyed[:-1]:
            r = rulings[earlier.passage]
            r.status, r.rule, r.by = SUPERSEDED, "later amendment", latest.passage
    out.rulings = list(rulings.values())
    return out
