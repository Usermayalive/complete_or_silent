"""P2. The trap benchmark: questions whose correct answer needs a provision planted far
from the clause that seems to answer them.

Built from the same CUAD contracts as the long benchmark (``longbench.load_contracts``,
same seed, same nested S ⊂ M ⊂ L rooms) into its own family ``traps``:
``data/longbench/traps_{tier}.md`` and ``eval/long/traps_{tier}.jsonl``. The original
rooms are not changed.

Every trap plants a **base clause** as a numbered section in one contract (a liability
cap for a named company, so several traps can live in one contract) and asks for that
company's maximum liability. What makes it a trap:

    far_exception     a "notwithstanding Section N" carve-out far away: later in the same
                      contract, in a schedule of it, or in a side letter (another unit
                      that names the contract)
    override_chain    Amendment No. 1 replaces Section N; Amendment No. 2 replaces it
                      again (two new units at the end of the room): the last amount wins
    unlinked          "the limitations set out above shall not apply to fraud": no section
                      number, so no link; only the sweep can find it
    precedence        a schedule states a different cap and the contract says schedules
                      prevail: the schedule's amount wins, the body's is superseded
    existence         "Does the X contain a change-of-control clause?": yes (a clause is
                      planted) or no (nothing planted; the contract's CUAD label for the
                      category is empty)

Each case carries, in ``tags``: ``required`` (the sentences a complete answer must rest
on, besides the base), ``superseded`` (sentences a correct answer must mark as replaced),
``certificate`` (``closure`` or ``absence``: what a correct system certifies),
``distance_pages`` (base to the farthest required sentence) and a distractor note: every
base clause has a look-alike with other values in another contract.

Text is templated with several wordings per kind (mutated real clauses: later).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..longbench import EVAL, OUT, PAGE_CHARS, TIERS, Unit, _insert, load_contracts

KINDS = ("far_exception", "override_chain", "unlinked", "precedence", "existence")
DEFAULT_COUNTS = {"S": 60, "M": 300, "L": 300}       # traps per tier
PER_CONTRACT = 6                                     # at most this many traps in one contract
FIRST_SECTION = 41                                   # base clauses are numbered from here, per contract

COMPANIES = ["Halvorsen", "Quintrell", "Marbury", "Oduya", "Vasquez", "Petrakis", "Lindgren", "Abernathy",
             "Sorensen", "Bellweather", "Tancredi", "Okafor", "Whitlock", "Ravenscroft", "Delacroix", "Ishikawa"]
SECOND = ["Atlas", "Meridian", "Harbour", "Summit", "Northgate", "Redwood", "Orion", "Keystone", "Lakeside",
          "Pioneer", "Crescent", "Granite", "Beacon", "Silverline", "Evergreen", "Ironbridge"]
SUFFIX = ["Ltd", "GmbH", "Holdings", "Partners", "Industries", "Logistics"]

BASE = [
    "The total liability of {co} under this Agreement shall not exceed {amount}.",
    "In no event shall the aggregate liability of {co} arising out of this Agreement exceed {amount}.",
    "{co}'s liability under this Agreement is limited to {amount} in the aggregate.",
]
EXCEPTION_LINKED = [
    "Notwithstanding Section {n}, the liability of {co} for breaches of its confidentiality obligations shall not exceed {amount2}.",
    "Notwithstanding the limitation in Section {n}, {co} shall be liable for losses arising from data protection breaches up to {amount2}.",
]
EXCEPTION_SIDE = [
    "This side letter supplements the {title}. Notwithstanding Section {n} of the {title}, the liability of {co} for breaches of its confidentiality obligations shall not exceed {amount2}.",
]
UNLINKED = [
    "The limitations of liability set out above shall not apply to losses arising from fraud or wilful misconduct of {co}.",
    "Nothing in the foregoing limitations shall limit the liability of {co} for fraud or wilful misconduct.",
]
AMEND1 = ["Section {n} of the {title} is hereby deleted and replaced in its entirety with the following: "
          "the total liability of {co} under this Agreement shall not exceed {amount2}."]
AMEND2 = ["Section {n} of the {title}, as amended by Amendment No. 1, is hereby deleted and replaced in its entirety with "
          "the following: the total liability of {co} under this Agreement shall not exceed {amount3}."]
SCHEDULE_CAP = ["The total liability of {co} under this Agreement shall not exceed {amount2}."]
PRECEDENCE = ["In the event of any conflict between the body of this Agreement and its Schedules, the Schedules shall prevail."]
EXISTS = ["This Agreement may be terminated by either party upon a change of control of the other party.",
          "Upon any change of control of {co}, the other party may terminate this Agreement on thirty days' notice."]

QUESTION = "What is the maximum liability of {co} under the {title}?"
EXIST_CATEGORY = "Change Of Control"
_COC = re.compile(r"change\s+(?:of|in)\s+control", re.I)
# "No" questions: (CUAD category, clause name in the question, words that must not occur)
ABSENT_TYPES = [
    ("Change Of Control", "change of control", r"change\s+(?:of|in)\s+control"),
    ("Most Favored Nation", "most favored nation", r"most\s+favou?red|\bMFN\b"),
    ("Liquidated Damages", "liquidated damages", r"liquidated\s+damages"),
    ("Non-Compete", "non-compete", r"non-?\s?compet|not\s+compete|shall\s+not\s+(?:directly\s+or\s+indirectly\s+)?engage"),
    ("Price Restrictions", "price restriction", r"price\s+(?:increase|restriction|adjustment)"),
    ("Volume Restriction", "volume restriction", r"volume\s+(?:restriction|limit|cap)"),
]
EXIST_Q_ANY = "Does the {title} contain a {clause} clause?"
EXIST_Q = "Does the {title} contain a change of control clause?"


def _amount(rng: random.Random) -> str:
    return f"{rng.choice([50, 100, 150, 200, 250, 300, 400, 500, 750])},000 USD"


def _company(rng: random.Random, used: set) -> str:
    for _ in range(1000):
        co = f"{rng.choice(COMPANIES)} {rng.choice(SECOND)} {rng.choice(SUFFIX)}"
        if co not in used:
            used.add(co)
            return co
    raise RuntimeError("ran out of company names")


def _section(unit: Unit, number: int, sentence: str, rng: random.Random, lo: float, hi: float) -> float:
    """Plant a numbered section (heading + sentence) so "Section N" resolves to it."""
    block = f"### {number}. Limitation of Liability\n\n{sentence}"
    return _insert(unit, block, rng, lo, hi)


def _schedule(unit: Unit, letter: str, sentence: str) -> float:
    unit.blocks.append(f"### Schedule {letter} Special Terms\n\n{sentence}")
    return 1.0


def _case(cid: str, kind: str, question: str, gold: List[str], answerable: bool, reference: str, **tags) -> Dict:
    return {"id": cid, "document": "", "question": question, "query_type": "unknown", "gold_evidence": gold,
            "answerable": answerable, "reference_answer": reference, "tags": {"family": "traps", "kind": kind, **tags}}


def plant_traps(units: List[Unit], rng: random.Random, counts: Dict[str, int] = DEFAULT_COUNTS) -> List[Dict]:
    cases: List[Dict] = []
    used_co: set = set()
    next_section: Dict[str, int] = {}
    per_unit: Dict[str, int] = {}
    schedules: Dict[str, int] = {}
    extra_units: List[Unit] = []
    planted_coc: set = set()

    def pick(pool: List[Unit]) -> Unit:
        candidates = [u for u in pool if per_unit.get(u.key, 0) < PER_CONTRACT] or pool
        u = rng.choice(candidates)
        per_unit[u.key] = per_unit.get(u.key, 0) + 1
        return u

    def section_number(u: Unit) -> int:
        n = next_section.get(u.key, FIRST_SECTION)
        next_section[u.key] = n + 1
        return n

    for tier in TIERS:
        pool = [u for u in units if u.tier == tier]
        if not pool:
            continue
        for i in range(counts.get(tier, 0)):
            kind = KINDS[i % len(KINDS)]
            u = pick(pool)
            co = _company(rng, used_co)
            amount, amount2, amount3 = _amount(rng), _amount(rng), _amount(rng)
            while amount2 == amount:
                amount2 = _amount(rng)
            while amount3 in (amount, amount2):
                amount3 = _amount(rng)
            cid = f"trap-{kind}-{tier}-{len(cases)}"
            v = {"co": co, "amount": amount, "amount2": amount2, "amount3": amount3, "title": u.title}
            tags = {"unit": u.title, "unit_tier": tier, "company": co}

            if kind == "existence":
                yes = i % 2 == 0
                if yes:
                    s = rng.choice(EXISTS).format(**v)
                    _insert(u, s, rng, 0.2, 0.9)
                    planted_coc.add(u.key)
                    cases.append(_case(cid, kind, EXIST_Q.format(**v), [s], True, "Yes", required=[], superseded=[],
                                       certificate="closure", expected="yes", **tags))
                else:   # the contract is chosen after all planting (see below)
                    cases.append(_case(cid, kind, "", [], False, "No", required=[], superseded=[],
                                       certificate="absence", expected="no", **tags))
                continue

            # every other kind starts from a numbered base clause, plus a look-alike elsewhere
            n = section_number(u)
            v["n"] = n
            base = rng.choice(BASE).format(**v)
            pos_base = _section(u, n, base, rng, 0.05, 0.35)
            others = [x for x in pool if x is not u]
            if others:
                d = dict(v, co=_company(rng, used_co), amount=_amount(rng))
                _section(rng.choice(others), section_number(u) if False else FIRST_SECTION + 30 + i % 7,
                         rng.choice(BASE).format(**d), rng, 0.05, 0.95)
            tags["section"] = n

            if kind == "far_exception":
                where = ("same", "schedule", "side")[i % 3]
                if where == "same":
                    q = rng.choice(EXCEPTION_LINKED).format(**v)
                    _insert(u, q, rng, 0.75, 0.98)
                elif where == "schedule":
                    q = rng.choice(EXCEPTION_LINKED).format(**v)
                    letter = chr(ord("A") + schedules.get(u.key, 0))
                    schedules[u.key] = schedules.get(u.key, 0) + 1
                    _schedule(u, letter, q)
                else:
                    q = rng.choice(EXCEPTION_SIDE).format(**v)
                    side = Unit(f"{u.key}-side-{n}", f"Side Letter to the {u.title}", [q], tier=tier, chars=len(q))
                    extra_units.append(side)
                    tags["second_unit"] = side.title
                tags["where"] = where
                cases.append(_case(cid, kind, QUESTION.format(**v), [base, q], True,
                                   f"{amount}, except {amount2} for breaches of confidentiality or data protection",
                                   required=[q], superseded=[], certificate="closure", **tags))
            elif kind == "override_chain":
                a1 = rng.choice(AMEND1).format(**v)
                a2 = rng.choice(AMEND2).format(**v)
                u1 = Unit(f"{u.key}-am1-{n}", f"Amendment No. 1 to the {u.title}", [a1], tier=tier, chars=len(a1))
                u2 = Unit(f"{u.key}-am2-{n}", f"Amendment No. 2 to the {u.title}", [a2], tier=tier, chars=len(a2))
                extra_units += [u1, u2]
                tags["second_unit"] = u2.title
                cases.append(_case(cid, kind, QUESTION.format(**v), [base, a1, a2], True, amount3,
                                   required=[a2], superseded=[base, a1], certificate="closure", **tags))
            elif kind == "unlinked":
                q = rng.choice(UNLINKED).format(**v)
                _insert(u, q, rng, 0.75, 0.98)
                cases.append(_case(cid, kind, QUESTION.format(**v), [base, q], True,
                                   f"{amount}, but no limit for fraud or wilful misconduct",
                                   required=[q], superseded=[], certificate="closure", **tags))
            elif kind == "precedence":
                cap = rng.choice(SCHEDULE_CAP).format(**v)
                letter = chr(ord("A") + schedules.get(u.key, 0))
                schedules[u.key] = schedules.get(u.key, 0) + 1
                _schedule(u, letter, cap)
                p = PRECEDENCE[0]
                if not any(p in b for b in u.blocks):
                    _insert(u, p, rng, 0.4, 0.7)
                cases.append(_case(cid, kind, QUESTION.format(**v), [base, cap, p], True, amount2,
                                   required=[cap, p], superseded=[base], certificate="closure", **tags))
    # "No" questions: a (contract, clause type) pair where the contract truly lacks the
    # clause: CUAD labels nothing for it, no trap planted it, and none of its key words
    # occur in the contract (amendments and side letters included). Chosen after all
    # planting, with its own generator, so the planted text does not depend on it.
    pick_rng = random.Random(rng.random())
    attached = {}
    for x in extra_units:
        attached.setdefault(x.key.rsplit("-", 2)[0], []).append(x)
    for c in cases:
        if c["tags"]["kind"] != "existence" or c["answerable"]:
            continue
        tier = c["tags"]["unit_tier"]
        pairs = []
        for x in units:
            if x.tier != tier:
                continue
            text = "\n".join(x.blocks + [b for y in attached.get(x.key, []) for b in y.blocks])
            for cat, clause, pattern in ABSENT_TYPES:
                if cat == EXIST_CATEGORY and x.key in planted_coc:
                    continue
                if x.meta.get("spans", {}).get(cat) or re.search(pattern, text, re.I):
                    continue
                pairs.append((x, cat, clause))
        if not pairs:
            raise RuntimeError(f"no contract lacks any of the clause types in tier {tier}")
        target, cat, clause = pick_rng.choice(pairs)
        c["question"] = EXIST_Q_ANY.format(title=target.title, clause=clause)
        c["tags"].update(unit=target.title, absent_category=cat)
    units.extend(extra_units)
    return cases


def render(units: List[Unit], tier: str) -> str:
    keep = [u for u in units if TIERS.index(u.tier) <= TIERS.index(tier)]
    head = f"# Contract data room with traps ({len(keep)} documents)"
    return head + "\n\n" + "\n\n".join(f"## {u.title}\n\n" + "\n\n".join(u.blocks) for u in keep) + "\n"


def build(seed: int = 7, counts: Dict[str, int] = DEFAULT_COUNTS, units: Optional[List[Unit]] = None,
          write: bool = True) -> Tuple[Dict, Dict[str, str], Dict[str, List[Dict]]]:
    """Returns (report, {tier: text}, {tier: rows})."""
    units = copy.deepcopy(units) if units is not None else load_contracts(seed)
    rng = random.Random(seed + 11)
    cases = plant_traps(units, rng, counts)
    report, texts, rows_by_tier = {}, {}, {}
    for tier in TIERS:
        if not counts.get(tier):
            continue
        text = render(units, tier)
        texts[tier] = text
        rows = []
        for c in cases:
            if TIERS.index(c["tags"]["unit_tier"]) > TIERS.index(tier):
                continue
            row = json.loads(json.dumps(c))
            row["document"] = str(Path("..") / ".." / "data" / "longbench" / f"traps_{tier}.md")
            row["tags"]["doc_tier"] = tier
            positions = [text.find(g) for g in row["gold_evidence"]]
            if positions and all(p >= 0 for p in positions):
                row["tags"]["depth"] = round(positions[0] / len(text), 3)
                row["tags"]["distance_pages"] = round((max(positions) - min(positions)) / PAGE_CHARS, 1)
            rows.append(row)
        rows_by_tier[tier] = rows
        kinds: Dict[str, int] = {}
        for r in rows:
            kinds[r["tags"]["kind"]] = kinds.get(r["tags"]["kind"], 0) + 1
        report[f"traps_{tier}"] = {"units": sum(1 for u in units if TIERS.index(u.tier) <= TIERS.index(tier)),
                                   "chars": len(text), "pages": round(len(text) / PAGE_CHARS), "questions": len(rows),
                                   "kinds": kinds}
        if write:
            OUT.mkdir(parents=True, exist_ok=True)
            EVAL.mkdir(parents=True, exist_ok=True)
            (OUT / f"traps_{tier}.md").write_text(text, encoding="utf-8")
            (EVAL / f"traps_{tier}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return report, texts, rows_by_tier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--counts", default="S=60,M=300,L=300")
    args = parser.parse_args()
    counts = {k: int(v) for k, v in (kv.split("=") for kv in args.counts.split(","))}
    report, _, _ = build(args.seed, counts)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
