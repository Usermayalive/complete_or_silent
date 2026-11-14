"""T6. Trap clauses in the style of real contracts: real CUAD clauses rewritten once by a
model into templates with placeholders, then validated.

Each **role** of the trap benchmark (the base cap, a linked exception, a side letter, an
unlinked carve-out, two amendments, a schedule cap, a precedence clause, a change-of-control
clause, a source-code escrow clause) gets templates. A template is made by showing the
model one real CUAD clause of a related category and asking for a rewrite that keeps its
style but states exactly the role's content, with placeholders:

    {co}        the party                 {title}   the contract's title
    {amount}    base cap                  {amount2} exception / new cap
    {amount3}   the second amendment's cap {n}      the base clause's section number

**Validation** (a template failing any check is dropped, never repaired):
- every placeholder the role needs occurs (amount placeholders exactly once)
- no other money or number is left from the source clause (no "$", "USD", "dollars", no
  digits outside "{n}"), so the gold amount is the only amount
- linked roles cite "Section {n}"; the unlinked carve-out cites no section and names fraud
- 60-600 characters, one or two sentences

**Sources:** clauses from CUAD contracts that are not in the 100- or 1,000-page rooms
(``longbench`` tier L only), so a template never has a near-twin in those rooms.

The library is generated once (``python -m gnn_pruner.certify.clauselib build --llm …``)
and stored as ``eval/traps2/clause_templates.json`` with the source contract and category
of every template.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..longbench import EVAL, load_contracts

LIBRARY = EVAL.parent / "traps2" / "clause_templates.json"

SCHEMA = {"type": "object", "properties": {"clause": {"type": "string"}},
          "required": ["clause"], "additionalProperties": False}
SYSTEM = (
    "You rewrite contract clauses. You are given a real clause from a commercial contract and a required "
    "content. Write ONE new clause in the same legal style, tone and sentence structure as the real clause "
    "(keep its kind of wording, qualifiers and length), but stating exactly the required content and nothing "
    "that contradicts it. Use the placeholders exactly as given, in curly braces, unchanged; never write an "
    "actual amount, date, number or party name in their place. Do not copy any amount, number or party name "
    "from the real clause. Return only the clause.")

# role -> (CUAD source categories, required content, placeholders that must occur)
ROLES: Dict[str, Tuple[List[str], str, List[str]]] = {
    "base": (["Cap On Liability"],
             "{co}'s total liability under this Agreement shall not exceed {amount}.", ["{co}", "{amount}"]),
    "exception": (["Cap On Liability", "Uncapped Liability"],
                  "Notwithstanding Section {n}, the liability of {co} for breaches of its confidentiality or data "
                  "protection obligations shall not exceed {amount2}.", ["{co}", "{amount2}", "Section {n}"]),
    "side": (["Cap On Liability", "Uncapped Liability"],
             "This side letter supplements the {title}. Notwithstanding Section {n} of the {title}, the liability of "
             "{co} for breaches of its confidentiality obligations shall not exceed {amount2}.",
             ["{co}", "{amount2}", "Section {n}", "{title}"]),
    "unlinked": (["Uncapped Liability"],
                 "The limitations of liability set out above shall not apply to losses arising from fraud or wilful "
                 "misconduct of {co}.", ["{co}"]),
    "amend1": (["Cap On Liability"],
               "Section {n} of the {title} is hereby deleted and replaced in its entirety with the following: the total "
               "liability of {co} under this Agreement shall not exceed {amount2}.",
               ["{co}", "{amount2}", "Section {n}", "{title}"]),
    "amend2": (["Cap On Liability"],
               "Section {n} of the {title}, as amended by Amendment No. 1, is hereby deleted and replaced in its "
               "entirety with the following: the total liability of {co} under this Agreement shall not exceed "
               "{amount3}.", ["{co}", "{amount3}", "Section {n}", "{title}", "Amendment No. 1"]),
    "schedule_cap": (["Cap On Liability"],
                     "{co}'s total liability under this Agreement shall not exceed {amount2}.", ["{co}", "{amount2}"]),
    # schedule_cap, amend1 and amend2 are also derived from the base templates (``derived``)
    "precedence": (["Cap On Liability", "Third Party Beneficiary"],
                   "In the event of any conflict between the body of this Agreement and its Schedules, the Schedules "
                   "shall prevail.", ["Schedule"]),
    "change_of_control": (["Change Of Control"],
                          "Either party may terminate this Agreement upon a change of control of {co}.", ["{co}"]),
    "escrow": (["Source Code Escrow"],
               "{co} shall deposit the source code of the licensed software with an independent escrow agent, to be "
               "released to the licensee if {co} becomes insolvent or ceases to support the software.", ["{co}"]),
}
PER_ROLE = 25
_MONEY = re.compile(r"\$|\bUSD\b|\bdollars?\b|\beuros?\b|€|£", re.I)
_PLACEHOLDER = re.compile(r"\{(co|title|amount|amount2|amount3|n)\}")


def validate(role: str, text: str) -> Optional[str]:
    """None if the template is usable, else the reason it is not."""
    _, _, required = ROLES[role]
    text = text.strip()
    if not 60 <= len(text) <= 600:
        return "length"
    if len(re.findall(r"[.;]\s+[A-Z]", text)) > 2:
        return "too many sentences"
    for r in required:
        if r not in text:
            return f"missing {r}"
    for amt in ("{amount}", "{amount2}", "{amount3}"):
        if text.count(amt) > 1:
            return f"{amt} more than once"
    stripped = _PLACEHOLDER.sub("", text).replace("Amendment No. 1", "")
    if _MONEY.search(stripped):
        return "stray money"
    if re.search(r"\d", stripped):
        return "stray number"
    unknown = set(re.findall(r"\{([^}]*)\}", text)) - {"co", "title", "amount", "amount2", "amount3", "n"}
    if unknown:
        return f"unknown placeholder {unknown}"
    if role == "unlinked" and ("Section" in text or "fraud" not in text.lower()):
        return "unlinked must name fraud and cite no section"
    if role in ("base", "unlinked", "change_of_control", "escrow") and "{amount2}" in text:
        return "wrong amount placeholder"
    if role == "schedule_cap" and "{amount}" in text:
        return "wrong amount placeholder"
    if role == "precedence":
        from .precedence import precedence_winner

        if precedence_winner(text) != "schedules":
            return "schedules do not prevail"
    if role in ("base",) and ("{amount}" not in text):
        return "missing {amount}"
    return None


def sources(seed: int = 7, exclude_tiers=("S", "M")) -> Dict[str, List[Tuple[str, str]]]:
    """category -> [(contract title, clause)] from contracts outside the excluded tiers."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    cats = {c for cs, _, _ in ROLES.values() for c in cs}
    for u in load_contracts(seed):
        if u.tier in exclude_tiers:
            continue
        for cat, spans in u.meta.get("spans", {}).items():
            if cat in cats:
                for s in spans:
                    s = re.sub(r"\s+", " ", s).strip()
                    if 60 < len(s) < 700 and "[***]" not in s and "[*]" not in s:
                        out.setdefault(cat, []).append((u.title, s))
    return out


def build(provider, per_role: int = PER_ROLE, seed: int = 7, max_tries: int = 2, log=print) -> Dict:
    rng = random.Random(seed)
    pool = sources(seed)
    library = {"seed": seed, "per_role": per_role, "roles": {}, "rejected": {}}
    for role, (cats, content, _) in ROLES.items():
        candidates = [x for c in cats for x in pool.get(c, [])]
        rng.shuffle(candidates)
        kept, rejected, calls = [], {}, 0
        for title, clause in candidates:
            if len(kept) >= per_role or calls >= per_role * max_tries:
                break
            user = (f"Real clause:\n{clause}\n\nRequired content:\n{content}\n\nPlaceholders to use exactly: "
                    f"{', '.join(sorted(set(_PLACEHOLDER.findall(content)), key=str))}")
            completion = provider.complete(SYSTEM, user, SCHEMA)
            calls += 1
            try:
                text = str(json.loads(completion.text).get("clause", "")).strip()
            except (json.JSONDecodeError, TypeError, AttributeError):
                rejected["unparsed"] = rejected.get("unparsed", 0) + 1
                continue
            reason = validate(role, text)
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            if text not in [k["text"] for k in kept]:
                kept.append({"text": text, "source_title": title, "source_category": next(c for c in cats if any(
                    t == title and cl == clause for t, cl in pool.get(c, [])))})
        library["roles"][role] = kept
        library["rejected"][role] = rejected
        log(f"{role:18s} kept {len(kept):2d} of {calls} calls; rejected {rejected}")
    return library


AMEND1_FRAME = "Section {n} of the {title} is hereby deleted and replaced in its entirety with the following: "
AMEND2_FRAME = ("Section {n} of the {title}, as amended by Amendment No. 1, is hereby deleted and replaced in its "
                "entirety with the following: ")


def derived(library: Dict) -> Dict[str, List[str]]:
    """Real-style variants without new model calls: a base cap clause restated in a
    schedule (cap -> {amount2}) and wrapped in the amendment frames (-> {amount2} /
    {amount3}). Each is validated like a generated template."""
    out: Dict[str, List[str]] = {"schedule_cap": [], "amend1": [], "amend2": []}
    for item in library.get("roles", {}).get("base", []):
        base = item["text"]
        lower = base[0].lower() + base[1:] if not base[:2].isupper() else base
        for role, text in (("schedule_cap", base.replace("{amount}", "{amount2}")),
                           ("amend1", AMEND1_FRAME + lower.replace("{amount}", "{amount2}")),
                           ("amend2", AMEND2_FRAME + lower.replace("{amount}", "{amount3}"))):
            if validate(role, text) is None:
                out[role].append(text)
    return out


def load(path: Path = LIBRARY) -> Dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--llm", required=True)
    parser.add_argument("--per-role", type=int, default=PER_ROLE)
    args = parser.parse_args()
    from ..answer import load_provider

    library = build(load_provider(args.llm), args.per_role)
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    LIBRARY.write_text(json.dumps(library, indent=1, ensure_ascii=False))
    print(f"wrote {LIBRARY}: " + ", ".join(f"{r} {len(v)}" for r, v in library["roles"].items()))


if __name__ == "__main__":
    main()
