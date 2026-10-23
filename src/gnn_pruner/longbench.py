"""Long-document benchmark (build step 2): documents of ~100, ~1,000 and ~10,000 pages with
known-answer questions.

Two families, each in three nested sizes (S inside M inside L, so the same questions are
asked at every size and any change in a score comes from document size alone):

- **contracts:** a "data room" of real contracts from CUAD (510 contracts with clauses
  labelled by lawyers, CC BY 4.0). Each contract is a chapter.
  - Questions ask about one contract's clauses (governing law, expiry, liability cap ...),
    naming it by its chapter title.
  - Gold: the lawyers' clause spans. A clause CUAD marks as absent gives an unanswerable
    question.
- **books:** a library of public-domain novels from NarrativeQA's test set (Gutenberg
  texts, Apache-2.0 question set). Each book is a part, its chapters are chapters.
  - Questions are NarrativeQA's human questions ("In <title>, ...") with two human answers;
    they have no evidence spans.

Both families also get **planted facts** ("needles") at known places:
- single facts, asked with reworded questions
- the same kind of fact with different values elsewhere (distractors)
- two-part facts split across distant chapters (two hops)
- questions about facts that were never planted (unanswerable)

Their exact sentences are the gold evidence.

Page = 3,000 characters (about 500 words). Built deterministically from a seed:

    python -m gnn_pruner.longbench build      # writes data/longbench/*.md and eval/long/*.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .parser import match_heading

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "longbench" / "raw"
OUT = ROOT / "data" / "longbench"
EVAL = ROOT / "eval" / "long"
PAGE_CHARS = 3000
TIER_PAGES = {"S": 100, "M": 1000, "L": 10000}
TIERS = ("S", "M", "L")

# CUAD categories asked about, with question templates ({X} names the contract).
CONTRACT_QUESTIONS = {
    "Governing Law": "Which jurisdiction's law governs {X}?",
    "Agreement Date": "On what date was {X} made?",
    "Expiration Date": "When does {X} expire, or how long is its term?",
    "Renewal Term": "Does {X} renew, and for how long?",
    "Notice Period To Terminate Renewal": "How much notice is needed to stop {X} from renewing?",
    "Cap On Liability": "Is liability limited or capped under {X}?",
    "Termination For Convenience": "Can a party terminate {X} for convenience, and with what notice?",
    "Non-Compete": "Does {X} restrict either party from competing?",
    "Exclusivity": "Does {X} grant exclusive rights?",
    "Audit Rights": "Does {X} give a right to audit or inspect records?",
    "Insurance": "What insurance does {X} require?",
    "Minimum Commitment": "Is there a minimum purchase or commitment under {X}?",
    "Anti-Assignment": "Can {X} be assigned without the other party's consent?",
    "Change Of Control": "What happens under {X} if a party undergoes a change of control?",
    "Liquidated Damages": "Does {X} provide for liquidated damages?",
    "Warranty Duration": "How long do the warranties last under {X}?",
    "Post-Termination Services": "What obligations continue after {X} ends?",
    "Revenue/Profit Sharing": "Does {X} involve sharing revenue or profits?",
    "Ip Ownership Assignment": "Who owns intellectual property created under {X}?",
    "License Grant": "What licence is granted under {X}?",
}

FIRST = ["Harriet", "Tobias", "Marguerite", "Ansel", "Ines", "Caspar", "Leonie", "Rufus", "Odile", "Bram",
         "Celestine", "Ezra", "Wilhelmina", "Jasper", "Perpetua", "Lorcan", "Ottilie", "Silas", "Henrike", "Fenwick"]
LAST = ["Okonkwo", "Wren", "Vance", "Lindqvist", "Marchetti", "Holloway", "Achterberg", "Quill", "Brannigan",
        "Delacroix", "Strand", "Pemberton", "Vasquez", "Thorne", "Ellingham", "Moreau", "Kowalczyk", "Ashdown"]
CITIES = ["Lisbon", "Tromso", "Valparaiso", "Ljubljana", "Hobart", "Antwerp", "Cork", "Gdansk", "Porto",
          "Tallinn", "Bergen", "Trieste", "Galway", "Aarhus", "Utrecht", "Bilbao"]
COLORS = ["blue", "green", "yellow", "crimson", "grey", "violet", "copper", "white"]
THINGS = ["lighthouse", "chapel", "granary", "boathouse", "observatory", "toll-house"]
ANIMALS = ["heron", "fox", "lantern-moth", "badger", "kingfisher", "hare"]
WORDS = ["juniper", "marrowbone", "saltpetre", "candlewick", "foxglove", "quartermoon"]

# (kind, planted sentence, question, answer). {X} names the contract / book.
CONTRACT_NEEDLES = [
    ("mediator", "All disputes under this Agreement shall first be referred to the mediator {name} of {city}.",
     "Who is the first mediator for disputes under {X}?", "{name} of {city}"),
    ("escrow", "The escrow agent shall hold the security deposit for this Agreement under reference number {code}.",
     "Under what reference number is the security deposit for {X} held in escrow?", "{code}"),
    ("ceremony", "This Agreement was executed in duplicate at a signing ceremony held in {city}.",
     "Where was {X} signed?", "{city}"),
    ("audit_fee", "Each additional audit requested under this Agreement is charged at {amount} per day.",
     "What daily charge applies to an extra audit under {X}?", "{amount} per day"),
    ("liaison", "Copies of all operational notices under this Agreement must be sent to the liaison officer, {name}.",
     "To whom must operational notices under {X} be copied?", "{name}"),
]
CONTRACT_TWO_HOP = ("rate_card",
                    "The hourly service charges under this Agreement are set out in rate card {code}.",
                    "Rate card {code} fixes the hourly rate for consulting services at {amount}.",
                    "What hourly rate applies to consulting services under {X}?", "{amount}")
BOOK_NEEDLES = [
    ("key", "{name} kept the {thing} key hidden inside a {color} tin of peppermints.",
     "In {X}, where did {name} hide the {thing} key?", "inside a {color} tin of peppermints"),
    ("note", "The note that {name} received was signed only with a small drawing of a {animal}.",
     "In {X}, how was the note that {name} received signed?", "with a drawing of a {animal}"),
    ("debt", "{name} owed the innkeeper exactly {n} shillings, a debt nobody else knew about.",
     "In {X}, how much did {name} owe the innkeeper?", "{n} shillings"),
    ("password", "To be let into the cellar, {name} had to whisper the word '{word}'.",
     "In {X}, what word did {name} whisper to get into the cellar?", "{word}"),
]
# Kinds never planted anywhere: unanswerable questions when every planted kind is used in a unit.
CONTRACT_NEVER = ("courier", "Which courier firm is appointed to carry original documents under {X}?")
BOOK_NEVER = ("umbrella", "In {X}, what colour was the umbrella that {name} left at the station?")
BOOK_TWO_HOP = ("parcel",
                "{name} said the parcel must be left with the man who wore a {color} scarf.",
                "The man who wore a {color} scarf was the ferryman, {name2}.",
                "In {X}, who was {name}'s parcel to be left with?", "{name2}, the ferryman")


@dataclass
class Unit:
    """One contract or one book: a chapter (contracts) or part (books) of every tier it is in."""
    key: str
    title: str
    blocks: List[str]                     # markdown blocks (paragraphs / headings) of the body
    tier: str = "L"                       # smallest tier containing it
    chars: int = 0
    questions: List[Dict] = field(default_factory=list)   # dataset questions (CUAD / NarrativeQA)
    meta: Dict = field(default_factory=dict)


# ===================================================================== helpers

def _ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _fill(template: str, values: Dict) -> str:
    return template.format(**values)


def _values(rng: random.Random, used: set) -> Dict:
    for _ in range(1000):
        v = {"name": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "name2": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
             "city": rng.choice(CITIES), "code": f"RC-{rng.randint(1000, 9999)}",
             "amount": f"{rng.randint(120, 980)} USD", "color": rng.choice(COLORS), "thing": rng.choice(THINGS),
             "animal": rng.choice(ANIMALS), "word": rng.choice(WORDS), "n": str(rng.randint(13, 97))}
        key = (v["name"], v["code"], v["amount"])
        if key not in used and v["name"] != v["name2"]:
            used.add(key)
            return v
    raise RuntimeError("ran out of needle values")


def _assign_tiers(units: List[Unit]) -> None:
    """Units in the given order fill S, then M, then L (nested by construction)."""
    total = 0
    for u in units:
        u.tier = "S" if total < TIER_PAGES["S"] * PAGE_CHARS else "M" if total < TIER_PAGES["M"] * PAGE_CHARS else "L"
        total += u.chars


def _insert(unit: Unit, sentence: str, rng: random.Random, lo: float = 0.1, hi: float = 0.9) -> float:
    """Insert ``sentence`` as its own paragraph between body blocks; returns its relative position."""
    slots = [i for i in range(1, len(unit.blocks)) if not unit.blocks[i].startswith("#")]
    if not slots:
        unit.blocks.append(sentence)
        return 1.0
    a, b = int(len(slots) * lo), max(int(len(slots) * hi), int(len(slots) * lo) + 1)
    slot = slots[rng.randrange(a, min(b, len(slots)))]
    unit.blocks.insert(slot, sentence)
    return round(slot / len(unit.blocks), 3)


# ================================================================== contracts

_HEADING_LINE = re.compile(r"^(article|section)\s+[\divxlc]+\.?\s*[-:.]?\s*[A-Z][A-Za-z ,&/-]{0,60}$", re.I)


def _contract_blocks(text: str) -> List[str]:
    """CUAD text -> markdown blocks; numbered / ARTICLE headings become "###+" headings so
    they nest under the contract's "##" title."""
    blocks = []
    for raw in re.split(r"\n\s*\n", text):
        block = _ws(raw)
        if not block:
            continue
        if len(block) <= 90:
            h = match_heading(block)
            if h and not block.startswith("#"):
                blocks.append("#" * min(6, 2 + h[0]) + " " + block)
                continue
            if _HEADING_LINE.match(block):
                blocks.append("### " + block)
                continue
        blocks.append(block.lstrip("#").strip() or block)
    return blocks


def _contract_title(cuad_title: str, doc_name: str) -> str:
    company = cuad_title.split("_")[0]
    m = re.search(r"_(\d{2})_(\d{2})_(\d{4})", cuad_title)
    date = f"{m.group(3)}-{m.group(1)}-{m.group(2)}" if m else ""
    name = _ws(doc_name) or cuad_title.split("-")[-1]
    if name.isupper():
        name = name.title()
    return f"{name} of {company}" + (f" ({date})" if date else "")


def load_contracts(seed: int) -> List[Unit]:
    data = json.loads((RAW / "CUADv1.json").read_text())["data"]
    units, seen = [], {}
    for item in data:
        para = item["paragraphs"][0]
        qas = {q["id"].split("__")[-1]: [_ws(a["text"]) for a in q["answers"]] for q in para["qas"]}
        title = _contract_title(item["title"], (qas.get("Document Name") or [""])[0])
        if title in seen:
            seen[title] += 1
            title = f"{title} #{seen[title]}"
        seen.setdefault(title, 1)
        u = Unit(item["title"], title, _contract_blocks(para["context"]), chars=len(para["context"]))
        u.meta["spans"] = qas
        units.append(u)
    random.Random(seed).shuffle(units)
    _assign_tiers(units)
    return units


# ====================================================================== books

_CHAPTER = re.compile(r"^(chapter|chap\.|stave|letter|canto|book|part|volume)\s+([0-9ivxlcdm]+|[a-z]+)\b.{0,70}$", re.I)
_PART = re.compile(r"^(book|part|volume)\b", re.I)


def _gutenberg_body(text: str) -> str:
    start = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I | re.S)
    end = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", text, re.I)
    return text[start.end() if start else 0: end.start() if end else len(text)]


def _book_blocks(text: str) -> List[str]:
    """Plain text -> markdown blocks; "CHAPTER IV ..." lines become "###" chapters; a book's
    own BOOK / PART / VOLUME lines stay as text (the book itself is the part)."""
    blocks = []
    for raw in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        block = _ws(raw)
        if not block:
            continue
        if len(block) <= 80 and _CHAPTER.match(block) and not _PART.match(block):
            blocks.append("### " + block)
        else:
            blocks.append(block.lstrip("#").strip() or block)
    return blocks


def _download(gid: str) -> Optional[str]:
    path = RAW / "books" / f"{gid}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    # curl, not urllib: on this machine Python's TLS handshakes to Gutenberg stall for minutes.
    for url in (f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
                f"https://www.gutenberg.org/ebooks/{gid}.txt.utf-8"):
        result = subprocess.run(["curl", "-sfL", "--max-time", "90", "-A", "gnn-pruner-benchmark/1.0", url],
                                capture_output=True)
        if result.returncode == 0 and len(result.stdout) > 10000:
            text = result.stdout.decode("utf-8", errors="replace")
            path.write_text(text, encoding="utf-8")
            time.sleep(1.0)  # be polite to Project Gutenberg
            return text
    return None


def load_books(seed: int, max_books: int = 60) -> List[Unit]:
    docs = [d for d in csv.DictReader(open(RAW / "nqa_documents.csv")) if d["kind"] == "gutenberg"
            and d["set"] == "test"]
    qaps: Dict[str, List[Dict]] = {}
    for q in csv.DictReader(open(RAW / "nqa_qaps.csv")):
        qaps.setdefault(q["document_id"], []).append(q)
    rng = random.Random(seed)
    rng.shuffle(docs)
    # S: a ~100-page book first; then fill in shuffled order.
    small = next(d for d in docs if 30000 <= int(d["story_word_count"]) <= 60000)
    docs.remove(small)
    docs.insert(0, small)
    units, total = [], 0
    for d in docs:
        if total >= TIER_PAGES["L"] * PAGE_CHARS or len(units) >= max_books:
            break
        m = re.search(r"/ebooks/(\d+)", d["story_url"])
        if not m:
            continue
        text = _download(m.group(1))
        if not text:
            continue
        body = _gutenberg_body(text)
        blocks = _book_blocks(body)
        if sum(1 for b in blocks if b.startswith("###")) < 2:   # no detectable chapters: skip
            continue
        title = re.sub(r"\s*\(.*?\)\s*$", "", d["wiki_title"]).strip()
        u = Unit(d["document_id"], title, blocks, chars=len(body))
        u.questions = qaps.get(d["document_id"], [])
        units.append(u)
        total += u.chars
    _assign_tiers(units)
    return units


# ================================================================== questions

def _case(cid: str, family: str, question: str, gold: List[str], answerable: bool, reference: str,
          **tags) -> Dict:
    return {"id": cid, "document": "", "question": question, "query_type": "unknown", "gold_evidence": gold,
            "answerable": answerable, "reference_answer": reference, "tags": {"family": family, **tags}}


def contract_questions(units: List[Unit], rng: random.Random, per_tier: Dict[str, Tuple[int, int]]) -> List[Dict]:
    """CUAD questions: per tier, (answerable, unanswerable) counts drawn from that tier's own
    contracts, spread over contracts and categories."""
    out = []
    for tier in TIERS:
        want_yes, want_no = per_tier[tier]
        pool = [u for u in units if u.tier == tier]
        yes, no = [], []
        for u in pool:
            cats = list(CONTRACT_QUESTIONS)
            rng.shuffle(cats)
            for cat in cats:
                spans = [s for s in u.meta["spans"].get(cat, []) if s]
                (yes if spans else no).append((u, cat, spans))
        rng.shuffle(yes)
        rng.shuffle(no)

        def spread(items, k):  # at most 3 per contract, round-robin over contracts
            picked, per = [], {}
            for u, cat, spans in items:
                if per.get(u.key, 0) < 3:
                    picked.append((u, cat, spans))
                    per[u.key] = per.get(u.key, 0) + 1
                if len(picked) >= k:
                    break
            return picked

        for u, cat, spans in spread(yes, want_yes):
            out.append(_case(f"cuad-{tier}-{len(out)}", "contracts",
                             CONTRACT_QUESTIONS[cat].format(X=f"the {u.title}"), spans, True, spans[0],
                             kind="cuad", category=cat, unit=u.title, unit_tier=tier))
        for u, cat, _ in spread(no, want_no):
            out.append(_case(f"cuad-{tier}-{len(out)}", "contracts",
                             CONTRACT_QUESTIONS[cat].format(X=f"the {u.title}"), [], False, "",
                             kind="cuad_absent", category=cat, unit=u.title, unit_tier=tier))
    return out


def book_questions(units: List[Unit], rng: random.Random, per_book: Dict[str, int]) -> List[Dict]:
    out = []
    for u in units:
        qs = list(u.questions)
        rng.shuffle(qs)
        for q in qs[: per_book[u.tier]]:
            question = q["question"].strip()
            question = f"In {u.title}, {question[0].lower() + question[1:]}" if question else question
            out.append(_case(f"nqa-{u.tier}-{len(out)}", "books", question, [], True, q["answer1"],
                             kind="narrativeqa", unit=u.title, unit_tier=u.tier,
                             answers=[q["answer1"], q["answer2"]]))
    return out


def plant_needles(units: List[Unit], family: str, rng: random.Random, per_tier: int, two_hop: int,
                  absent: int) -> List[Dict]:
    """Plant facts into units of each tier region; return their questions."""
    singles, pair = (CONTRACT_NEEDLES, CONTRACT_TWO_HOP) if family == "contracts" else (BOOK_NEEDLES, BOOK_TWO_HOP)
    name = (lambda u: f"the {u.title}") if family == "contracts" else (lambda u: u.title)
    used: set = set()
    out = []
    for tier in TIERS:
        pool = [u for u in units if u.tier == tier]
        if len(pool) < 1:
            continue
        for k in range(per_tier):
            kind, sentence, question, answer = singles[k % len(singles)]
            u = rng.choice(pool)
            v = _values(rng, used)
            text = _fill(sentence, v)
            pos = _insert(u, text, rng)
            out.append(_case(f"needle-{family}-{tier}-{len(out)}", family, question.format(X=name(u), **v),
                             [text], True, _fill(answer, v), kind="needle", needle=kind, unit=u.title,
                             unit_tier=tier, position_in_unit=pos))
            others = [o for o in pool if o is not u] or [u]   # distractor: same kind, other values
            d = _values(rng, used)
            _insert(rng.choice(others), _fill(sentence, d), rng)
        for k in range(two_hop):
            kind, first, second, question, answer = pair
            v = _values(rng, used)
            if family == "contracts" and len(pool) > 1:
                ua, ub = rng.sample(pool, 2)                    # the rate card is in another contract
            else:
                ua = ub = rng.choice(pool)                      # same book, distant chapters
            a, b = _fill(first, v), _fill(second, v)
            pos_a = _insert(ua, a, rng, 0.05, 0.3)
            pos_b = _insert(ub, b, rng, 0.7, 0.95)
            out.append(_case(f"needle2-{family}-{tier}-{len(out)}", family, question.format(X=name(ua), **v),
                             [a, b], True, _fill(answer, v), kind="needle_2hop", needle=kind, unit=ua.title,
                             unit_tier=tier, second_unit=ub.title, position_in_unit=pos_a))
        planted = {(c["tags"]["needle"], c["tags"]["unit"]) for c in out}
        never = CONTRACT_NEVER if family == "contracts" else BOOK_NEVER
        for k in range(absent):
            kind, sentence, question, answer = singles[(k + 2) % len(singles)]
            candidates = [u for u in pool if (kind, u.title) not in planted]
            if not candidates:            # every unit has this kind: ask about a kind planted nowhere
                kind, question = never
                candidates = pool
            u = rng.choice(candidates)
            v = _values(rng, used)
            out.append(_case(f"absent-{family}-{tier}-{len(out)}", family, question.format(X=name(u), **v),
                             [], False, "", kind="needle_absent", needle=kind, unit=u.title, unit_tier=tier))
    return out


# ===================================================================== writing

def render(units: List[Unit], family: str, tier: str) -> str:
    keep = [u for u in units if TIERS.index(u.tier) <= TIERS.index(tier)]
    if family == "contracts":
        head = f"# Contract data room ({len(keep)} contracts)"
        parts = [f"## {u.title}\n\n" + "\n\n".join(u.blocks) for u in keep]
    else:
        head = f"# Library ({len(keep)} books)"
        parts = [f"## Book: {u.title}\n\n" + "\n\n".join(u.blocks) for u in keep]
    return head + "\n\n" + "\n\n".join(parts) + "\n"


def build(seed: int = 7) -> Dict:
    OUT.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    report: Dict = {}
    families = {
        "contracts": (load_contracts(seed),
                      lambda us, rng: contract_questions(us, rng, {"S": (40, 15), "M": (40, 15), "L": (40, 15)}),
                      (6, 2, 4)),
        "books": (load_books(seed),
                  lambda us, rng: book_questions(us, rng, {"S": 40, "M": 8, "L": 2}),
                  (6, 2, 4)),
    }
    for family, (units, make_questions, (needles, two_hop, absent)) in families.items():
        rng = random.Random(seed + (1 if family == "books" else 0))
        cases = make_questions(units, rng) + plant_needles(units, family, rng, needles, two_hop, absent)
        for tier in TIERS:
            text = render(units, family, tier)
            doc = OUT / f"{family}_{tier}.md"
            doc.write_text(text, encoding="utf-8")
            rows = []
            for c in cases:
                if TIERS.index(c["tags"]["unit_tier"]) > TIERS.index(tier):
                    continue
                row = json.loads(json.dumps(c))
                row["document"] = str(Path("..") / ".." / "data" / "longbench" / doc.name)
                row["tags"]["doc_tier"] = tier
                for g in row["gold_evidence"][:1]:
                    at = text.find(g)
                    row["tags"]["depth"] = round(at / len(text), 3) if at >= 0 else None
                rows.append(row)
            (EVAL / f"{family}_{tier}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            kinds: Dict[str, int] = {}
            for r in rows:
                kinds[r["tags"]["kind"]] = kinds.get(r["tags"]["kind"], 0) + 1
            report[f"{family}_{tier}"] = {
                "units": sum(1 for u in units if TIERS.index(u.tier) <= TIERS.index(tier)),
                "chars": len(text), "pages": round(len(text) / PAGE_CHARS), "questions": len(rows), "kinds": kinds}
    (OUT / "manifest.json").write_text(json.dumps({"seed": seed, "page_chars": PAGE_CHARS, "tiers": report}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(build(args.seed), indent=2))


if __name__ == "__main__":
    main()
