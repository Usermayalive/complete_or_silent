"""U4. Canary calibration: how many chapters must be opened to find an answer with
probability >= alpha, measured on THIS document (no model calls).

1. **Canaries.** Real sentences of the document are copied and mutated: numbers change,
   capitalised names are swapped for other names from the same document. A canary reads
   like the document's own text but states a fact the document does not contain.
2. **Hidden copy.** Canaries go into a copy of the sentence index; in that copy the
   sentence each canary was made from is blanked, so the canary does not compete with
   its own twin (a real answer has none). The real index and tree are never changed.
3. **Questions.** Each canary is looked up the way a question would find it: with its
   *most common* content words only (the lowest-IDF ``QUERY_SHARE``), never the changed
   value. Real questions share few rare words with their answer ("Does the contract have
   a non-compete?" vs the clause's own wording).

   Measured (2026-11-12, lexical gate, no routing): a random half of the canary's words
   ranks it first almost always (p98 rank 2 of 55 chapters) while real CUAD and planted
   questions rank their answer far lower (p98 rank 47). The common-word 10% query was
   chosen on the 130-page room and reproduced the real ranks on the 977-page room it was
   not tuned on (p90 38 vs 41, p98 47 vs 47).
4. **Ranks.** The gate's lexical signal (each chapter's best sentence, BM25) ranks the
   chapters; the canary's chapter gets rank r (1 = first).
5. **Split-conformal budget.** With n ranks of one class, k(alpha) is the
   ceil((n + 1) * alpha)-th smallest rank. If real answers rank like canaries
   (exchangeability: the assumption P3 tests), opening the top k chapters finds the answer
   with probability >= alpha.
6. **Reported with every k:** the canary miss rate at k and its Clopper-Pearson upper bound.

Two tables, because the gate ranks in two ways:

    room    rank among every chapter of the document (a question that names no unit)
    unit    rank among the chapters of the canary's own unit only (a question that names
            a book or contract with more chapters than the gate opens outright)

A canary is planted in a random chapter of the unit its source sentence came from, as a
real answer sits inside its own contract or book.

Classes (by the sentence the canary was made from):

    qualifier   has a qualifier cue (subject to, notwithstanding, ...)
    fact        has a number, amount, date or code
    clause      everything else
    type        (optional) a real clause of a known type from OUTSIDE this document,
                looked up with the words of a question about that type ("Does X restrict
                either party from competing?"). Fact-style canaries do not simulate
                clause-type questions, which share almost no words with their answer;
                type canaries do (measured 2026-11-14: on the 977-page room the real CUAD
                miss rate at alpha 0.90 / 0.95 / 0.98 was 25% / 17% / 4% with fact canaries,
                11% / 1.8% / 1.8% with type canaries). ``type_library`` supplies them:
                (question words, clause text) pairs, e.g. ``cuad_type_library``.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .qualifiers import CUES


def words(text):  # imported lazily: workers imports certify.plan
    from ..workers.routing import words as _words

    return _words(text)

ALPHAS = (0.90, 0.95, 0.98)
QUERY_SHARE = 0.1          # share of a canary's content words (its most common ones) used to look it up
MIN_WORDS = 6              # shorter sentences make poor canaries
BATCH = 50                 # canaries planted together in one hidden copy
MIN_CLASS_N = 30           # a class with fewer canaries uses the "all" table
CLASSES = ("fact", "clause", "qualifier")

_NUMBER = re.compile(r"\d[\d,.]*")
_NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")


@dataclass
class Canary:
    text: str
    source: str            # sentence id it was made from
    chapter: str           # chapter it was planted in
    cls: str
    query: List[str]
    unit: List[str] = field(default_factory=list)  # chapters of the unit it belongs to
    rank: int = 0          # rank of its chapter among every chapter (1 = first)
    rank_in_unit: int = 0  # rank among its unit's chapters
    twins: List[str] = field(default_factory=list)  # sentence ids of the document blanked in the hidden copy
    rel_score: float = 0.0  # its chapter's best score / the top chapter's score (U4.5)


@dataclass
class ClassCalibration:
    n: int
    k: Dict[str, int]                    # str(alpha) -> chapters to open
    miss_at_k: Dict[str, float]          # canary miss rate at that k
    miss_upper: Dict[str, float]         # Clopper-Pearson 95% upper bound
    ranks: List[int] = field(default_factory=list)
    threshold: Dict[str, float] = field(default_factory=dict)   # str(alpha) -> relative-score threshold (U4.5)


def _pick(classes: Dict[str, "ClassCalibration"], cls) -> "ClassCalibration":
    """``cls``: a class name or a preference tuple ("type", "clause"); falls back to "all"."""
    for c in (cls if isinstance(cls, (tuple, list)) else (cls,)):
        if c in classes:
            return classes[c]
    return classes["all"]


@dataclass
class Calibration:
    chapters: int
    tables: Dict[str, Dict[str, ClassCalibration]]   # "room" | "unit" -> class -> calibration
    seed: int
    query_share: float = QUERY_SHARE

    def k(self, alpha: float, cls="all", table: str = "room") -> int:
        return _pick(self.tables[table], cls).k[str(alpha)]

    def stats(self, cls="all", table: str = "room") -> "ClassCalibration":
        return _pick(self.tables[table], cls)

    def alphas(self) -> List[float]:
        return sorted(float(a) for a in self.tables["room"]["all"].k)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Calibration":
        tables = {t: {c: ClassCalibration(**v) for c, v in classes.items()} for t, classes in data["tables"].items()}
        return cls(data["chapters"], tables, data["seed"], data.get("query_share", QUERY_SHARE))

    def save(self, path: str) -> None:
        import json
        from pathlib import Path

        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str) -> "Calibration":
        import json

        with open(path) as f:
            return cls.from_dict(json.load(f))


# -------------------------------------------------------------- statistics

def conformal_k(ranks: Sequence[int], alpha: float, chapters: int) -> int:
    """Smallest k with P(rank <= k) >= alpha under exchangeability (split conformal).
    Too few canaries for this alpha: every chapter (no saving, but still honest)."""
    order = sorted(ranks)
    q = math.ceil((len(order) + 1) * alpha)
    return order[q - 1] if 0 < q <= len(order) else chapters


def clopper_pearson_upper(misses: int, n: int, confidence: float = 0.95) -> float:
    """One-sided exact (Clopper-Pearson) upper bound on a rate from ``misses`` out of ``n``:
    the ``confidence`` quantile of Beta(misses + 1, n - misses). (A direct binomial sum
    overflowed for n above about 1,000.)"""
    if n == 0 or misses >= n:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(confidence, misses + 1, n - misses))


# ----------------------------------------------------------------- canaries

def classify(sentence: str) -> str:
    if any(p.search(sentence) for _, p in CUES):
        return "qualifier"
    if _NUMBER.search(sentence):
        return "fact"
    return "clause"


def mutate(sentence: str, names: List[str], rng: random.Random) -> Optional[str]:
    """Change every number and up to two names; None if nothing could change."""
    changed = [False]

    def num(m):
        changed[0] = True
        digits = m.group(0)
        return str(rng.randint(10 ** (max(1, sum(c.isdigit() for c in digits)) - 1),
                               10 ** max(1, sum(c.isdigit() for c in digits)) - 1))

    out = _NUMBER.sub(num, sentence)
    found = list(dict.fromkeys(_NAME.findall(out)))[1:]   # keep the sentence's first word
    rng.shuffle(found)
    for name in found[:2]:
        options = [n for n in names if n != name]
        if options:
            out = re.sub(rf"\b{re.escape(name)}\b", rng.choice(options), out)
            changed[0] = True
    return out if changed[0] and out != sentence else None


def unit_map(tree) -> Dict[str, List[str]]:
    """Chapter id -> the chapters of its unit (a book's part, or a contract with its
    slices and schedules)."""
    from .qualifiers import units

    body = [c.id for c in tree.chapters() if c.role != "contents"]
    if tree.parts():
        groups = {p.id: list(p.children) for p in tree.parts()}
        loose = [c.id for c in tree.chapters() if not c.part]
        if loose:
            groups["_loose"] = loose
    else:
        groups = units(tree)
        if tree.kind == "book" or all(len(v) == 1 for v in groups.values()):
            groups = {"_all": body}     # one book: the whole document is the unit
    return {ch: chs for chs in groups.values() for ch in chs}


def make_canaries(tree, n: int, rng: random.Random, index=None) -> List[Canary]:
    sentences = [(sid, text) for p in tree.passages() for sid, text in p.sentences
                 if len(words(text)) >= MIN_WORDS]
    names = list(dict.fromkeys(m for _, t in sentences[:5000] for m in _NAME.findall(t)))
    chapters = [c.id for c in tree.chapters() if c.role != "contents"]
    units_of = unit_map(tree)
    chapter_of_passage = {p.id: ch.id for ch in tree.chapters() for p in tree.passages(ch.id)}
    if index is None:
        from ..workers.worker import SentenceIndex
        index = SentenceIndex(tree)
    idf, vocab = index.bm25.idf, index.bm25.vocab
    out: List[Canary] = []
    pool = sentences[:]
    rng.shuffle(pool)
    for sid, text in pool:
        if len(out) >= n:
            break
        canary = mutate(text, names, rng)
        if canary is None:
            continue
        content = [w for w in dict.fromkeys(words(canary)) if not any(ch.isdigit() for ch in w)]
        if len(content) < 3:
            continue
        size = max(2, round(len(content) * QUERY_SHARE))
        query = sorted(content, key=lambda w: idf[vocab[w]] if w in vocab else float("inf"))[:size]
        home = chapter_of_passage[tree.sentence(sid)[1]]
        unit = [c for c in units_of.get(home, [home]) if c in chapters] or [home]
        out.append(Canary(canary, sid, rng.choice(unit), classify(text), query, unit))
    return out


# ---------------------------------------------------------------- calibrate

def _rank_batch(index, batch: List[Canary]) -> None:
    """Plant a batch in a hidden copy (extra rows scored next to the real ones) and rank."""
    from ..index import BM25

    sources = {index.position[c.source] for c in batch if c.source in index.position}
    sources |= {index.position[t] for c in batch for t in c.twins if t in index.position}
    texts = [("" if i in sources else t) for i, t in enumerate(index.texts)] + [c.text for c in batch]
    chapter_of = list(index.chapter_of) + [c.chapter for c in batch]
    bm25 = BM25([words(t) for t in texts])
    names = list(dict.fromkeys(chapter_of))
    lookup = {c: i for i, c in enumerate(names)}
    idx = np.array([lookup[c] for c in chapter_of], dtype=np.int64)
    for j, c in enumerate(batch):
        scores = bm25.scores(c.query)
        best = np.zeros(len(names), dtype=np.float32)
        np.maximum.at(best, idx, scores)
        mine = lookup[c.chapter]
        own = best[mine]
        top = float(best.max()) if len(best) else 0.0
        c.rel_score = float(own) / top if top > 0 else 0.0
        # ties count against the canary (pessimistic): a chapter scoring as high as the
        # canary's is opened no later than it; a canary nothing matches ranks last
        c.rank = int((best >= own).sum())
        unit_idx = [lookup[u] for u in c.unit if u in lookup]
        c.rank_in_unit = int((best[unit_idx] >= own).sum()) if unit_idx else 1


def conformal_threshold(scores: Sequence[float], alpha: float) -> float:
    """Largest tau with P(score >= tau) >= alpha (lower split-conformal quantile); 0 (open
    every chapter) when there are too few canaries for this alpha."""
    order = sorted(scores)
    i = math.floor((len(order) + 1) * (1 - alpha))
    return float(order[i - 1]) if i >= 1 else 0.0


def _class_calibration(ranks: List[int], alphas: Sequence[float], chapters: int,
                       scores: Optional[Sequence[float]] = None) -> ClassCalibration:
    k, miss, upper, threshold = {}, {}, {}, {}
    for a in alphas:
        kk = conformal_k(ranks, a, chapters)
        misses = sum(r > kk for r in ranks)
        k[str(a)] = kk
        miss[str(a)] = round(misses / len(ranks), 4) if ranks else 1.0
        upper[str(a)] = round(clopper_pearson_upper(misses, len(ranks)), 4)
        if scores is not None:
            threshold[str(a)] = round(conformal_threshold(scores, a), 4)
    return ClassCalibration(len(ranks), k, miss, upper, sorted(ranks), threshold)


def _twins(index, text: str) -> List[str]:
    """Sentences of the document that the clause contains (it was copied from them)."""
    flat = re.sub(r"\s+", " ", text).lower()
    scores = index.bm25.scores(words(text))
    out = []
    for r in np.argsort(-scores)[:30]:
        if scores[r] <= 0:
            break
        t = re.sub(r"\s+", " ", index.texts[r]).lower().strip()
        if len(t) >= 25 and t in flat:
            out.append(index.ids[r])
    return out


def type_canaries(tree, library: Sequence, n: int, rng: random.Random, index=None) -> List[Canary]:
    """Real clauses of known types, each planted in a random chapter of a random unit and
    looked up with its type's question words. A clause the document itself contains (a
    library drawn from inside the room) has its twin sentences blanked in the hidden
    copy, so it ranks as a single real answer would."""
    chapters = [c.id for c in tree.chapters() if c.role != "contents"]
    units_of = unit_map(tree)
    picked = rng.sample(list(library), min(n, len(library)))
    out = []
    for qwords, text in picked:
        ch = rng.choice(chapters)
        unit = [c for c in units_of.get(ch, [ch]) if c in chapters] or [ch]
        twins = _twins(index, text) if index is not None else []
        out.append(Canary(text, "", ch, "type", list(qwords), unit, twins=twins))
    return out


def cuad_type_library(exclude_titles: Sequence[str] = (), seed: int = 7) -> List:
    """(question words, clause) pairs from the CUAD contracts whose titles are not in
    ``exclude_titles`` (the document being calibrated), for the categories the long
    benchmark asks about."""
    from ..longbench import CONTRACT_QUESTIONS, load_contracts

    skip = set(exclude_titles)
    out = []
    for u in load_contracts(seed):
        if u.title in skip:
            continue
        for cat, spans in u.meta.get("spans", {}).items():
            if cat not in CONTRACT_QUESTIONS:
                continue
            q = [w for w in words(CONTRACT_QUESTIONS[cat].format(X="the agreement")) if w != "agreement"]
            out += [(q, re.sub(r"\s+", " ", s).strip()) for s in spans if 40 < len(s) < 600]
    return out


def calibrate(tree, index=None, n: int = 300, seed: int = 0, alphas: Sequence[float] = ALPHAS,
              canaries: Optional[List[Canary]] = None, type_library: Optional[Sequence] = None) -> Calibration:
    """Per-document reading budgets. ``index`` is the workers' SentenceIndex (built if None).

    Rebuilding BM25 per batch costs O(document) each time: fine at 100-1,000 pages; at
    10,000 pages use fewer, larger batches."""
    from ..workers.worker import SentenceIndex

    rng = random.Random(seed)
    index = index or SentenceIndex(tree)
    canaries = canaries if canaries is not None else make_canaries(tree, n, rng, index)
    if type_library:
        canaries = canaries + type_canaries(tree, type_library, n, rng, index)
    for start in range(0, len(canaries), BATCH):
        _rank_batch(index, canaries[start:start + BATCH])
    chapters = len(index.chapter_names)
    tables: Dict[str, Dict[str, ClassCalibration]] = {}
    for table, rank_of in (("room", lambda c: c.rank), ("unit", lambda c: c.rank_in_unit)):
        own = [c for c in canaries if c.cls != "type"]      # "all" = the document's own canaries
        classes = {"all": _class_calibration([rank_of(c) for c in own], alphas, chapters,
                                             [c.rel_score for c in own])}
        for cls in CLASSES + ("type",):
            members = [c for c in canaries if c.cls == cls]
            if len(members) >= MIN_CLASS_N:
                classes[cls] = _class_calibration([rank_of(c) for c in members], alphas, chapters,
                                                  [c.rel_score for c in members])
        tables[table] = classes
    return Calibration(chapters, tables, seed)
