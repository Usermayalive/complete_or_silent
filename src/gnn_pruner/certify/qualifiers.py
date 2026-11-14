"""U2. Qualifier graph: who qualifies, overrides or amends what (no model).

Every sentence with a qualifier cue becomes a ``QualifierLink`` to its target:

    qualifies    "subject to", "except as provided in", "provided that", "unless", "shall not apply"
    overrides    "notwithstanding", "in lieu of", "shall prevail", "takes precedence"
    amends       "is hereby amended", "deleted and replaced", "supersedes"
    precedence   "order of precedence", "in the event of any conflict"

Targets, looked for in the text right after the cue (or the whole sentence for amends and
precedence):

    section      "Section 12.3", "Clause 4(b)", "Article 5", "§ 7"
    schedule     "Schedule 3", "Exhibit A", "Annex II", "Appendix 1"
    term         a term defined in the same contract ("the Fees")
    contract     another contract of the data room, named by its title words
    foregoing    "notwithstanding the foregoing", "this Section": the sentence's own section

Targets resolve **inside the sentence's own contract** (its unit: the chapter, its slices
"(k/n)" and the schedules after it). A reference to "Section 12.3" in a data room of 510
contracts means 12.3 of *this* contract, not the first 12.3 in the room. A link crosses
into another contract only when the sentence names that contract.

A cue without a resolvable target is kept (``resolved=False``): "shall not apply to fraud"
names no section, and the full-document sweep (Q4) must still find it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..tree import CHAPTER, LIFT_TO_CHAPTER, PASSAGE, ROOT, DocTree

QUALIFIES, OVERRIDES, AMENDS, PRECEDENCE = "qualifies", "overrides", "amends", "precedence"
SECTION_T, SCHEDULE_T, TERM_T, CONTRACT_T, FOREGOING_T = "section", "schedule", "term", "contract", "foregoing"

CUES: List[Tuple[str, re.Pattern]] = [
    (PRECEDENCE, re.compile(r"\b(order of precedence|in the event of (?:any )?(?:conflict|inconsistency)"
                            r"(?: or (?:conflict|inconsistency))?)\b", re.I)),
    (AMENDS, re.compile(r"\b((?:is|are) hereby (?:amended|deleted|replaced)|deleted (?:in (?:its|their) entirety )?"
                        r"and replaced|(?:amended and )?restated in (?:its|their) entirety|supersedes?|superseded)\b", re.I)),
    (OVERRIDES, re.compile(r"\b(notwithstanding|in lieu of|shall (?:prevail|control|govern) over|"
                           r"(?:shall )?takes? precedence over|shall prevail)\b", re.I)),
    (QUALIFIES, re.compile(r"\b(subject to|except (?:as (?:otherwise )?(?:expressly )?provided|as set forth|as stated)|"
                           r"save as (?:otherwise )?provided|provided,? (?:however|further|always|that)|"
                           r"unless otherwise|(?:shall|will|does|do) not apply|"
                           r"nothing in (?:the foregoing|this (?:section|clause|article|agreement))[^.]{0,80}?shall limit)\b",
                           re.I)),
]

WINDOW = 90  # characters after the cue searched for its target
TERM_WINDOW = 30         # a defined term counts only if it starts this close after the cue
TERM_MAX_SHARE = 0.05    # a term in more than this share of its contract's sentences is a name, not a target ...
TERM_MIN_ABS = 10        # ... unless it occurs in no more than this many sentences (small contracts)
GENERIC_TERMS = {"agreement", "this agreement", "parties", "party", "the parties", "contract", "effective date"}

_SECTION_REF = re.compile(
    r"\b(?:sections?|clauses?|articles?|paragraphs?|§§?)\s*"
    r"(\d+(?:\.\d+)*(?:\s*\([a-z0-9]{1,4}\))*)", re.I)
_SCHEDULE_REF = re.compile(r"\b(schedule|exhibit|annex|appendix|attachment)\s+([A-Z0-9]{1,4}(?:\.\d+)?)\b", re.I)
_FOREGOING = re.compile(r"\b(the foregoing|this (?:section|clause|article|paragraph)|the preceding)\b", re.I)
_LABEL = re.compile(r"^\s*(?:section|clause|article|§)?\s*(\d+(?:\.\d+)*)\s*(\([a-z0-9]{1,4}\))?[\s.):]", re.I)
_DEFINES = re.compile(
    r"[\"“']([A-Z][\w\- ]{1,40}?)[\"”']\s*(?:\([^)]*\)\s*)?(?:means|shall mean|refers to|has the meaning|is defined as)\b"
    r"|\((?:the |a |an )?[\"“']([A-Z][\w\- ]{1,40}?)[\"”']\)")
_SLICE = re.compile(r"\s*\(\d+/\d+\)\s*$")
_AMENDING_TITLE = re.compile(r"^\s*(amendment|addendum|supplement|variation|side letter|novation)\b", re.I)
_STOP = {"agreement", "contract", "inc", "corp", "ltd", "llc", "of", "and", "the", "a", "an", "co", "company",
         "corporation", "limited", "between", "dated", "for", "to", "in", "on", "by", "with", "number", "no"}


@dataclass
class QualifierLink:
    sentence: str                 # sentence id
    passage: str                  # passage id
    unit: str                     # the contract (first chapter id of its unit)
    kind: str                     # qualifies | overrides | amends | precedence
    cue: str                      # the matched cue, lower case
    target_type: Optional[str]    # section | schedule | term | contract | foregoing | None
    target_label: Optional[str]   # "12.3", "Schedule 3", "fees", ...
    targets: List[str] = field(default_factory=list)  # tree node ids
    resolved: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


# ------------------------------------------------------------------- units

def base_title(title: str) -> str:
    return _SLICE.sub("", title or "").strip().lower()


def units(tree: DocTree) -> Dict[str, List[str]]:
    """Contract units: first chapter id -> chapter ids. Slices "(k/n)" of one contract and
    the schedules / exhibits after it belong to the same unit."""
    out: Dict[str, List[str]] = {}
    current: Optional[str] = None
    current_title = None
    for ch in tree.chapters():
        title = base_title(ch.title)
        lifted = bool(LIFT_TO_CHAPTER.match(ch.title or ""))
        if current is not None and (title == current_title or lifted):
            out[current].append(ch.id)
            continue
        current, current_title = ch.id, title
        out[current] = [ch.id]
    return out


def _title_words(title: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z0-9]+", base_title(title)) if w not in _STOP and len(w) > 2}


# ------------------------------------------------------------------- index

class _UnitIndex:
    """Numbers, schedules and defined terms of one contract."""

    def __init__(self, tree: DocTree, unit: str, chapters: List[str]):
        self.unit = unit
        self.chapters = chapters
        self.title = tree.node(unit).title or ""
        self.numbers: Dict[str, str] = {}      # "12.3" -> first node id labelled 12.3
        self.schedules: Dict[str, str] = {}    # "schedule 3" -> node id
        self.terms: Dict[str, str] = {}        # "fees" -> passage id defining it
        self.amends: Optional["_UnitIndex"] = None  # an amendment: the contract its title names
        self._sentences: List[str] = []
        for ch in chapters:
            for node in tree.walk(ch):
                self._index(node)
        self._drop_name_like_terms()

    def _drop_name_like_terms(self) -> None:
        """Party names and "Agreement" are defined terms too, but they appear everywhere and
        qualify nothing in particular: keep only terms used in a small share of sentences."""
        n = max(1, len(self._sentences))
        low = [t.lower() for t in self._sentences]
        keep = {}
        for term, pid in self.terms.items():
            if term in GENERIC_TERMS:
                continue
            pattern = re.compile(rf"\b{re.escape(term)}\b")
            count = sum(1 for t in low if pattern.search(t))
            if count <= max(TERM_MAX_SHARE * n, TERM_MIN_ABS):
                keep[term] = pid
        self.terms = keep

    def _index(self, node) -> None:
        heads = []
        if node.kind == PASSAGE:
            heads.append(node.text or "")
        elif not node.virtual:
            heads.append(node.title or "")
            if node.number:
                self.numbers.setdefault(str(node.number).lower().strip("."), node.id)
        for head in heads:
            m = _LABEL.match(head)
            if m:
                number = m.group(1).lower().strip(".")
                self.numbers.setdefault(number, node.id)
                if m.group(2):
                    self.numbers.setdefault(number + m.group(2).lower().replace(" ", ""), node.id)
            s = _SCHEDULE_REF.match(head.strip())
            if s:
                self.schedules.setdefault(f"{s.group(1).lower()} {s.group(2).lower()}", node.id)
        if node.kind == PASSAGE:
            for _, sentence in node.sentences:
                self._sentences.append(sentence)
                for m in _DEFINES.finditer(sentence):
                    term = (m.group(1) or m.group(2) or "").strip().lower()
                    if term and len(term.split()) <= 5:
                        self.terms.setdefault(term, node.id)

    def section(self, number: str) -> List[str]:
        number = re.sub(r"\s+", "", number.lower()).strip(".")
        while number:
            if number in self.numbers:
                return [self.numbers[number]]
            if number.endswith(")"):
                number = number[:number.rindex("(")]
            else:
                break
        prefix = [nid for num, nid in self.numbers.items() if num.startswith(number + ".")]
        return prefix[:1]


# ------------------------------------------------------------------- graph

@dataclass
class QualifierGraph:
    links: List[QualifierLink]
    unit_of: Dict[str, str]           # chapter id -> unit id
    units: Dict[str, List[str]]
    term_passages: Set[str] = field(default_factory=set)   # passages defining a kept (non-generic) term

    def by_kind(self, kind: str) -> List[QualifierLink]:
        return [l for l in self.links if l.kind == kind]

    def unresolved(self) -> List[QualifierLink]:
        return [l for l in self.links if not l.resolved]

    def stats(self) -> Dict[str, int]:
        out: Dict[str, int] = {"links": len(self.links), "resolved": sum(l.resolved for l in self.links)}
        for l in self.links:
            out[l.kind] = out.get(l.kind, 0) + 1
            if l.target_type:
                out["to_" + l.target_type] = out.get("to_" + l.target_type, 0) + 1
        return out

    def to_dict(self) -> Dict:
        return {"links": [l.to_dict() for l in self.links], "units": self.units}


def _chapter_of(tree: DocTree, node_id: str) -> str:
    node = tree.node(node_id)
    while node.parent not in (None, ROOT):
        node = tree.node(node.parent)
    return node.id


def _section_of(tree: DocTree, passage_id: str) -> str:
    """The nearest non-virtual ancestor below the chapter, else the chapter."""
    node = tree.node(tree.node(passage_id).parent)
    while node.kind != CHAPTER and node.virtual:
        node = tree.node(node.parent)
    return node.id


def _targets(tree: DocTree, idx: _UnitIndex, others: Dict[str, Tuple[set, _UnitIndex]], passage_id: str,
             window: str) -> List[Tuple[str, str, List[str]]]:
    """(target_type, label, node ids) found in the window, most specific first."""
    found: List[Tuple[str, str, List[str]]] = []
    own_index = idx
    # another contract named in the window: resolve its sections there
    words = set(re.findall(r"[a-z][a-z0-9]+", window.lower()))
    best, best_overlap = None, 0
    for unit, (title_words, other_idx) in others.items():
        if unit == idx.unit or not title_words or _AMENDING_TITLE.match(other_idx.title or ""):
            continue      # "Section 42 of the MSA, as amended by Amendment No. 1" is the MSA's 42
        overlap = len(title_words & words)
        if overlap >= 2 and overlap > best_overlap:
            best, best_overlap = other_idx, overlap
    if best is None and idx.amends is not None:
        best = idx.amends  # "Amendment No. 2 to the MSA": its "Section 12.3" is the MSA's
    if best is not None:
        found.append((CONTRACT_T, best.unit, [best.chapters[0]]))
        own_index = best
    for m in _SECTION_REF.finditer(window):
        ids = own_index.section(m.group(1))
        found.append((SECTION_T, re.sub(r"\s+", "", m.group(1)).lower(), ids))
    for m in _SCHEDULE_REF.finditer(window):
        label = f"{m.group(1).lower()} {m.group(2).lower()}"
        nid = own_index.schedules.get(label)
        found.append((SCHEDULE_T, label, [nid] if nid else []))
    if _FOREGOING.search(window):
        found.append((FOREGOING_T, "foregoing", [_section_of(tree, passage_id)]))
    head = window[:TERM_WINDOW + 40].lower()
    for term, pid in own_index.terms.items():
        m = re.search(rf"\b{re.escape(term)}\b", head)
        if pid != passage_id and m and m.start() <= TERM_WINDOW:
            found.append((TERM_T, term, [pid]))
    return found


def build_qualifier_graph(tree: DocTree) -> QualifierGraph:
    unit_map = units(tree)
    unit_of = {ch: u for u, chs in unit_map.items() for ch in chs}
    indexes = {u: _UnitIndex(tree, u, chs) for u, chs in unit_map.items()}
    others = {u: (_title_words(tree.node(u).title), indexes[u]) for u in unit_map}
    for u in unit_map:
        if _AMENDING_TITLE.match(tree.node(u).title or ""):
            indexes[u].amends = _named_contract(u, others[u][0], others)
    links: List[QualifierLink] = []
    for passage in tree.passages():
        unit = unit_of[_chapter_of(tree, passage.id)]
        idx = indexes[unit]
        for sid, sentence in passage.sentences:
            for kind, pattern in CUES:
                for m in pattern.finditer(sentence):
                    whole = kind in (AMENDS, PRECEDENCE)
                    window = sentence if whole else sentence[m.end(): m.end() + WINDOW]
                    found = _targets(tree, idx, others, passage.id, window)
                    cue = m.group(1).lower()
                    if not found:
                        links.append(QualifierLink(sid, passage.id, unit, kind, cue, None, None))
                        continue
                    for ttype, label, ids in found:
                        links.append(QualifierLink(sid, passage.id, unit, kind, cue, ttype, label,
                                                   list(ids), bool(ids)))
    term_passages = {pid for idx in indexes.values() for pid in idx.terms.values()}
    return QualifierGraph(_dedupe(links), unit_of, unit_map, term_passages)


def _named_contract(unit: str, words: set, others: Dict[str, Tuple[set, "_UnitIndex"]]) -> Optional["_UnitIndex"]:
    """The other contract sharing the most title words (at least two), if any."""
    best, best_overlap = None, 1
    for other, (title_words, idx) in others.items():
        if other == unit or idx.amends is not None or _AMENDING_TITLE.match(idx.title or ""):
            continue      # an amendment binds to the contract, never to another amendment
        overlap = len(title_words & words)
        if overlap > best_overlap:
            best, best_overlap = idx, overlap
    return best


def _dedupe(links: Iterable[QualifierLink]) -> List[QualifierLink]:
    seen, out = set(), []
    for l in links:
        key = (l.sentence, l.kind, l.target_type, l.target_label, tuple(l.targets))
        if key not in seen:
            seen.add(key)
            out.append(l)
    return out
