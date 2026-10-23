"""Phase 4: turn a question into a retrieval plan.

The planner classifies the question, then, given the document graph, resolves
what the question points at:

- defined terms ("What's a Subscription Term?" -> the "subscription term" definition)
- cited sections and clauses ("Section 2(c)", "sections 4.1 and 4.2")
- sections named by topic ("summarize the liability terms" -> "4. Liability")
- comparison sides ("renewal vs termination" -> 9.1 Renewal | 10. Termination)
- the condition in a what-if question ("if availability drops below 99.9%")

and picks a retrieval strategy. Every decision is recorded in ``plan.notes`` so
the plan can be inspected (``plan.explain()``, CLI ``--plan``).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

from .textutil import stem as _stem

if TYPE_CHECKING:
    from .document_graph import DocumentGraph
    from .index import GraphIndex

FACT = "fact_lookup"
DEFINITION = "definition_lookup"
COMPARISON = "clause_comparison"
SUMMARY = "summary"
CROSS_REFERENCE = "cross_reference_reasoning"
CONTRADICTION = "contradiction_detection"
QUERY_TYPES = (FACT, DEFINITION, COMPARISON, SUMMARY, CROSS_REFERENCE, CONTRADICTION)

# Retrieval strategies (how retrieval / expansion / pruning behave).
HYBRID = "hybrid"                        # lexical + dense + structural candidates
SCOPED = "scoped"                        # the question cites sections: favour them
DEFINITION_FIRST = "definition_first"    # the definition, then where the term is used
FOLLOW_REFERENCES = "follow_references"  # condition -> consequences, 2-hop references
PER_SIDE = "per_side"                    # retrieve each side; keep evidence for both
SECTION_SUMMARY = "section_summary"      # the target section's content, in order
PAIRWISE = "pairwise"                    # similar statements in pairs, to expose conflicts

DEFAULT_WEIGHTS: Dict[str, float] = {
    "semantic": 0.35,
    "lexical": 0.25,
    "parent_context": 0.10,
    "section_importance": 0.10,
    "cross_reference": 0.05,
    "type_value": 0.10,
    "definition": 0.0,
    "target_section": 0.0,
    # Phase 7, chosen on eval/calibration_v1.jsonl (dev): the Phase 5 fused rank
    # breaks ties that one similarity alone gets wrong. Centrality and path coherence
    # did not help there (0 = computed and shown in --explain, not used).
    "fusion": 0.3,
    "centrality": 0.0,
    "path_coherence": 0.0,
    "noise": 0.15,
}

_REF = re.compile(
    r"\b(?:sections?|clauses?|articles?|paragraphs?|§§?)\s*"
    r"(\d+(?:\.\d+)*(?:\([a-z0-9]{1,3}\))*"
    r"(?:\s*(?:,|and|or|&)\s*\d+(?:\.\d+)*(?:\([a-z0-9]{1,3}\))*)*)", re.I)
_REF_NUMBER = re.compile(r"\d+(?:\.\d+)*(?:\([a-z0-9]{1,3}\))*", re.I)

# --- classification rules, checked in this order ------------------------------
_CONTRADICTION = re.compile(
    r"\b(contradict\w*|inconsisten\w*|consistent|conflict\w*|clash\w*|discrepan\w*|"
    r"at odds|incompatible)\b", re.I)
_DEFINITION_PATTERNS = [
    re.compile(r"^(.+?)\s*[-—:]\s*what does (?:that|this) (?:term|word|phrase) mean", re.I),
    re.compile(r"definition of (?:the term )?(.+)", re.I),
    re.compile(r"\bmeant? by (.+)", re.I),
    re.compile(r"meaning of (?:the term )?(.+)", re.I),
    re.compile(r"^define (.+)", re.I),
    re.compile(r"how (?:is|are) (.+?) defined", re.I),
    re.compile(r"what counts as (?:an? |the )?(.+)", re.I),
    re.compile(r"what (?:does|do) (.+?) mean", re.I),
]
# "What is X?" is a definition question only if X is a defined term (or quoted).
_WHAT_IS = re.compile(r"^what(?:'s| is| are)\s+(?:an?\s+|the\s+)?(.+?)\s*\??$", re.I)
_QUOTED = re.compile(r"[\"'‘“](.+?)[\"'’”]")
_COMPARISON = re.compile(
    r"\b(compare|comparison|difference between|differ\w*|versus|vs\.?|contrast|"
    r"which is (?:longer|shorter|higher|lower|greater|larger|smaller|stricter|better|worse)|"
    r"how does .+ relate to)(?=\W|$)", re.I)
_SUMMARY = re.compile(
    r"\b(summari[sz]e|summary|overview|outline|key points|main points|describe|"
    r"tl;?dr|gist|recap|walk me through|high[- ]level)\b", re.I)
_CROSS_REFERENCE = re.compile(
    r"(\bwhat happens (?:if|when)\b|\bwhat if\b|\bin the event\b|\bconsequences?\b|"
    r"\bsuppose\b|\bassuming\b|\brefer(?:s|red|ring)? to\b|"
    r"\bif\b.+\b(?:then|what|can|is|does|do|will)\b|\bif\b[^?]+,)", re.I)

# --- extraction ----------------------------------------------------------------
_SIDE_PATTERNS = [
    re.compile(r"difference between (?:the )?(.+?) and (?:the )?(.+?)\s*[?.!]?$", re.I),
    re.compile(r"\b(?:compare|contrast) (?:the )?(.+?) (?:with|to|against|and) (?:the )?(.+?)\s*[?.!]?$", re.I),
    re.compile(r"^(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\s*[:?,]|$)", re.I),
    re.compile(r"which is \w+:? (?:the )?(.+?) or (?:the )?(.+?)\s*[?.!]?$", re.I),
    re.compile(r"how (?:do|does) (?:the )?(.+?) and (?:the )?(.+?) differ", re.I),
]
_CONDITION_PATTERNS = [
    re.compile(r"\bin the event (?:of|that) (.+?)(?:,|\?|$)", re.I),
    re.compile(r"\bsuppose (?:that )?(.+?)(?:\s+[-—]\s+|,|\?|$)", re.I),
    re.compile(r"\bassuming (?:that )?(.+?)(?:,|\?|$)", re.I),
    re.compile(r"\bconsequences? of (.+?)(?:\?|$)", re.I),
    re.compile(r"\bif (.+?)(?:,|\?|$)", re.I),
]
_STOP = frozenset(
    "a an the of to in on for and or with by is are was be it its this that these those what "
    "which who when where why how does do did can could should would will shall may must "
    "any all our we us they them their there about from as at than then so if section sections "
    "clause clauses provision provisions terms term agreement contract say says mean".split())
_TOKEN = re.compile(r"[a-z][a-z\-]+")
TITLE_MATCH_THRESHOLD = 0.5


@dataclass
class Side:
    text: str
    section_ids: List[str] = field(default_factory=list)


@dataclass
class QueryPlan:
    query: str
    query_type: str
    strategy: str = HYBRID
    target_types: Tuple[str, ...] = ("paragraph", "clause", "sentence", "table", "figure")
    focus_terms: List[str] = field(default_factory=list)
    section_refs: List[str] = field(default_factory=list)       # as written: "4.2", "2(c)"
    target_sections: List[str] = field(default_factory=list)    # resolved node ids
    sides: List[Side] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)        # extra retrieval queries
    expansion_radius: int = 1
    expand_same_section: bool = False  # pull in the whole section, not just neighbours
    max_candidates: int = 30
    max_expanded: int = 10             # content nodes expansion may add (Phase 6)
    max_evidence: int = 6
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def explain(self, graph: Optional["DocumentGraph"] = None) -> str:
        def name(nid: str) -> str:
            if graph is None or nid not in graph.nodes:
                return nid
            node = graph.nodes[nid]
            return graph.path(nid) if node.is_section else f"{node.metadata.get('number') or nid}"

        lines = [f"type: {self.query_type}   strategy: {self.strategy}"]
        if self.focus_terms:
            lines.append(f"focus terms: {', '.join(self.focus_terms)}")
        if self.target_sections:
            lines.append(f"target sections: {' | '.join(name(s) for s in self.target_sections)}")
        for i, side in enumerate(self.sides, 1):
            where = ", ".join(name(s) for s in side.section_ids) or "unresolved"
            lines.append(f"side {i}: {side.text!r} -> {where}")
        if self.sub_queries:
            lines.append(f"sub-queries: {self.sub_queries}")
        lines.append(f"node types: {', '.join(self.target_types)}; expansion radius "
                     f"{self.expansion_radius}{' + same section' if self.expand_same_section else ''}; "
                     f"max candidates {self.max_candidates}; max expanded {self.max_expanded}; "
                     f"max evidence {self.max_evidence}")
        lines += [f"- {note}" for note in self.notes]
        return "\n".join(lines)


# ------------------------------------------------------------------ helpers

def _clean_term(term: str) -> str:
    term = term.strip().strip("?.!\"'‘’“” ")
    term = re.sub(r"^(?:the|a|an)\s+", "", term, flags=re.I)
    term = re.sub(r"\s+(?:in|under) (?:this|the) (?:agreement|contract)$", "", term, flags=re.I)
    return term.strip("\"'‘’“” ").lower()


def _tokens_match(a: str, b: str) -> bool:
    a, b = _stem(a), _stem(b)
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return (len(short) >= 5 and long_.startswith(short)) or (len(short) >= 6 and short[:6] == long_[:6])


def _content_tokens(text: str) -> List[str]:
    # "non-renewal" contributes "non-renewal", "non" and "renewal".
    tokens = []
    for token in _TOKEN.findall(text.lower()):
        tokens += [token] + (token.split("-") if "-" in token else [])
    return [t for t in tokens if t and t not in _STOP]


def _title_tokens(title: str) -> List[str]:
    return [t for t in _content_tokens(re.sub(r"^[\d.()a-z]*\d[\d.()]*\s*", "", title, flags=re.I))]


def _same_term(a: str, b: str) -> bool:
    return [_stem(t) for t in a.split()] == [_stem(t) for t in b.split()]


def resolve_term(term: str, graph: Optional["DocumentGraph"]) -> Optional[str]:
    """Map a phrase to a defined term in ``graph`` (case, plural and containment tolerant)."""
    if graph is None or not graph.definitions:
        return None
    term = _clean_term(term)
    if term in graph.definitions:
        return term
    for defined in graph.definitions:
        if _same_term(defined, term):
            return defined
    contained = [d for d in graph.definitions if re.search(rf"\b{re.escape(d)}\b", term)]
    return max(contained, key=len) if contained else None


def match_sections(text: str, graph: "DocumentGraph") -> List[Tuple[str, float]]:
    """Sections whose title matches ``text``; overlapping ancestors / descendants
    resolve to the better match (ties go to the more specific section)."""
    query_tokens = _content_tokens(text)
    scored = {}
    for section in graph.sections():
        title_tokens = _title_tokens(section.title or "")
        if not title_tokens or not query_tokens:
            continue
        hits = sum(1 for t in title_tokens if any(_tokens_match(t, q) for q in query_tokens))
        score = hits / len(title_tokens)
        if score >= TITLE_MATCH_THRESHOLD:
            scored[section.id] = score
    keep = dict(scored)
    for sid, score in scored.items():
        ancestors = [a.id for a in graph.ancestors(sid) if a.id in scored]
        for anc in ancestors:
            if score >= scored[anc]:
                keep.pop(anc, None)       # descendant at least as good: more specific wins
            elif sid in keep and anc in keep:
                keep.pop(sid, None)       # ancestor strictly better
    return sorted(keep.items(), key=lambda kv: (-kv[1], graph.order.index(kv[0])))


def _resolve_refs(query: str, graph: Optional["DocumentGraph"], plan: QueryPlan) -> List[str]:
    numbers = [n.lower() for m in _REF.finditer(query) for n in _REF_NUMBER.findall(m.group(1))]
    plan.section_refs = numbers
    resolved = []
    for number in numbers:
        if graph is None:
            continue
        node = graph.resolve_reference(number)
        if node is None:
            plan.notes.append(f"Section {number} is not in this document")
        else:
            resolved.append(node.id)
            if (node.metadata.get("number") or "").lower() != number:
                plan.notes.append(f"Section {number} not found; using {graph.path(node.id)}")
    return list(dict.fromkeys(resolved))


def _semantic_section(text: str, index: Optional["GraphIndex"]) -> Optional[str]:
    if index is None or not len(index.section_ids):
        return None
    sims = index.section_matrix @ index.encode_query(text)
    return index.section_ids[int(sims.argmax())]


def _classify(query: str, graph: Optional["DocumentGraph"]) -> Tuple[str, Optional[str], str]:
    """(query type, raw focus phrase, rule that fired)."""
    if _CONTRADICTION.search(query):
        return CONTRADICTION, None, "contradiction wording"
    for pattern in _DEFINITION_PATTERNS:
        match = pattern.search(query)
        if match:
            return DEFINITION, match.group(1), f"definition pattern /{pattern.pattern[:30]}…/"
    what_is = _WHAT_IS.match(query.strip())
    if what_is:
        phrase = what_is.group(1)
        if _QUOTED.search(query):
            return DEFINITION, _QUOTED.search(query).group(1), "'what is' + quoted term"
        resolved = resolve_term(phrase, graph)
        if resolved and _same_term(_clean_term(phrase), resolved):  # not "X length", "X notice"
            return DEFINITION, phrase, "'what is' + defined term"
    if _COMPARISON.search(query):
        return COMPARISON, None, "comparison wording"
    if _SUMMARY.search(query):
        return SUMMARY, None, "summary wording"
    if _CROSS_REFERENCE.search(query):
        return CROSS_REFERENCE, None, "conditional / reference wording"
    return FACT, None, "default"


# -------------------------------------------------------------------- planner

def plan_query(query: str, graph: Optional["DocumentGraph"] = None,
               index: Optional["GraphIndex"] = None,
               classifier: Optional["PrototypeClassifier"] = None) -> QueryPlan:
    """Build a plan. ``graph`` enables term / section resolution; ``index`` adds
    semantic fallbacks: for the question type when no rule fires (dense embedders
    only), and for summaries / comparison sides that name no section title."""
    query_type, focus, rule = _classify(query, graph)
    if query_type == FACT:
        classifier = classifier or _default_classifier(index)
        if classifier is not None:
            label, note = classifier.decide(query)
            if label == DEFINITION and resolve_term(query, graph) is None:
                label = None  # a definition question must name a defined term
            if label:
                query_type, rule = label, note
    plan = QueryPlan(query, query_type)
    plan.notes.append(f"type from {rule}")
    refs = _resolve_refs(query, graph, plan)

    if query_type == DEFINITION:
        term = _clean_term(focus or "")
        resolved = resolve_term(term, graph)
        if resolved is None and graph is not None and graph.definitions:
            # "that term" / "it": look for any defined term mentioned in the question.
            resolved = resolve_term(query, graph)
        if resolved:
            plan.focus_terms = [resolved]
            if resolved != term:
                plan.notes.append(f"'{term}' resolved to defined term '{resolved}'")
        else:
            plan.focus_terms = [term]
            if graph is not None:
                plan.notes.append(f"'{term}' is not a defined term in this document")
        plan.strategy = DEFINITION_FIRST
        plan.max_evidence = 4
        plan.weights["definition"] = 0.6
        plan.target_sections = refs
        return plan

    if query_type == COMPARISON:
        plan.sides = _extract_sides(query, refs, graph, index, plan)
        plan.target_sections = list(dict.fromkeys(refs + [s for side in plan.sides for s in side.section_ids]))
        plan.sub_queries = [side.text for side in plan.sides if side.text]
        plan.strategy = PER_SIDE if len(plan.sides) >= 2 else HYBRID
        plan.expand_same_section = True
        plan.max_expanded = 30
        plan.max_evidence = 8
        plan.weights["section_importance"] = 0.05
        plan.weights["target_section"] = 0.2
        return plan

    if query_type == CONTRADICTION:
        plan.target_sections = refs or ([sid for sid, _ in match_sections(query, graph)] if graph else [])
        plan.strategy = PAIRWISE
        plan.expansion_radius = 2
        plan.max_expanded = 20
        plan.max_evidence = 10
        plan.weights["target_section"] = 0.15
        return plan

    if query_type == SUMMARY:
        targets = refs or ([sid for sid, _ in match_sections(query, graph)] if graph else [])
        if not targets:
            fallback = _semantic_section(query, index)
            if fallback:
                targets = [fallback]
                plan.notes.append("no section named in the question; using the closest section by meaning")
        plan.target_sections = targets
        plan.strategy = SECTION_SUMMARY if targets else HYBRID
        plan.target_types = ("paragraph", "clause", "table")
        plan.expand_same_section = True
        plan.max_candidates = 50
        plan.max_expanded = 30
        plan.max_evidence = 10
        plan.weights["parent_context"] = 0.25
        plan.weights["target_section"] = 0.4
        return plan

    title_matches = [sid for sid, _ in match_sections(query, graph)] if graph else []
    plan.target_sections = list(dict.fromkeys(refs + title_matches))
    if query_type == CROSS_REFERENCE:
        condition = _extract_condition(query)
        if condition:
            plan.sub_queries = [condition]
            plan.notes.append(f"condition: {condition!r}")
        plan.strategy = FOLLOW_REFERENCES
        plan.expansion_radius = 2
        plan.max_expanded = 20
        plan.max_evidence = 8
        plan.weights["cross_reference"] = 0.2
        plan.weights["target_section"] = 0.15
        return plan

    plan.strategy = SCOPED if refs else HYBRID
    plan.weights["target_section"] = 0.3 if refs else 0.1
    return plan


def _extract_sides(query: str, refs: List[str], graph, index, plan: QueryPlan) -> List[Side]:
    if len(refs) >= 2:
        names = [graph.path(r) if graph else r for r in refs]
        plan.notes.append("sides from cited sections")
        return [Side(name, [ref]) for name, ref in zip(names, refs)]
    for pattern in _SIDE_PATTERNS:
        match = pattern.search(query)
        if not match:
            continue
        sides = []
        for raw in match.groups():
            text = _clean_term(raw)
            sections: List[str] = []
            number = _REF_NUMBER.fullmatch(re.sub(r"^(?:sections?|clauses?)\s*", "", text))
            if graph is not None and number and graph.resolve_reference(number.group(0)):
                sides.append(Side(text, [graph.resolve_reference(number.group(0)).id]))
                continue
            if graph is not None:
                matches = match_sections(text, graph)
                best = max((s for _, s in matches), default=0)
                sections = [sid for sid, s in matches if s == best]
                if not sections:
                    fallback = _semantic_section(text, index)
                    if fallback:
                        sections = [fallback]
                        plan.notes.append(f"side {text!r} matched by meaning, not title")
            sides.append(Side(text, sections))
        return sides
    plan.notes.append("could not split the comparison into sides")
    return []


def _extract_condition(query: str) -> Optional[str]:
    for pattern in _CONDITION_PATTERNS:
        match = pattern.search(query)
        if match:
            condition = match.group(1).strip(" -—")
            if len(condition.split()) >= 2:
                return condition
    return None


# ------------------------------------------------- semantic fallback classifier

# Example questions per type, from the development planner set (eval/planner_v1.jsonl).
# The held-out set (eval/planner_holdout_v1.jsonl) is deliberately not used here.
EXEMPLARS: Dict[str, List[str]] = {
    DEFINITION: [
        "What's a Subscription Term?", "Explain the meaning of Affiliate",
        "What counts as Confidential Information?", "How is Misconduct defined?",
        "Define Losses.", "What does the agreement mean by Customer Data?",
        "Service Levels - what does that term mean?", "What is 'Losses'?",
    ],
    FACT: [
        "When are fees invoiced?", "How much is the Standard plan per month?",
        "Is support available on weekends?", "Who can use the Platform?",
        "How long is the initial term?", "Does the Customer get a refund on termination for convenience?",
        "What notice is required for non-renewal?", "Where is customer data stored?",
        "What is the liability cap?", "When must invoices be paid?",
        "Can the Customer terminate immediately?", "What is the uptime commitment?",
    ],
    CROSS_REFERENCE: [
        "What happens if the Customer breaches Section 2(c)?", "What if availability drops below 99.9 percent?",
        "If the Supplier commits fraud, what are the consequences?",
        "What are the consequences of a personal data breach?",
        "In the event of a material breach, can the agreement be ended?",
        "Which clauses refer to Section 4.2?", "Suppose the Provider raises prices - what notice applies?",
        "Assuming the Supplier is grossly negligent, is its liability capped?",
    ],
    COMPARISON: [
        "Compare sections 4.1 and 4.2", "How do sections 4.1 and 4.2 differ?",
        "What is the difference between the Standard and Enterprise plans?",
        "Renewal vs termination: how do they differ?", "Contrast the support terms with the service levels",
        "Which is longer: the confidentiality period or the termination notice period?",
        "Compare the indemnity with the liability cap",
    ],
    SUMMARY: [
        "Summarize the liability terms", "Give me an overview of the fees",
        "What are the key points of the data protection section?", "Outline the termination provisions",
        "Briefly describe Section 9", "TL;DR of the confidentiality clause",
    ],
    CONTRADICTION: [
        "Is there a conflict in the incident response time?",
        "Are there any inconsistent provisions about support response?",
        "Do sections 5 and 7 contradict each other?",
        "Does anything clash with the 99.9 percent availability commitment?",
        "Are the incident response commitments consistent?",
    ],
}
# How much another type must beat "fact" to override the default. Chosen on the
# Phase 0 set (eval/contracts_v1.jsonl, all fact questions, none used as exemplars):
# there, fact questions reach +0.246 toward other types with MiniLM, so non-definition
# types need 0.25. Definitions are also gated on naming a defined term, so they can
# use a lower bar. Similarity to exemplars separates the other types from fact
# questions poorly; see docs/USE_CASE.md.
FALLBACK_MARGINS: Dict[str, float] = {DEFINITION: 0.15}
FALLBACK_DEFAULT_MARGIN = 0.25
FALLBACK_TOP_K = 3


class PrototypeClassifier:
    """Nearest-exemplar question classifier: mean similarity to each type's top-k
    exemplars. Used only when no rule fires, and only with a dense embedder (TF-IDF
    fitted on one contract cannot compare questions meaningfully)."""

    def __init__(self, embedder, exemplars: Dict[str, List[str]] = EXEMPLARS,
                 margins: Optional[Dict[str, float]] = None,
                 default_margin: float = FALLBACK_DEFAULT_MARGIN, top_k: int = FALLBACK_TOP_K):
        self.embedder, self.top_k = embedder, top_k
        self.margins = dict(FALLBACK_MARGINS if margins is None else margins)
        self.default_margin = default_margin
        self.labels = [label for label, texts in exemplars.items() for _ in texts]
        self.vectors = embedder.encode([t for texts in exemplars.values() for t in texts])

    def scores(self, query: str) -> Dict[str, float]:
        sims = self.vectors @ self.embedder.encode_query(query)
        out = {}
        for label in dict.fromkeys(self.labels):
            values = sorted((s for s, l in zip(sims, self.labels) if l == label), reverse=True)
            out[label] = float(sum(values[: self.top_k]) / min(self.top_k, len(values)))
        return out

    def decide(self, query: str) -> Tuple[Optional[str], str]:
        scores = self.scores(query)
        label = max(scores, key=scores.get)
        if label == FACT or scores[label] - scores[FACT] < self.margins.get(label, self.default_margin):
            return None, ""
        return label, f"semantic fallback ({label} {scores[label]:.2f} vs fact {scores[FACT]:.2f})"


_CLASSIFIERS: Dict[str, PrototypeClassifier] = {}


def _default_classifier(index: Optional["GraphIndex"]) -> Optional[PrototypeClassifier]:
    if index is None or not getattr(index.embedder, "cacheable", False):
        return None
    name = index.embedder.name
    if name not in _CLASSIFIERS:
        _CLASSIFIERS[name] = PrototypeClassifier(index.embedder)
    return _CLASSIFIERS[name]
