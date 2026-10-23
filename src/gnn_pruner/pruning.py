"""Phase 8: adaptive pruning that preserves local logic.

1. Primary evidence: content nodes in score order until the scores fall below a
   floor (share of the top score) or drop by more than ``gap`` from one node to the
   next. No fixed top-k: one strong passage stays one item, a tie of several stays
   several. Past a gap, a node still qualifies if semantic and lexical match agree
   that it is relevant (the top score can be inflated by a boost, e.g. a definition's).
   ``plan.max_evidence`` is only a safety cap. Skipped on the way:
   low-signal nodes (neither semantic nor lexical match), nodes with no provenance
   (reached only by a weak expansion link), text already covered by kept evidence,
   and redundant siblings (more than ``per_section`` from one section, unless they
   score near the top or match the question strongly on their own).
2. Strategy guarantees: both sides of a comparison, the target section of a summary,
   the counterpart of each top result in a contradiction question.
3. Dependencies of the kept evidence, whatever their score:
   - the passage a kept one continues into ("These obligations survive...")
   - definitions of defined terms a kept node uses
   - both ends of a clause reference: clauses a kept node cites, and clauses that
     cite a kept node to qualify it ("The cap in Section 4.1 does not apply...";
     the qualifier phrase must be next to the citation, so "...does not apply to
     Losses arising from a breach of Section 3" does not qualify Section 3)
   - qualifiers in the same section ("Claims under this indemnity are subject to
     Section 4.2"), and what they cite
   - both sides of a contradiction: statements about the same thing with
     different numbers ("within four hours" / "within one hour")
Every dropped node records why (``PrunedSubgraph.drop_reasons``).
"""

from __future__ import annotations

import re
import weakref
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .document_graph import DocumentGraph
from .index import tokenize
from .models import CROSS_REFERENCE, DEFINITION_LINK, SEMANTIC_SIMILARITY, SENTENCE, Node
from .query_planner import PAIRWISE, PER_SIDE, SECTION_SUMMARY, QueryPlan
from .retrieval import continues
from .scoring import ScoredNode

PRIMARY = "primary"
DEFINITION = "definition"
REFERENCE = "reference"
QUALIFIER = "qualifier"      # restricts a kept clause: exception, condition, carve-out
CONTEXT = "context"          # the passage a kept one runs on into ("These obligations...")
COUNTERPART = "counterpart"  # the other side of a potential conflict
PAIRWISE_TOP = 3

# Expansion links that make a node evidence in its own right (Phase 6 kinds).
STRONG_LINKS = ("parent_of", "paragraph_of", "definition", "reference", "continuation")
_QUALIFIES = re.compile(
    r"\b(?:subject to|except|excluding|exclusions?|does not apply|do not apply|shall not apply|"
    r"notwithstanding|provided that|in addition to|instead of)\b", re.I)
QUALIFIER_DISTANCE = 6  # words between a qualifier phrase and the citation it qualifies
_NUMBER = re.compile(r"^(?:\d+|one)$")  # tokens are number-normalised; "one" is left as a word


@dataclass
class PruneConfig:
    """Selection thresholds; chosen on the dev sets (docs/USE_CASE.md, Phase 8)."""

    floor: float = 0.45         # keep nothing below this share of the top score
    gap: float = 0.2            # stop at the first drop larger than this share of the top...
    agree: float = 0.6          # ...except for nodes whose semantic and lexical match both reach this
    gap_after: int = 1          # primaries kept before the gap rule applies
    min_signal: float = 0.2     # max(semantic, lexical) below this: low signal
    per_section: int = 2        # primaries per section, unless the node scores this close
    near_top: float = 0.9       # to the top, or its own semantic or lexical match reaches it
    max_dependencies: int = 4   # context + definitions + references + qualifiers + counterparts
    conflict_overlap: float = 0.75  # word overlap for "same statement, different number"


# Per-embedder overrides, chosen on the dev sets like the defaults. TF-IDF compresses
# semantic similarity, so its scores fall in bigger steps: at gap 0.2 it lost 3 of 22
# multi-passage dev questions the old pruner answered; at 0.25, 1. Dense models lose
# none at 0.2 and their packs grow by ~1 item at 0.25.
EMBEDDER_DEFAULTS = {"tfidf": {"gap": 0.25}}


def default_config(embedder_name: str) -> PruneConfig:
    return PruneConfig(**EMBEDDER_DEFAULTS.get(embedder_name, {}))


@dataclass
class EvidenceNode:
    node: Node
    role: str
    score: float
    required_by: Optional[str] = None


@dataclass
class PrunedSubgraph:
    evidence: List[EvidenceNode]
    context_sections: Dict[str, List[str]] = field(default_factory=dict)  # node -> ancestor ids
    dropped: List[str] = field(default_factory=list)
    drop_reasons: Dict[str, str] = field(default_factory=dict)
    stop: str = ""  # why primary selection ended

    def explain(self) -> str:
        roles: Dict[str, int] = {}
        for e in self.evidence:
            roles[e.role] = roles.get(e.role, 0) + 1
        why: Dict[str, int] = {}
        for reason in self.drop_reasons.values():
            why[reason.split(":")[0]] = why.get(reason.split(":")[0], 0) + 1
        kept = ", ".join(f"{n} {r}" for r, n in roles.items()) or "nothing"
        dropped = ", ".join(f"{n} {r}" for r, n in sorted(why.items(), key=lambda kv: -kv[1]))
        return f"kept {kept}; stopped: {self.stop or 'end of list'}; dropped {dropped or 'nothing'}"


def _near_duplicate(a: str, b: str, threshold: float = 0.9) -> bool:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return a.strip() == b.strip()
    return len(ta & tb) / len(ta | tb) >= threshold


def _has_provenance(s: ScoredNode) -> bool:
    """Found by a retrieval source, or reached by a link that makes it needed."""
    return any(not r.startswith("expansion:") or r.split(":")[1] in STRONG_LINKS for r in s.reasons)


def prune(graph: DocumentGraph, scored: List[ScoredNode], plan: QueryPlan,
          config: Optional[PruneConfig] = None) -> PrunedSubgraph:
    config = config or PruneConfig()
    content = [s for s in scored if s.node.is_content]
    if not content:
        return PrunedSubgraph(evidence=[])
    top = content[0].score if content[0].score > 0 else 1.0
    score_of = {s.node.id: s.score for s in scored}
    out = PrunedSubgraph(evidence=[])
    kept, kept_ids = out.evidence, set()
    per_section: Dict[str, int] = {}

    def covered(node: Node) -> bool:
        # A sentence / list item is redundant if its paragraph is kept.
        if node.parent_id in kept_ids and graph.nodes[node.parent_id].is_content:
            return True
        return any(_near_duplicate(node.text, e.node.text) for e in kept)

    def keep(node: Node, role: str, score: float, required_by: Optional[str] = None) -> None:
        kept.append(EvidenceNode(node, role, score, required_by))
        kept_ids.add(node.id)
        out.drop_reasons.pop(node.id, None)

    def drop(nid: str, reason: str) -> None:
        out.drop_reasons.setdefault(nid, reason)

    # 1. Primary evidence.
    previous, gapped = 1.0, ""
    for s in content:
        share = s.score / top
        if out.stop:
            drop(s.node.id, out.stop)
            continue
        # Widening and covered text are the same evidence again: they never end the
        # selection and do not count as a step for the gap rule.
        kept_children = [e for e in kept if e.node.parent_id == s.node.id]
        if kept_children and share >= config.floor:
            # The paragraph qualifies on its own score: widen its kept sentence /
            # list items to the whole paragraph rather than dropping the paragraph.
            slot = kept.index(kept_children[0])
            kept[slot] = EvidenceNode(s.node, PRIMARY, max(kept_children[0].score, s.score))
            for child in kept_children[1:]:
                kept.remove(child)
            kept_ids.difference_update(c.node.id for c in kept_children)
            kept_ids.add(s.node.id)
            for child in kept_children:
                drop(child.node.id, "widened: paragraph kept")
            continue
        if kept and covered(s.node):
            drop(s.node.id, "redundant: covered by kept text")
            continue
        if primaries(kept) and share < config.floor:
            out.stop = gapped or f"floor: {share:.2f} of top < {config.floor}"
        elif len(primaries(kept)) >= plan.max_evidence:
            out.stop = f"budget: {plan.max_evidence} items"
        elif len(primaries(kept)) >= config.gap_after and not gapped and previous - share > config.gap:
            gapped = f"gap: {previous:.2f} -> {share:.2f} of top"
        if out.stop:
            drop(s.node.id, out.stop)
            continue
        if gapped and min(s.features.get("semantic", 0.0), s.features.get("lexical", 0.0)) < config.agree:
            drop(s.node.id, gapped)
            continue
        previous = share
        strong = max(s.features.get("semantic", 0.0), s.features.get("lexical", 0.0))
        if strong < config.min_signal and not s.features.get("definition") \
                and not s.features.get("target_section"):
            drop(s.node.id, "low signal")
        elif not _has_provenance(s):
            drop(s.node.id, "no provenance")
        else:
            section = graph.section_of(s.node.id)
            key = section.id if section else ""
            if per_section.get(key, 0) >= config.per_section and share < config.near_top \
                    and strong < config.near_top and plan.strategy not in (SECTION_SUMMARY, PER_SIDE):
                drop(s.node.id, "redundant sibling")
                continue
            per_section[key] = per_section.get(key, 0) + 1
            keep(s.node, PRIMARY, s.score)

    # 2. Strategy guarantees.
    _apply_strategy_guarantees(graph, plan, content, kept, kept_ids, score_of, covered, keep)

    # 3. Dependencies, strongest evidence first, depth first: what a qualifier or a
    # cited clause needs comes right after it ("subject to Section 4.2" -> 4.2 ->
    # the cap 4.2 carves out of), before the next piece of evidence's dependencies.
    added = 0
    queue: List[EvidenceNode] = list(kept)
    while queue and added < config.max_dependencies:
        ev = queue.pop(0)
        if ev.role not in (PRIMARY, QUALIFIER, REFERENCE, COUNTERPART):
            continue
        for target, role in dependencies(graph, ev.node, config):
            if added >= config.max_dependencies:
                break
            if target.id in kept_ids or covered(target):
                continue
            keep(target, role, score_of.get(target.id, 0.0), required_by=ev.node.id)
            added += 1
            if role in (QUALIFIER, REFERENCE):
                queue.insert(0, kept[-1])

    for nid in score_of:
        if nid not in kept_ids and graph.nodes[nid].is_content:
            drop(nid, "not selected")
    out.dropped = [s.node.id for s in content if s.node.id not in kept_ids]
    out.context_sections = {e.node.id: [a.id for a in graph.ancestors(e.node.id)] for e in kept}
    return out


def primaries(kept: List[EvidenceNode]) -> List[EvidenceNode]:
    return [e for e in kept if e.role == PRIMARY]


def dependencies(graph: DocumentGraph, node: Node, config: PruneConfig) -> List[Tuple[Node, str]]:
    """What ``node`` needs to be read correctly, most necessary first."""
    source = graph.parent(node.id) if node.node_type == SENTENCE and node.parent_id else node
    found: List[Tuple[Node, str]] = []
    # The passage it runs on into.
    following = _next_content(graph, source)
    if following is not None and continues(source, following):
        found.append((following, CONTEXT))
    # Defined terms it uses.
    for nbr, _, _ in graph.neighbors(source.id, (DEFINITION_LINK,), "out"):
        if nbr.is_content:
            found.append((nbr, DEFINITION))
    # Clauses it cites; a cited section contributes its opening.
    for nbr, _, _ in graph.neighbors(source.id, (CROSS_REFERENCE,), "out"):
        target = nbr if nbr.is_content else next((c for c in graph.children(nbr.id) if c.is_content), None)
        if target is not None:
            found.append((target, REFERENCE))
    # The other end: clauses that cite it (or its section) to qualify it.
    cited = [source] + [a for a in graph.ancestors(source.id) if a.is_section][:1]
    for target in cited:
        number = target.metadata.get("number")
        for nbr, _, _ in graph.neighbors(target.id, (CROSS_REFERENCE,), "in"):
            if nbr.is_content and nbr.id != source.id and number and _qualifies(nbr.text, number):
                found.append((nbr, QUALIFIER))
    # Qualifiers in the same section.
    parent = graph.parent(source.id)
    if parent is not None and parent.is_section:
        for sibling in graph.children(parent.id):
            if sibling.is_content and sibling.id != source.id and _QUALIFIES.search(sibling.text):
                found.append((sibling, QUALIFIER))
    # Both sides of a contradiction.
    for partner in conflicts(graph, node, config.conflict_overlap):
        found.append((partner, COUNTERPART))
    return found


def _next_content(graph: DocumentGraph, node: Node) -> Optional[Node]:
    siblings = [c for c in graph.children(node.parent_id)] if node.parent_id else []
    ids = [c.id for c in siblings]
    if node.id in ids and ids.index(node.id) + 1 < len(ids):
        following = siblings[ids.index(node.id) + 1]
        return following if following.is_content else None
    return None


def _qualifies(text: str, number: str) -> bool:
    """A qualifier phrase within ``QUALIFIER_DISTANCE`` words of a citation of ``number``."""
    citations = [m.start() for m in re.finditer(
        rf"\b(?:sections?|clauses?|articles?|paragraphs?|§§?)\s*{re.escape(number)}(?![\d.]*\d)", text, re.I)]
    phrases = [(m.start(), m.end()) for m in _QUALIFIES.finditer(text)]
    for cite in citations:
        for start, end in phrases:
            between = text[min(cite, end):max(cite, start)]
            if len(between.split()) <= QUALIFIER_DISTANCE:
                return True
    return False


def _units(graph: DocumentGraph, node: Node) -> List[Node]:
    """Statement-sized pieces of a node: its sentences, or the node itself."""
    sentences = [c for c in graph.children(node.id) if c.node_type == SENTENCE]
    return sentences or [node]


@lru_cache(maxsize=65536)
def _statement(text: str) -> Tuple[frozenset, frozenset]:
    """(words, numbers) of a statement."""
    tokens = tokenize(text)
    return (frozenset(t for t in tokens if not _NUMBER.match(t)),
            frozenset(t for t in tokens if _NUMBER.match(t)))


class _ConflictIndex:
    """Numeric statement units of a graph with an inverted word index. Built once per
    graph (G6 found the linear scan cost ~4M comparisons per question at 10,000 pages)."""

    def __init__(self, graph: DocumentGraph):
        self.units: List[Tuple[Node, frozenset, frozenset]] = []
        for other in graph.nodes.values():
            if not other.is_content:
                continue
            kids = graph.children(other.id)
            if kids and any(c.node_type == SENTENCE for c in kids):
                continue  # compare statement-sized units only
            words, numbers = _statement(other.text)
            if numbers:
                self.units.append((other, words, numbers))
        self.postings: Dict[str, List[int]] = {}
        for i, (_, words, _) in enumerate(self.units):
            for w in words:
                self.postings.setdefault(w, []).append(i)

    def candidates(self, words: frozenset, overlap: float) -> List[int]:
        """Units that can reach ``overlap`` Jaccard with ``words``. Jaccard >= t implies
        the other unit holds at least t·|words| of them, so it must hold one of any
        floor((1 - t)·|words|) + 1 of them; the rarest are used (exact, no misses)."""
        need = int((1 - overlap) * len(words)) + 1
        rare = sorted(words, key=lambda w: len(self.postings.get(w, ())))[:need]
        found: Set[int] = set()
        for w in rare:
            found.update(self.postings.get(w, ()))
        return sorted(found)


_CONFLICT_INDEX: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def conflicts(graph: DocumentGraph, node: Node, overlap: float) -> List[Node]:
    """Statements elsewhere that say the same thing as ``node`` with different
    numbers ("respond within four hours" vs "within one hour")."""
    mine = [_statement(u.text) for u in _units(graph, node)]
    mine = [(words, numbers) for words, numbers in mine if numbers and len(words) >= 3]
    if not mine:
        return []
    index = _CONFLICT_INDEX.get(graph)
    if index is None:
        index = _CONFLICT_INDEX[graph] = _ConflictIndex(graph)
    own = {node.id, node.parent_id} | {c.id for c in graph.children(node.id)}
    hits: Set[int] = set()
    for my_words, my_numbers in mine:
        for i in index.candidates(my_words, overlap):
            other, words, numbers = index.units[i]
            union = my_words | words
            if other.id not in own and numbers != my_numbers and union \
                    and len(my_words & words) / len(union) >= overlap:
                hits.add(i)
    return [index.units[i][0] for i in sorted(hits)]


def _within(graph: DocumentGraph, node: Node, targets: Set[str]) -> bool:
    return node.id in targets or any(a.id in targets for a in graph.ancestors(node.id))


def _apply_strategy_guarantees(graph, plan, content, kept, kept_ids, score_of, covered, keep) -> None:
    """Evidence a strategy needs even if it falls below the score cutoff."""
    if plan.strategy == PER_SIDE:
        # A comparison needs both sides: keep each side's best passage.
        for side in plan.sides:
            targets = set(side.section_ids)
            if not targets or any(_within(graph, e.node, targets) for e in kept):
                continue
            best = next((s for s in content if _within(graph, s.node, targets)
                         and not covered(s.node)), None)
            if best is not None:
                keep(best.node, PRIMARY, best.score, f"side:{side.text}")
    elif plan.strategy == SECTION_SUMMARY:
        # A summary covers its section: every passage of the target sections that
        # reached the subgraph, best first, within the evidence budget.
        targets = set(plan.target_sections)
        for s in content:
            if len(primaries(kept)) >= plan.max_evidence:
                break
            if _within(graph, s.node, targets) and s.node.id not in kept_ids and not covered(s.node):
                keep(s.node, PRIMARY, s.score, "summary:section")
    elif plan.strategy == PAIRWISE:
        # A conflict needs both statements: pair each top result with its most
        # similar passage elsewhere in the document.
        for ev in list(kept)[:PAIRWISE_TOP]:
            source = ev.node.parent_id if graph.nodes[ev.node.id].parent_id and \
                graph.nodes[ev.node.parent_id].is_content else ev.node.id
            partners = sorted(graph.neighbors(source, (SEMANTIC_SIMILARITY,)), key=lambda t: -t[2])
            for partner, _, _ in partners[:1]:
                if partner.is_content and partner.id not in kept_ids and not covered(partner):
                    keep(partner, COUNTERPART, score_of.get(partner.id, 0.0), ev.node.id)
