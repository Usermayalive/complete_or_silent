"""Phases 9-10: evidence pack construction and grounded answer generation.

A pack holds, per evidence item:
- the exact clause text, with offsets into the source and a page
- the section path (chapter > section > clause titles)
- its Phase 8 role (why it is there, and for which item)
- the clauses it cites: labels when they are in the pack, short excerpts when they are not

Building a pack (Phase 9):
1. route the pack by its topology (``prompt.analyse``)
2. trim it to the route's token budget (``prompt.apply_budget``)
3. label the items E1..En and render the route's prompt template (``prompt.render``)

``citation_map()`` maps every label back to its node, offsets and page (Phase 10
validates citations against it).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from .document_graph import DocumentGraph
from .models import CROSS_REFERENCE
from .prompt import DEFAULT_BUDGETS, QA, BudgetReport, Topology, analyse, apply_budget, count_tokens, render
from .pruning import PRIMARY, PrunedSubgraph
from .query_planner import QueryPlan

EXCERPT_WORDS = 25
# Kept for callers that import it; the route templates in prompt.py extend it.
INSTRUCTIONS = """\
- Answer only from the evidence above.
- Cite evidence by its label and section, e.g. [E2: Definitions].
- If the evidence is insufficient, say so explicitly.
- Do not infer beyond the evidence. If evidence items conflict, report both."""


@dataclass
class EvidenceItem:
    label: str
    node_id: str
    role: str
    section_path: str
    text: str
    score: float
    start: int
    end: int
    page: Optional[int] = None
    required_by: Optional[str] = None
    cites: List[str] = field(default_factory=list)                 # node ids of items in the pack
    cite_excerpts: Dict[str, str] = field(default_factory=dict)    # cited clauses not in the pack
    truncated: bool = False

    @property
    def citation(self) -> str:
        return f"{self.section_path}, p. {self.page}" if self.page else self.section_path


@dataclass
class EvidencePack:
    question: str
    query_type: str
    items: List[EvidenceItem] = field(default_factory=list)
    route: str = QA
    topology: Optional[Topology] = None
    confidence: Optional[str] = None       # Phase 7 evidence-confidence level, when known
    budget: Optional[BudgetReport] = None

    def render_prompt(self) -> str:
        return render(self)

    def prompt_tokens(self, counter: Callable[[str], int] = count_tokens) -> int:
        return counter(self.render_prompt())

    def citation_map(self) -> Dict[str, Dict]:
        """Label -> where the evidence is in the document."""
        return {i.label: {"node_id": i.node_id, "section": i.section_path, "start": i.start, "end": i.end,
                          "page": i.page, "truncated": i.truncated} for i in self.items}

    def to_dict(self) -> dict:
        return asdict(self)


def _owner(graph: DocumentGraph, nid: str, kept: set) -> Optional[str]:
    if nid in kept:
        return nid
    return next((a.id for a in graph.ancestors(nid) if a.id in kept), None)


def _citations(graph: DocumentGraph, item: EvidenceItem, kept: set) -> None:
    """What the item cites: kept items (by node id) and excerpts of the rest."""
    node = graph.nodes[item.node_id]
    sources = [node.id] + [c.id for c in graph.children(node.id) if c.is_content]
    if node.parent_id and graph.nodes[node.parent_id].is_content:
        sources.append(node.parent_id)  # a sentence's references are recorded on its paragraph
    for src in sources:
        for nbr, _, _ in graph.neighbors(src, (CROSS_REFERENCE,), "out"):
            targets = [c for c in graph.children(nbr.id) if c.is_content] if nbr.is_section else [nbr]
            owners = [o for o in (_owner(graph, t.id, kept) for t in targets) if o and o != item.node_id]
            if owners:
                for o in owners:
                    if o not in item.cites:
                        item.cites.append(o)
            elif targets and nbr.id not in item.cite_excerpts:
                words = targets[0].text.split()
                excerpt = " ".join(words[:EXCERPT_WORDS]) + (" …" if len(words) > EXCERPT_WORDS else "")
                title = nbr.title or graph.path(nbr.id)
                item.cite_excerpts[nbr.id] = f"{title}: {excerpt}"


def build_evidence_pack(graph: DocumentGraph, plan: QueryPlan, pruned: PrunedSubgraph,
                        confidence: Optional[str] = None,
                        budgets: Optional[Dict[str, Optional[int]]] = None,
                        counter: Callable[[str], int] = count_tokens) -> EvidencePack:
    """``budgets``: evidence-token budget per route (default ``prompt.DEFAULT_BUDGETS``;
    pass ``{}`` for no budget)."""
    budgets = DEFAULT_BUDGETS if budgets is None else budgets
    # Primary evidence by score, then dependencies; ties and dependencies in document
    # order, so the pack reads naturally and renders deterministically.
    order = {nid: i for i, nid in enumerate(graph.order)}
    primary = sorted((e for e in pruned.evidence if e.role == PRIMARY), key=lambda e: (-e.score, order[e.node.id]))
    support = sorted((e for e in pruned.evidence if e.role != PRIMARY), key=lambda e: order[e.node.id])
    pack = EvidencePack(plan.query, plan.query_type, confidence=confidence)
    for ev in primary + support:
        pack.items.append(EvidenceItem(
            label="", node_id=ev.node.id, role=ev.role, section_path=graph.path(ev.node.id),
            text=ev.node.text, score=round(ev.score, 4), start=ev.node.start, end=ev.node.end,
            page=ev.node.page, required_by=ev.required_by))
    pack.topology = analyse(graph, [i.node_id for i in pack.items])
    pack.route = pack.topology.route
    pack.budget = apply_budget(pack.items, budgets.get(pack.route), counter)
    kept = {i.node_id for i in pack.items}
    for n, item in enumerate(pack.items, 1):
        item.label = f"E{n}"
        _citations(graph, item, kept)
    return pack


LLM = Callable[[str], str]


def generate_answer(pack: EvidencePack, llm: Optional[LLM] = None) -> str:
    """Call ``llm(prompt)`` if provided; otherwise return an extractive answer."""
    if not pack.items:
        return "Insufficient evidence: no relevant passages were found in the document."
    if llm is not None:
        return llm(pack.render_prompt())
    top = pack.items[0]
    lines = [f"Answer (extractive, no LLM configured): {top.text}", "", "Evidence:"]
    lines += [f"  [{i.label}] {i.citation} ({i.role})" for i in pack.items]
    return "\n".join(lines)
