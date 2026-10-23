"""Phases 5-7: hybrid candidate retrieval, graph expansion, then scoring
(``scoring.py``).

Every scored node carries its feature breakdown and the reasons it entered the
subgraph, so rankings are explainable and can later become training data for a
learned ranker / GNN.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from .document_graph import DocumentGraph
from .embeddings import Embedder
from .index import GraphIndex, tokenize
from .models import (CROSS_REFERENCE, DEFINITION_LINK, NEXT_PREV, SAME_SECTION, SEMANTIC_SIMILARITY,
                     SENTENCE, Node)
from .query_planner import (DEFINITION, PAIRWISE, SECTION_SUMMARY, QueryPlan, _content_tokens,
                            _title_tokens, _tokens_match, plan_query)
from .scoring import (Calibration, EvidenceConfidence, Scorer, ScoredNode, evidence_confidence,
                      score_subgraph)

MAX_REFERENCED_PARAGRAPHS = 3


@dataclass
class QueryContext:
    """Per-query signals computed once over the whole index."""

    plan: QueryPlan
    tokens: List[str]
    query_vector: np.ndarray
    semantic: np.ndarray
    lexical: np.ndarray
    section_semantic: np.ndarray

    sub_vectors: List[np.ndarray] = field(default_factory=list)
    target_nodes: Set[str] = field(default_factory=set)  # nodes inside planned sections
    fused: Dict[str, float] = field(default_factory=dict)  # candidate -> fused rank score
    sources: Dict[str, List[str]] = field(default_factory=dict)  # method -> ranked node ids

    @classmethod
    def build(cls, index: GraphIndex, plan: QueryPlan) -> "QueryContext":
        """Signals for the question; each plan sub-query (comparison side, what-if
        condition) contributes too, so a node relevant to one side is not drowned out."""
        queries = [plan.query] + list(plan.sub_queries)
        vectors = [index.encode_query(q) for q in queries]
        semantic = np.zeros(len(index.ids), dtype=np.float32)
        section_semantic = np.zeros(len(index.section_ids), dtype=np.float32)
        lexical = np.zeros(len(index.ids), dtype=np.float32)
        for query, vector in zip(queries, vectors):
            if len(index.ids):
                semantic = np.maximum(semantic, _scale(index.matrix @ vector, index.semantic_scaling))
                bm25 = index.bm25.scores(tokenize(query))
                if bm25.max() > 0:
                    lexical = np.maximum(lexical, bm25 / bm25.max())
            if len(index.section_ids):
                section_semantic = np.maximum(
                    section_semantic, _scale(index.section_matrix @ vector, index.semantic_scaling))
        tokens = list(dict.fromkeys(t for q in queries for t in tokenize(q)))
        targets = {nid for nid in index.ids
                   if nid in plan.target_sections
                   or any(a.id in plan.target_sections for a in index.graph.ancestors(nid))}
        return cls(plan, tokens, vectors[0], semantic, lexical, section_semantic,
                   vectors[1:], targets)


def _scale(similarities: np.ndarray, scaling: str) -> np.ndarray:
    """"raw": clipped cosine. "minmax": rescaled to [0, 1] within this document, which
    makes the scoring weights mean the same thing for embedders whose cosine ranges
    differ (TF-IDF ~0-0.4, dense models ~0.1-0.8)."""
    sims = np.clip(similarities, 0, None)
    if scaling == "minmax" and sims.size:
        low, high = float(sims.min()), float(sims.max())
        return (sims - low) / (high - low) if high > low else np.zeros_like(sims)
    return sims


# ----------------------------------------------------------- Phase 5: retrieve

RRF_K = 60            # reciprocal-rank-fusion constant (standard value)
SOURCE_DEPTH = 100    # how deep each source's ranked list is read before fusion
# Structural and graph lists are derived (from titles / from other sources' hits),
# so each of their votes counts for less than a direct text match. "plan" is the
# planner's resolved targets: pinned into the candidate set, and ranked like the rest.
SOURCE_WEIGHTS = {"lexical": 1.0, "phrase": 1.0, "semantic": 1.0, "structural": 0.5,
                  "graph": 0.5, "plan": 1.0}
STRUCTURAL_SECTIONS = 3   # sections whose content forms the structural list
GRAPH_SEEDS = 10          # top fused nodes whose references form the graph list
# Each direct source's top hits always enter the candidate set: RRF demotes a node
# only one source finds (a paraphrase only the dense model gets, a rare term only
# BM25 gets), and the quota keeps fused recall >= every single method's recall@quota.
SOURCE_QUOTA = 5
QUOTA_SOURCES = ("lexical", "phrase", "semantic")
SOURCES = ("lexical", "phrase", "semantic", "structural", "graph", "plan")


def _ranked(scores: np.ndarray, ids: List[str], depth: int) -> List[str]:
    order = np.argsort(-scores)[:depth]
    return [ids[i] for i in order if scores[i] > 0]


def _section_content(graph: DocumentGraph, section_id: str, allowed: Set[str]) -> List[str]:
    """Allowed content under a section, in document order, at the coarsest level:
    a paragraph stands for its sentences unless paragraphs are not allowed."""
    out: List[str] = []

    def walk(nid: str) -> None:
        for child in graph.children(nid):
            if child.node_type in allowed:
                out.append(child.id)
            elif child.is_section or child.is_content:
                walk(child.id)

    walk(section_id)
    return out


def structural_list(index: GraphIndex, ctx: QueryContext, depth: int) -> List[str]:
    """Content of the sections whose path matches the question.

    A section scores its own title match plus half its best ancestor's (the chapter
    path: "Liability > Exclusions" matches "liability exclusions"). The top sections'
    content is listed best section first. Titles only: adding section-summary
    similarity here duplicated the semantic source and lost hit@1 on long documents
    (docs/USE_CASE.md, Phase 5)."""
    graph = index.graph
    query_tokens = list(dict.fromkeys(
        t for q in [ctx.plan.query] + list(ctx.plan.sub_queries) for t in _content_tokens(q)))

    def title_score(section: Node) -> float:
        title = _title_tokens(section.title or "")
        if not title or not query_tokens:
            return 0.0
        return sum(any(_tokens_match(t, q) for q in query_tokens) for t in title) / len(title)

    own = {sid: title_score(graph.nodes[sid]) for sid in index.section_ids}
    scores = {}
    for sid in index.section_ids:
        ancestors = [own.get(a.id, 0.0) for a in graph.ancestors(sid) if a.is_section]
        scores[sid] = own[sid] + 0.5 * max(ancestors, default=0.0)
    allowed = set(ctx.plan.target_types)
    out: List[str] = []
    for sid in sorted(scores, key=lambda s: -scores[s])[:STRUCTURAL_SECTIONS]:
        if scores[sid] <= 0:
            break
        out += [n for n in _section_content(graph, sid, allowed) if n not in out]
    return out[:depth]


def graph_list(graph: DocumentGraph, seeds: List[str], allowed: Set[str]) -> List[str]:
    """Nodes one hop from the best anchors: the paragraph of an anchored sentence,
    and what an anchor cites (defined terms it uses, clauses it references)."""
    out: List[str] = []

    def add(nid: str) -> None:
        if nid not in out and nid not in seeds and graph.nodes[nid].node_type in allowed:
            out.append(nid)

    for nid in seeds:
        node, parent = graph.nodes[nid], graph.parent(nid)
        if parent is not None and parent.is_content:
            add(parent.id)
        source = parent.id if node.node_type == SENTENCE and parent is not None else nid
        for nbr, _, _ in graph.neighbors(source, (DEFINITION_LINK, CROSS_REFERENCE), "out"):
            if nbr.is_content:
                add(nbr.id)
            else:
                for child_id in _section_content(graph, nbr.id, allowed)[:MAX_REFERENCED_PARAGRAPHS]:
                    add(child_id)
    return out


def _fuse(lists: Dict[str, List[str]], allowed: Set[str], graph: DocumentGraph
          ) -> Dict[str, float]:
    fused: Dict[str, float] = {}
    for source, ranked in lists.items():
        weight = SOURCE_WEIGHTS[source.split(":")[0]]
        for rank, nid in enumerate(ranked):
            if graph.nodes[nid].node_type in allowed:
                fused[nid] = fused.get(nid, 0.0) + weight / (RRF_K + rank + 1)
    return fused


def source_rankings(lists: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """One ranking per method: its lists (question + each sub-query) merged by best rank."""
    best: Dict[str, Dict[str, int]] = {}
    for name, ranked in lists.items():
        ranks = best.setdefault(name.split(":")[0], {})
        for rank, nid in enumerate(ranked):
            ranks[nid] = min(rank, ranks.get(nid, rank))
    return {m: sorted(r, key=lambda n: r[n]) for m, r in best.items()}


def retrieve_candidates(index: GraphIndex, ctx: QueryContext) -> Dict[str, Set[str]]:
    """Phase 5: hybrid candidate retrieval.

    Ranked lists from each source are fused with weighted reciprocal rank fusion
    (RRF), which needs no score calibration between BM25, phrase counts and cosine
    similarity:
      - lexical: BM25 for the question and each plan sub-query
      - phrase: quoted phrases and shared word pairs
      - semantic: vector search for the question and each sub-query
      - structural: content of the sections whose title / chapter path matches
      - plan: the planner's target sections / clauses and definitions of focus
        terms. These are also pinned: always included, outside the budget.
      - graph: references and parent paragraphs of the top fused anchors
    The fused list is cut to ``plan.max_candidates``. ``ctx.fused`` keeps the normalised fused score of every candidate,
    ``ctx.sources`` each method's ranking and the final candidate order (for
    per-method evaluation). Each candidate's reasons name the sources that found it.
    """
    graph, plan = index.graph, ctx.plan
    allowed = set(plan.target_types)
    ids = index.ids
    depth = max(SOURCE_DEPTH, plan.max_candidates)

    lists: Dict[str, List[str]] = {}
    queries = [plan.query] + list(plan.sub_queries)
    for i, query in enumerate(queries):
        suffix = "" if i == 0 else f":sub{i}"
        lists["lexical" + suffix] = _ranked(index.bm25.scores(tokenize(query)), ids, depth)
        lists["phrase" + suffix] = _ranked(index.phrases.scores(query), ids, depth)
    for i, vector in enumerate([ctx.query_vector] + ctx.sub_vectors):
        suffix = "" if i == 0 else f":sub{i}"
        lists["semantic" + suffix] = [nid for nid, sim in index.search(vector, depth, plan.target_types)
                                      if sim > 0]

    pinned: Dict[str, Set[str]] = {}

    def pin(nid: str, reason: str) -> None:
        if graph.nodes[nid].node_type in allowed:
            pinned.setdefault(nid, set()).add(reason)

    # Plan: the sections and clauses the planner resolved. A summary takes
    # the whole section; other strategies take its direct content.
    for target_id in plan.target_sections:
        target = graph.nodes[target_id]
        if target.is_content:
            pin(target_id, f"plan:target:{target_id}")
            continue
        members = ([n for n in ids if n in ctx.target_nodes
                    and target_id in {a.id for a in graph.ancestors(n)}]
                   if plan.strategy == SECTION_SUMMARY else
                   [c.id for c in graph.children(target_id)])
        for nid in members:
            pin(nid, f"plan:section:{target_id}")
    if plan.query_type == DEFINITION:
        for term in plan.focus_terms:
            for defined, nid in graph.definitions.items():
                if defined == term or re.search(rf"\b{re.escape(term)}\b", defined):
                    pin(nid, f"definition:{defined}")

    lists["structural"] = structural_list(index, ctx, depth)
    lists["plan"] = list(pinned)
    first_pass = _fuse(lists, allowed, graph)
    seeds = sorted(first_pass, key=lambda n: -first_pass[n])[:GRAPH_SEEDS]
    lists["graph"] = graph_list(graph, seeds, allowed)
    fused = _fuse(lists, allowed, graph)

    reasons: Dict[str, Set[str]] = {}
    for source, ranked in lists.items():
        for nid in ranked[:plan.max_candidates]:
            if source != "plan":  # pins carry their own, more specific reasons
                reasons.setdefault(nid, set()).add(source)

    # Pinned nodes and each direct source's quota are always in; the rest of the
    # budget goes by fused rank. The result is ordered by fused score so its
    # prefixes are comparable with each method's ranking.
    chosen = set(pinned)
    for source, ranked in lists.items():
        if source.split(":")[0] in QUOTA_SOURCES:
            chosen.update(n for n in ranked[:SOURCE_QUOTA] if n in fused)
    budget = max(plan.max_candidates, len(chosen))
    order = sorted(fused, key=lambda n: -fused[n])
    for nid in order:
        if len(chosen) >= budget:
            break
        chosen.add(nid)
    candidates = {nid: pinned.get(nid, set()) | reasons.get(nid, set()) | {"fused"}
                  for nid in order if nid in chosen}
    top = fused[order[0]] if order else 0.0
    ctx.fused = {nid: fused[nid] / top for nid in candidates}
    ctx.sources = source_rankings(lists)
    ctx.sources["fused"] = list(candidates)
    return candidates


# ------------------------------------------------------------- Phase 6: expand

# How likely a neighbour is needed to read its anchor correctly. Expansion is
# best-first on anchor strength x prior x HOP_DECAY^hop, so when the node budget
# runs out it is the loosely related neighbours that are left behind.
EXPANSION_PRIOR = {
    "parent_of": 1.0,       # the anchor's section: heading and citation path
    "paragraph_of": 0.95,   # the paragraph around an anchored sentence / list item
    "definition": 0.9,      # definition of a term the anchor uses
    "reference": 0.85,      # a clause the anchor cites
    "continuation": 0.7,    # prev / next that the anchor reads into or out of
    "same_section": 0.6,    # sibling, when the plan wants the whole section
    "adjacent": 0.35,       # prev / next with no sequencing cue
    "similar": 0.3,         # semantic_similarity neighbour
}
HOP_DECAY = 0.5
PROPAGATING = ("paragraph_of", "definition", "reference")  # reached nodes expanded again
_CONTINUES = re.compile(
    r"^(?:however|provided|except|notwithstanding|unless|otherwise|accordingly|thereafter|"
    r"such|this|these|those|the foregoing|in (?:that|such) case|if so)\b", re.I)
_LEADS_ON = re.compile(r"(?:[:;,]|\b(?:and|or|including))\s*$", re.I)


def continues(first: Node, second: Node) -> bool:
    """Sequencing cue between consecutive nodes: ``first`` introduces a list or ends
    mid-thought, or ``second`` opens by pointing back ("Such fees...", "provided that")."""
    text = second.text.lstrip()
    return bool(_LEADS_ON.search(first.text)) or bool(text) and (
        text[0].islower() or text[0] == "(" or bool(_CONTINUES.match(text)))


class Subgraph(dict):
    """Node id -> reasons it is in the subgraph, plus for every expanded node the path
    from the anchor that reached it and the priority it was reached with."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paths: Dict[str, List[str]] = {}
        self.priority: Dict[str, float] = {}
        self.skipped = 0  # neighbours left out by the budget


def expand_subgraph(graph: DocumentGraph, candidates: Dict[str, Set[str]], plan: QueryPlan,
                    strength: Optional[Dict[str, float]] = None,
                    budget: Optional[int] = None) -> Subgraph:
    """Phase 6: grow anchors into a bounded local neighbourhood.

    For each anchor: its section, the paragraph around it, definitions it uses,
    clauses it cites, its prev / next nodes and (``plan.expand_same_section``) its
    siblings. Definitions, cited clauses and enclosing paragraphs are expanded again
    up to ``plan.expansion_radius`` hops. At most ``budget`` (default
    ``plan.max_expanded``) content nodes are added, best first; sections cost
    nothing (one per anchor at most, and needed for citations).
    """
    import heapq

    budget = plan.max_expanded if budget is None else budget
    strength = strength or {}
    subgraph = Subgraph({nid: set(reasons) for nid, reasons in candidates.items()})
    for nid in candidates:
        subgraph.paths[nid] = [nid]
        subgraph.priority[nid] = max(strength.get(nid, 0.0), 0.1) if strength else 1.0
    heap: List[tuple] = []
    counter = 0

    def push(target: str, kind: str, origin: str, hop: int) -> None:
        nonlocal counter
        prior = EXPANSION_PRIOR[kind]
        if kind == "similar" and plan.strategy == PAIRWISE:
            prior = EXPANSION_PRIOR["continuation"]  # conflicts live in similar passages
        priority = subgraph.priority[origin] * prior * HOP_DECAY ** hop
        counter += 1
        heapq.heappush(heap, (-priority, counter, target, kind, origin, hop))

    def push_neighbours(nid: str, hop: int) -> None:
        node, parent = graph.nodes[nid], graph.parent(nid)
        if parent is not None and parent.is_section:
            push(parent.id, "parent_of", nid, hop)
        if parent is not None and parent.is_content:  # sentence / list item
            push(parent.id, "paragraph_of", nid, hop)
        edges = (NEXT_PREV, SEMANTIC_SIMILARITY) + ((SAME_SECTION,) if plan.expand_same_section else ())
        for nbr, edge_type, _ in graph.neighbors(nid, edges):
            if edge_type == NEXT_PREV:
                if not nbr.is_content:
                    continue
                first, second = sorted((node, nbr), key=lambda n: n.start)
                push(nbr.id, "continuation" if continues(first, second) else "adjacent", nid, hop)
            elif edge_type == SAME_SECTION:
                push(nbr.id, "same_section", nid, hop)
            elif hop > 0 or plan.strategy == PAIRWISE:
                push(nbr.id, "similar", nid, hop)
        # Follow references outward only: usage -> definition, clause -> cited clause.
        # Sentences carry no reference edges; use their paragraph's.
        ref_source = parent.id if node.node_type == SENTENCE and parent else nid
        for nbr, edge_type, _ in graph.neighbors(ref_source, (DEFINITION_LINK, CROSS_REFERENCE), "out"):
            kind = "definition" if edge_type == DEFINITION_LINK else "reference"
            push(nbr.id, kind, nid, hop)
            if not nbr.is_content:  # a cited section: its opening paragraphs
                for child in graph.children(nbr.id)[:MAX_REFERENCED_PARAGRAPHS]:
                    if child.is_content:
                        push(child.id, kind, nid, hop)

    for nid in candidates:
        push_neighbours(nid, 0)
    added = 0
    while heap:
        neg_priority, _, target, kind, origin, hop = heapq.heappop(heap)
        reason = f"expansion:{kind}:{origin}"
        if target in subgraph:
            subgraph[target].add(reason)
            continue
        is_content = graph.nodes[target].is_content
        if is_content and added >= budget:
            subgraph.skipped += 1
            continue
        subgraph[target] = {reason}
        subgraph.paths[target] = subgraph.paths[origin] + [target]
        subgraph.priority[target] = -neg_priority
        if is_content:
            added += 1
            if kind in PROPAGATING and hop + 1 < plan.expansion_radius:
                push_neighbours(target, hop + 1)
    return subgraph


# ------------------------------------------------------------------ helpers

@dataclass
class RetrievalTrace:
    """Every stage's output plus timings, so each stage can be evaluated on its own."""

    plan: QueryPlan
    candidates: Dict[str, Set[str]]
    subgraph: Subgraph
    scored: List[ScoredNode]
    timings: Dict[str, float]
    sources: Dict[str, List[str]] = field(default_factory=dict)  # per-method rankings
    confidence: Optional[EvidenceConfidence] = None              # Phase 7 abstention signal


def retrieve(index: GraphIndex, query: str, plan: Optional[QueryPlan] = None,
             candidate_budget: Optional[int] = None,
             expansion_budget: Optional[int] = None,
             weights: Optional[Dict[str, float]] = None, scorer: Optional[Scorer] = None,
             calibration: Optional[Calibration] = None) -> RetrievalTrace:
    """``candidate_budget`` / ``expansion_budget`` override the plan's
    ``max_candidates`` / ``max_expanded`` (for budget sweeps). ``weights`` override
    individual scoring weights of the plan; ``scorer`` replaces the linear scorer.
    ``calibration`` turns the evidence-confidence signal into an abstention decision."""
    clock = time.perf_counter
    timings: Dict[str, float] = {}
    start = clock()
    plan = plan or plan_query(query, index.graph, index)
    if candidate_budget is not None:
        plan.max_candidates = candidate_budget
    if expansion_budget is not None:
        plan.max_expanded = expansion_budget
    if weights:
        plan.weights.update(weights)
    timings["plan"] = clock() - start
    start = clock()
    ctx = QueryContext.build(index, plan)
    candidates = retrieve_candidates(index, ctx)
    timings["candidates"] = clock() - start
    start = clock()
    subgraph = expand_subgraph(index.graph, candidates, plan, ctx.fused)
    timings["expansion"] = clock() - start
    start = clock()
    scored = score_subgraph(index, ctx, subgraph, scorer)
    confidence = evidence_confidence(index, plan, ctx.query_vector, calibration)
    timings["scoring"] = clock() - start
    return RetrievalTrace(plan, candidates, subgraph, scored, timings, ctx.sources, confidence)


def run_retrieval(index: GraphIndex, query: str, plan: Optional[QueryPlan] = None
                  ) -> List[ScoredNode]:
    return retrieve(index, query, plan).scored


def retrieve_relevant_nodes(graph: DocumentGraph, query: str, top_k: int = 10,
                            embedder: Optional[Embedder] = None) -> List[Node]:
    """Convenience API: ranked content nodes for ``query``."""
    index = GraphIndex(graph, embedder=embedder)
    scored = run_retrieval(index, query)
    return [s.node for s in scored if s.node.is_content][:top_k]
