"""Phase 7: relevance scoring of the expanded subgraph, and the abstention signal.

Every subgraph node gets a feature vector (``FEATURES``). A scorer turns it into one
score plus a per-feature contribution breakdown, so a ranking can always be
explained ("semantic +0.31, lexical +0.18, noise -0.15"). ``LinearScorer`` is the
default; a learned ranker (or a GNN over the subgraph) can replace it behind the
same ``Scorer`` interface, and the feature vectors are its training inputs.

Scores rank nodes within one question; they are not comparable across questions,
because semantic and lexical features are min-max scaled per document. Whether the
document holds an answer at all is a separate, query-level judgement:
``EvidenceConfidence`` uses absolute signals (raw cosine margin, IDF-weighted term
coverage, explicit anchors) and a logistic calibration fitted per embedder on
``eval/calibration_v1.jsonl``. Below the calibrated threshold the evidence is
flagged insufficient; ``DocumentQA(abstain=True)`` then returns no evidence.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Sequence, Set, Tuple

import numpy as np

from .document_graph import DocumentGraph
from .index import GraphIndex, tokenize
from .models import (CHAPTER, CLAUSE, CROSS_REFERENCE, DEFINITION_LINK, FIGURE, PARAGRAPH,
                     SECTION, SENTENCE, SUBSECTION, TABLE, Node)
from .query_planner import DEFINITION, PER_SIDE, SECTION_SUMMARY, QueryPlan

if TYPE_CHECKING:
    from .retrieval import QueryContext

TYPE_VALUE = {PARAGRAPH: 1.0, CLAUSE: 1.0, TABLE: 0.9, SENTENCE: 0.8, FIGURE: 0.6,
              CHAPTER: 0.3, SECTION: 0.3, SUBSECTION: 0.3}
MIN_USEFUL_CHARS = 25

# Feature -> what it measures. All lie in [0, 1]; "noise" is a penalty.
FEATURES = {
    "semantic": "embedding similarity to the question (or a sub-query), min-max scaled",
    "lexical": "BM25 score for the question (or a sub-query), scaled by the best node's",
    "centrality": "query relevance flowing in from subgraph neighbours (personalised PageRank)",
    "parent_context": "how well the enclosing section's title / summary matches the question",
    "section_importance": "share of retrieval anchors in the node's section, vs the busiest section",
    "cross_reference": "how often other clauses cite or define into this node",
    "path_coherence": "strength of the best path from an anchor (Phase 6 expansion priority)",
    "type_value": "prior value of the node type (paragraph / clause > sentence > section)",
    "definition": "the node defines a focus term of the question",
    "target_section": "the node is inside a section the planner resolved",
    "fusion": "Phase 5 fused rank score (0 for nodes added by expansion)",
    "noise": "penalty: very short text, or reached only by expansion with no similarity",
}
PENALTIES = ("noise",)
PAGERANK_ALPHA = 0.85
PAGERANK_ITERATIONS = 30


@dataclass
class ScoredNode:
    node: Node
    score: float
    features: Dict[str, float]
    reasons: List[str] = field(default_factory=list)
    contributions: Dict[str, float] = field(default_factory=dict)  # feature -> signed share of score

    def explain(self, top: int = 4) -> str:
        parts = sorted(self.contributions.items(), key=lambda kv: -abs(kv[1]))[:top]
        return " ".join(f"{k}{v:+.2f}" for k, v in parts if v)


class Scorer(Protocol):
    name: str

    def score(self, features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """(score, per-feature contributions). Contributions should sum to the score."""
        ...


class LinearScorer:
    """score = sum(weight * feature) - sum(weight * penalty)."""

    name = "linear"

    def __init__(self, weights: Dict[str, float]):
        unknown = set(weights) - set(FEATURES)
        if unknown:
            raise ValueError(f"unknown scoring features: {sorted(unknown)}")
        self.weights = dict(weights)

    def score(self, features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        contributions = {}
        for name, value in features.items():
            weight = self.weights.get(name, 0.0)
            contributions[name] = (-weight if name in PENALTIES else weight) * value
        return sum(contributions.values()), contributions


def load_weights(path: str) -> Dict[str, float]:
    """Scoring weights from a JSON object, e.g. {"semantic": 0.4, "centrality": 0.1}."""
    weights = json.loads(Path(path).read_text(encoding="utf-8"))
    LinearScorer(weights)  # validate names
    return {k: float(v) for k, v in weights.items()}


# ------------------------------------------------------------------ features

def _title_overlap(title: str, tokens: List[str]) -> float:
    title_tokens = set(tokenize(title))
    if not title_tokens or not tokens:
        return 0.0
    return len(title_tokens & set(tokens)) / len(title_tokens)


def definition_ids(graph: DocumentGraph, plan: QueryPlan) -> Set[str]:
    """Definition nodes of the plan's focus terms."""
    return {nid for term in plan.focus_terms
            for defined, nid in graph.definitions.items()
            if defined == term or re.search(rf"\b{re.escape(term)}\b", defined)}


def neighbour_support(graph: DocumentGraph, nodes: Sequence[str],
                      relevance: Dict[str, float]) -> Dict[str, float]:
    """Personalised PageRank over the subgraph, restarting at nodes in proportion to
    their own relevance; a node's support is the rank it receives from neighbours
    (its rank minus its restart share), scaled to [0, 1]. High for a node that
    relevant nodes link to - the section several hits sit in, a definition or clause
    they all cite - even when its own text matches the question weakly."""
    position = {nid: i for i, nid in enumerate(nodes)}
    links = [sorted({position[nbr.id] for nbr, _, _ in graph.neighbors(nid)
                     if nbr.id in position and nbr.id != nid}) for nid in nodes]
    restart = np.array([max(relevance.get(nid, 0.0), 0.0) for nid in nodes], dtype=np.float64)
    if not len(nodes) or restart.sum() <= 0:
        return {nid: 0.0 for nid in nodes}
    restart /= restart.sum()
    rank = restart.copy()
    for _ in range(PAGERANK_ITERATIONS):
        nxt = (1 - PAGERANK_ALPHA) * restart
        for i, targets in enumerate(links):
            if targets:
                nxt[targets] += PAGERANK_ALPHA * rank[i] / len(targets)
            else:  # dangling node: its mass restarts
                nxt += PAGERANK_ALPHA * rank[i] * restart
        rank = nxt
    support = np.clip(rank - (1 - PAGERANK_ALPHA) * restart, 0, None)
    top = support.max()
    return {nid: float(support[i] / top) if top > 0 else 0.0 for i, nid in enumerate(nodes)}


def subgraph_features(index: GraphIndex, ctx: "QueryContext",
                      subgraph: Dict[str, Set[str]]) -> Dict[str, Dict[str, float]]:
    """Feature vector of every content and section node in the subgraph."""
    graph, plan = index.graph, ctx.plan

    def section_signal(section_id: Optional[str]) -> float:
        if section_id is None or section_id not in index.section_position:
            return 0.0
        title = graph.nodes[section_id].title or ""
        sem = float(ctx.section_semantic[index.section_position[section_id]])
        return max(sem, _title_overlap(title, ctx.tokens))

    anchors = {nid for nid, r in subgraph.items() if any(not x.startswith("expansion") for x in r)}
    anchor_sections: Dict[str, int] = {}
    for nid in anchors:
        sec = graph.section_of(nid)
        if sec:
            anchor_sections[sec.id] = anchor_sections.get(sec.id, 0) + 1
    max_hits = max(anchor_sections.values(), default=1)
    defined = definition_ids(graph, plan)
    priority = getattr(subgraph, "priority", {})
    top_priority = max(priority.values(), default=0.0) or 1.0

    out: Dict[str, Dict[str, float]] = {}
    for nid in subgraph:
        node = graph.nodes[nid]
        if node.is_content:
            pos = index.position[nid]
            semantic, lexical = float(ctx.semantic[pos]), float(ctx.lexical[pos])
        elif node.is_section:
            semantic = section_signal(nid)
            lexical = _title_overlap(node.title or "", ctx.tokens)
        else:
            continue
        section = graph.section_of(nid)
        inbound_refs = len(graph.neighbors(nid, (CROSS_REFERENCE, DEFINITION_LINK), "in"))
        out[nid] = {
            "semantic": semantic,
            "lexical": lexical,
            "centrality": 0.0,
            "parent_context": section_signal(section.id if section else None),
            "section_importance": (anchor_sections.get(section.id, 0) / max_hits) if section else 0.0,
            "cross_reference": min(1.0, inbound_refs / 2),
            "path_coherence": priority.get(nid, 1.0 if nid in anchors else 0.0) / top_priority
                              if priority else float(nid in anchors),
            "type_value": TYPE_VALUE.get(node.node_type, 0.0),
            "definition": 1.0 if nid in defined else 0.0,
            "target_section": 1.0 if nid in ctx.target_nodes or nid in plan.target_sections else 0.0,
            "fusion": ctx.fused.get(nid, 0.0),
            "noise": (1.0 if len(node.text) < MIN_USEFUL_CHARS and node.is_content else 0.0)
                     + (0.5 if nid not in anchors and semantic < 0.05 else 0.0),
        }
    relevance = {nid: max(f["semantic"], f["lexical"]) for nid, f in out.items()}
    for nid, value in neighbour_support(graph, list(out), relevance).items():
        out[nid]["centrality"] = value
    return out


def score_subgraph(index: GraphIndex, ctx: "QueryContext", subgraph: Dict[str, Set[str]],
                   scorer: Optional[Scorer] = None) -> List[ScoredNode]:
    """Rank the subgraph: best first, each node with features and contributions.

    A scorer with ``score_nodes(index, ctx, features)`` (e.g. the GNN) scores all nodes
    at once and sees the whole graph; others score one feature vector at a time."""
    scorer = scorer or LinearScorer(ctx.plan.weights)
    features_by_node = subgraph_features(index, ctx, subgraph)
    if hasattr(scorer, "score_nodes"):
        scored = scorer.score_nodes(index, ctx, features_by_node)
    else:
        scored = {nid: scorer.score(f) for nid, f in features_by_node.items()}
    results = []
    for nid, features in features_by_node.items():
        score, contributions = scored[nid]
        results.append(ScoredNode(index.graph.nodes[nid], score, features,
                                  sorted(subgraph[nid]), contributions))
    results.sort(key=lambda s: s.score, reverse=True)
    return results


# ------------------------------------------------------- abstention signal

CALIBRATION_FILE = Path(__file__).with_name("calibration.json")
HIGH_CONFIDENCE = 0.5
# Inverse regularisation of the calibration fit. Both features lie in [0, 1], so the
# sklearn default (1.0) flattened the model: on the calibration set it abstained on
# 5 of 16 unanswerable questions vs 11 of 16 at 100, with no answerable one lost.
CALIBRATION_C = 100.0


@dataclass
class EvidenceConfidence:
    """Does the document contain evidence for the question at all?

    ``margin``: best raw cosine minus the document's median cosine. A question about
    something the document covers has a passage that stands out; raw cosines alone
    drift with document length and embedder.
    ``coverage``: the best node's share of the question's IDF-weighted terms (a term
    no node contains weighs the most).
    ``anchor``: the question names something that verifiably exists (a cited section,
    a defined term, a quoted phrase found verbatim, resolved comparison sides or a
    resolved summary section). An anchored question is never abstained on.
    """

    margin: float
    coverage: float
    anchor: str = ""
    probability: Optional[float] = None  # calibrated P(answerable); None if uncalibrated
    threshold: Optional[float] = None

    @property
    def insufficient(self) -> bool:
        return (not self.anchor and self.probability is not None
                and self.threshold is not None and self.probability < self.threshold)

    @property
    def level(self) -> str:
        if self.anchor:
            return "high"
        if self.probability is None:
            return "uncalibrated"
        if self.insufficient:
            return "insufficient"
        return "high" if self.probability >= HIGH_CONFIDENCE else "low"

    def explain(self) -> str:
        p = "n/a" if self.probability is None else f"{self.probability:.2f}"
        t = "n/a" if self.threshold is None else f"{self.threshold:.2f}"
        anchor = f", anchor: {self.anchor}" if self.anchor else ""
        return (f"evidence confidence {self.level} (p={p}, threshold {t}; margin "
                f"{self.margin:.3f}, coverage {self.coverage:.2f}{anchor})")


@dataclass
class Calibration:
    """Logistic model P(answerable | margin, coverage) for one embedder."""

    embedder: str
    coef: List[float]
    intercept: float
    threshold: float
    fitted_on: str = ""
    cases: int = 0

    def probability(self, margin: float, coverage: float) -> float:
        z = self.intercept + self.coef[0] * margin + self.coef[1] * coverage
        return 1.0 / (1.0 + math.exp(-z))


def load_calibration(embedder_name: str, path: Path = CALIBRATION_FILE) -> Optional[Calibration]:
    if not path.exists():
        return None
    entry = json.loads(path.read_text(encoding="utf-8")).get(embedder_name)
    return Calibration(**entry) if entry else None


def save_calibration(calibration: Calibration, path: Path = CALIBRATION_FILE) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[calibration.embedder] = asdict(calibration)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _anchor(index: GraphIndex, plan: QueryPlan) -> str:
    if plan.section_refs and plan.target_sections:
        return "cited section"
    if plan.query_type == DEFINITION and definition_ids(index.graph, plan):
        return "defined term"
    if plan.strategy == PER_SIDE and plan.sides and all(s.section_ids for s in plan.sides):
        return "comparison sides"
    if plan.strategy == SECTION_SUMMARY and plan.target_sections:
        return "summarised section"
    if index.phrases.quoted_found(plan.query):
        return "quoted phrase"
    return ""


def evidence_confidence(index: GraphIndex, plan: QueryPlan, query_vector: np.ndarray,
                        calibration: Optional[Calibration] = None) -> EvidenceConfidence:
    if not index.ids:
        return EvidenceConfidence(0.0, 0.0, probability=0.0 if calibration else None,
                                  threshold=calibration.threshold if calibration else None)
    cosines = index.matrix @ query_vector
    margin = float(cosines.max() - np.median(cosines))
    bm25 = index.bm25
    tokens = list(dict.fromkeys(tokenize(plan.query)))
    coverage = 0.0
    if tokens:
        weights = np.array([bm25.idf[bm25.vocab[t]] if t in bm25.vocab else bm25.max_idf
                            for t in tokens])
        known = [(bm25.vocab[t], w) for t, w in zip(tokens, weights) if t in bm25.vocab]
        if known:
            present = (bm25.weights[:, [c for c, _ in known]] > 0).astype(np.float64)
            covered = np.asarray(present @ np.array([w for _, w in known])).ravel()
            coverage = float(covered.max() / weights.sum())
    confidence = EvidenceConfidence(margin, coverage, _anchor(index, plan))
    if calibration is not None:
        confidence.probability = calibration.probability(margin, coverage)
        confidence.threshold = calibration.threshold
    return confidence


def fit_calibration(confidences: Sequence[EvidenceConfidence], answerable: Sequence[bool],
                    embedder: str, fitted_on: str = "", keep: float = 1.0) -> Calibration:
    """Fit P(answerable) on unanchored questions; the threshold keeps the ``keep``
    share of the answerable ones (1.0: abstain only below every answerable question
    seen, so wrong abstentions are rare at the cost of abstaining less)."""
    from sklearn.linear_model import LogisticRegression

    rows = [(c, a) for c, a in zip(confidences, answerable) if not c.anchor]
    labels = [int(a) for _, a in rows]
    if len(set(labels)) < 2:
        raise ValueError("calibration needs both answerable and unanswerable questions")
    X = np.array([[c.margin, c.coverage] for c, _ in rows])
    model = LogisticRegression(C=CALIBRATION_C).fit(X, labels)
    calibration = Calibration(embedder, [float(v) for v in model.coef_[0]],
                              float(model.intercept_[0]), 0.0, fitted_on, len(rows))
    kept = sorted(calibration.probability(c.margin, c.coverage) for c, a in rows if a)
    calibration.threshold = math.floor(kept[int((1 - keep) * len(kept))] * 1000) / 1000
    return calibration
