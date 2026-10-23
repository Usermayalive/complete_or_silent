"""End-to-end flow: parse -> graph -> index -> plan -> retrieve -> expand ->
score -> prune -> evidence pack -> answer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from .document_graph import DocumentGraph
from .embeddings import Embedder
from .evidence import LLM, EvidencePack, build_evidence_pack, generate_answer
from .index import GraphIndex
from .pruning import PruneConfig, PrunedSubgraph, default_config, prune
from .query_planner import QueryPlan, plan_query
from .retrieval import RetrievalTrace, retrieve
from .scoring import Calibration, EvidenceConfidence, Scorer, ScoredNode, load_calibration

if TYPE_CHECKING:
    from .answer import Answer
    from .store import GraphStore


@dataclass
class QAResult:
    plan: QueryPlan
    scored: List[ScoredNode]
    pruned: PrunedSubgraph
    evidence: EvidencePack
    answer: str
    trace: Optional[RetrievalTrace] = None
    grounded: Optional["Answer"] = None  # Phase 10: answer, cited evidence, confidence, usage

    @property
    def confidence(self) -> Optional[EvidenceConfidence]:
        return self.trace.confidence if self.trace else None


class DocumentQA:
    def __init__(self, graph: DocumentGraph, embedder: Optional[Embedder] = None,
                 llm: Optional[LLM] = None, candidate_budget: Optional[int] = None,
                 expansion_budget: Optional[int] = None,
                 weights: Optional[Dict[str, float]] = None, scorer: Optional[Scorer] = None,
                 calibration: Union[str, Calibration, None] = "auto", abstain: bool = False,
                 prune_config: Optional[PruneConfig] = None, budgets: Optional[Dict[str, Optional[int]]] = None,
                 **index_options):
        """``index_options`` go to ``GraphIndex`` (contextual, vector_index, semantic_scaling).
        ``candidate_budget`` / ``expansion_budget`` override every plan's candidate
        count / expansion node budget. ``weights`` override scoring weights;
        ``scorer`` replaces the linear scorer. ``calibration="auto"`` loads the
        abstention calibration fitted for the embedder (none fitted: uncalibrated);
        ``None`` leaves the confidence uncalibrated. ``abstain=True`` returns no
        evidence when the calibrated confidence is insufficient. Off by default: on
        the eval sets it also drops colloquial paraphrase questions (docs/USE_CASE.md)."""
        self.graph = graph
        self.candidate_budget = candidate_budget
        self.expansion_budget = expansion_budget
        self.weights = weights
        self.scorer = scorer
        self.index = GraphIndex(graph, embedder=embedder, **index_options)
        if calibration == "auto":  # fitted on plain (non-contextual) node embeddings
            calibration = None if index_options.get("contextual") \
                else load_calibration(self.index.embedder.name)
        self.calibration = calibration
        self.abstain = abstain
        self.prune_config = prune_config or default_config(self.index.embedder.name)
        self.budgets = budgets  # per-route evidence token budgets (Phase 9); None = defaults, {} = none
        # ``llm``: a provider spec ("claude", "claude:<model>", "openai:<model>",
        # "extractive"), a Provider, or the older ``prompt -> str`` callable.
        if isinstance(llm, str):
            from .answer import load_provider

            llm = load_provider(llm)
        self.llm = llm

    @classmethod
    def from_text(cls, text: str, title: str = "Document", **kwargs) -> "DocumentQA":
        return cls(DocumentGraph.from_text(text, title=title), **kwargs)

    @classmethod
    def from_file(cls, path: str, store: Optional["GraphStore"] = None, parser: str = "native",
                  **kwargs) -> "DocumentQA":
        """Parse ``path`` (``parser``: native or docling), or reuse its graph from
        ``store`` if the file is unchanged."""
        graph = store.load_or_ingest(path) if store is not None else DocumentGraph.from_file(path, parser)
        return cls(graph, **kwargs)

    def ask(self, question: str) -> QAResult:
        trace = retrieve(self.index, question, candidate_budget=self.candidate_budget,
                         expansion_budget=self.expansion_budget, weights=self.weights,
                         scorer=self.scorer, calibration=self.calibration)
        start = time.perf_counter()
        if self.abstain and trace.confidence.insufficient:
            pruned = PrunedSubgraph(evidence=[], dropped=[s.node.id for s in trace.scored])
        else:
            pruned = prune(self.graph, trace.scored, trace.plan, self.prune_config)
        trace.timings["pruning"] = time.perf_counter() - start
        level = trace.confidence.level if trace.confidence is not None else None
        pack = build_evidence_pack(self.graph, trace.plan, pruned, confidence=level, budgets=self.budgets)
        from .answer import CallableProvider, answer_question

        if callable(self.llm) and not hasattr(self.llm, "complete"):
            # Older interface: the callable's reply is the answer text, parsed for grounding.
            text = generate_answer(pack, self.llm)
            grounded = answer_question(pack, CallableProvider(self.llm, pack), abstain=self.abstain)
            return QAResult(trace.plan, trace.scored, pruned, pack, text, trace, grounded)
        grounded = answer_question(pack, self.llm, abstain=self.abstain)
        text = generate_answer(pack, None) if self.llm is None else grounded.render()
        return QAResult(trace.plan, trace.scored, pruned, pack, text, trace, grounded)
