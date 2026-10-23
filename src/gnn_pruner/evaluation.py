"""Phase 0 evaluation harness: score the pipeline against a gold question set.

Gold evidence is a list of verbatim snippets; an evidence item "matches" a
snippet if it contains it (case- and whitespace-insensitive). Unanswerable
questions should produce an empty evidence pack (abstention).

    python -m gnn_pruner.evaluation eval/contracts_v1.jsonl [--verbose] [--json]
    python -m gnn_pruner.evaluation eval/contracts_v1.jsonl --embedder st:all-MiniLM-L6-v2
    python -m gnn_pruner.evaluation eval/contracts_v1.jsonl --compare tfidf st:all-mpnet-base-v2
    python -m gnn_pruner.evaluation eval/planner_v1.jsonl --planner   # query planner only
    python -m gnn_pruner.evaluation eval/contracts_v1.jsonl --sources # recall per retrieval method
    python -m gnn_pruner.evaluation eval/contracts_v1.jsonl --expansion-budgets 0 10 20 40

``context_recall`` measures expansion (Phase 6): of the nodes each gold passage
depends on (its section, definitions of terms it uses, clauses it cites), the
share that reached the expanded subgraph.

``embedding_recall_at_5`` measures the embedder alone: the share of gold snippets
found in the top 5 nodes of a pure vector search, before any graph logic.

Pruning (Phase 8): ``mean_pack_words`` is pack size in words; ``primary_precision``
counts only primary items (dependencies such as definitions are context, not noise);
``pack_context_recall`` is the share of the gold passages' dependencies (definitions
they use, clauses they cite) that made it into the pack.

Ranking (Phase 7): ``mrr`` and ``ndcg_at_10`` score the order of the scored content
nodes; ``--ranking`` compares it with the Phase 5 fused candidate order and each
retrieval method. Abstention: ``confidence_auc`` is how well the evidence-confidence
probability separates answerable from unanswerable questions;
``false_abstention_rate`` is the share of answerable questions abstained on
(``--abstain``; otherwise the pipeline never returns an empty pack on confidence).

    python -m gnn_pruner.evaluation eval/contracts_v1.jsonl --ranking
    python -m gnn_pruner.evaluation eval/calibration_v1.jsonl --fit-calibration --embedder auto
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .embeddings import Embedder, load_embedder
from .models import CONTENT_TYPES, CROSS_REFERENCE, DEFINITION_LINK, SENTENCE
from .parser import normalize_ws
from .pipeline import DocumentQA
from .scoring import fit_calibration, load_weights, save_calibration

EMBEDDING_RECALL_K = 5
SOURCE_RECALL_KS = (5, 10, 20, 30)
NDCG_K = 10
RANKING_METRICS = ("hit_at_1", "mrr", f"ndcg_at_{NDCG_K}", "recall_at_5")


@dataclass
class EvalCase:
    id: str
    document: str
    question: str
    query_type: str
    gold_evidence: List[str]
    answerable: bool = True
    reference_answer: str = ""
    tags: Dict = field(default_factory=dict)  # e.g. generated corpus: kind, paraphrase, split


@dataclass
class CaseResult:
    id: str
    question: str
    predicted_type: str
    type_correct: bool
    answerable: bool
    pack_size: int
    found: List[str] = field(default_factory=list)
    missed: List[str] = field(default_factory=list)
    recall: Optional[float] = None
    precision: Optional[float] = None
    hit_at_1: Optional[bool] = None
    embedding_recall: Optional[float] = None
    candidate_recall: Optional[float] = None   # gold reached the candidate set (Phase 5)
    expanded_recall: Optional[float] = None    # gold reached the expanded subgraph (Phase 6)
    context_recall: Optional[float] = None     # gold's dependencies in the subgraph (Phase 6)
    candidate_count: int = 0
    subgraph_size: int = 0                     # content nodes after expansion
    abstained: bool = False
    pack_words: int = 0
    primary_precision: Optional[float] = None
    pack_context_recall: Optional[float] = None
    # Phase 5, per retrieval method: "semantic@10" -> share of gold in its top 10.
    source_recall: Dict[str, float] = field(default_factory=dict)
    # Phase 7: ordering -> ranking metrics ("scored", "fused", each method).
    ranking: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidence: Optional[float] = None   # calibrated P(answerable)
    anchor: str = ""
    margin: float = 0.0
    coverage: float = 0.0


def _norm(text: str) -> str:
    return normalize_ws(text).lower()


def load_cases(path: str) -> List[EvalCase]:
    base = Path(path).resolve().parent
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = EvalCase(**json.loads(line))
            case.document = str((base / case.document).resolve())
            cases.append(case)
    return cases


def evaluate_case(qa: DocumentQA, case: EvalCase) -> CaseResult:
    result = qa.ask(case.question)
    texts = [_norm(item.text) for item in result.evidence.items]
    trace = result.trace
    gold = [_norm(g) for g in case.gold_evidence]
    row = CaseResult(
        id=case.id,
        question=case.question,
        predicted_type=result.plan.query_type,
        type_correct=result.plan.query_type == case.query_type,
        answerable=case.answerable,
        pack_size=len(texts),
        abstained=not texts,
    )
    row.pack_words = sum(len(item.text.split()) for item in result.evidence.items)
    conf = trace.confidence
    row.confidence, row.anchor = conf.probability, conf.anchor
    row.margin, row.coverage = conf.margin, conf.coverage
    if case.answerable and gold:
        row.found = [g for g in case.gold_evidence if any(_norm(g) in t for t in texts)]
        row.missed = [g for g in case.gold_evidence if g not in row.found]
        row.recall = len(row.found) / len(gold)
        matching = [t for t in texts if any(g in t for g in gold)]
        row.precision = len(matching) / len(texts) if texts else 0.0
        row.hit_at_1 = bool(texts) and any(g in texts[0] for g in gold)
        primary = [_norm(i.text) for i in result.evidence.items if i.role == "primary"]
        if primary:
            row.primary_precision = sum(any(g in t for g in gold) for t in primary) / len(primary)
        pack_deps = {d for d in gold_dependencies(qa.graph, gold) if qa.graph.nodes[d].is_content}
        if pack_deps:
            in_pack = {i.node_id for i in result.evidence.items}
            in_pack |= {qa.graph.nodes[n].parent_id for n in in_pack}
            row.pack_context_recall = len(pack_deps & in_pack) / len(pack_deps)
        hits = qa.index.search(qa.index.encode_query(case.question), EMBEDDING_RECALL_K,
                               CONTENT_TYPES)
        top = [_norm(qa.graph.nodes[nid].text) for nid, _ in hits]
        row.embedding_recall = sum(any(g in t for t in top) for g in gold) / len(gold)
        row.candidate_recall = _stage_recall(qa, trace.candidates, gold)
        row.expanded_recall = _stage_recall(qa, trace.subgraph, gold)
        needed = gold_dependencies(qa.graph, gold)
        if needed:
            row.context_recall = len(needed & set(trace.subgraph)) / len(needed)
        for method, ranked in trace.sources.items():
            for k in SOURCE_RECALL_KS:
                row.source_recall[f"{method}@{k}"] = _stage_recall(qa, ranked[:k], gold)
            row.ranking[method] = ranking_metrics(qa, ranked, gold)
        row.ranking["scored"] = ranking_metrics(
            qa, [s.node.id for s in result.scored if s.node.is_content], gold)
    row.candidate_count = len(trace.candidates)
    row.subgraph_size = sum(qa.graph.nodes[n].is_content for n in trace.subgraph)
    return row


def gold_dependencies(graph, gold: List[str]) -> set:
    """What the gold passages need to be read correctly: the section of each, the
    definitions of terms it uses and the clauses it cites (for a cited section, its
    first content node). A gold passage is the smallest non-sentence content node
    containing the snippet."""
    needed = set()
    for snippet in gold:
        holders = [n for n in graph.nodes.values() if n.is_content
                   and n.node_type != SENTENCE and snippet in _norm(n.text)]
        if not holders:
            continue
        node = min(holders, key=lambda n: len(n.text))
        section = graph.section_of(node.id)
        if section is not None:
            needed.add(section.id)
        for nbr, edge_type, _ in graph.neighbors(node.id, (DEFINITION_LINK, CROSS_REFERENCE), "out"):
            if nbr.is_content:
                needed.add(nbr.id)
            else:
                first = next((c for c in graph.children(nbr.id) if c.is_content), None)
                needed.add(first.id if first else nbr.id)
    return needed


def _stage_recall(qa: DocumentQA, node_ids, gold: List[str]) -> float:
    """Share of gold snippets contained in some node of a retrieval stage. A snippet
    inside a kept sentence or clause counts via the sentence; inside a paragraph, via
    the paragraph or any of its children (a later stage can widen to the paragraph)."""
    texts = []
    for nid in node_ids:
        node = qa.graph.nodes[nid]
        texts.append(_norm(node.text))
        parent = qa.graph.parent(nid)
        if parent is not None and parent.is_content:
            texts.append(_norm(parent.text))
    return sum(any(g in t for t in texts) for g in gold) / len(gold)


def ranking_metrics(qa: DocumentQA, order: List[str], gold: List[str]) -> Dict[str, float]:
    """A node is relevant if its text contains a gold snippet; each snippet counts
    once, at the first node that contains it (graded gain for nDCG)."""
    seen: set = set()
    first, dcg, hit5 = None, 0.0, 0
    for rank, nid in enumerate(order, 1):
        text = _norm(qa.graph.nodes[nid].text)
        new = {g for g in gold if g in text} - seen
        if new and first is None:
            first = rank
        if rank <= NDCG_K:
            dcg += len(new) / math.log2(rank + 1)
        if rank <= 5:
            hit5 += len(new)
        seen |= new
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(len(gold), NDCG_K) + 1))
    return {"hit_at_1": float(first == 1), "mrr": 1 / first if first else 0.0,
            f"ndcg_at_{NDCG_K}": min(1.0, dcg / ideal), "recall_at_5": hit5 / len(gold)}


def _auc(positives: List[float], negatives: List[float]) -> Optional[float]:
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return round(wins / (len(positives) * len(negatives)), 3)


def _mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def summarize(rows: List[CaseResult]) -> Dict[str, float]:
    answerable = [r for r in rows if r.answerable and r.recall is not None]
    unanswerable = [r for r in rows if not r.answerable]
    return {
        "cases": len(rows),
        "planner_accuracy": _mean([float(r.type_correct) for r in rows]),
        "evidence_recall": _mean([r.recall for r in answerable]),
        "full_recall_rate": _mean([float(r.recall == 1.0) for r in answerable]),
        "hit_at_1": _mean([float(r.hit_at_1) for r in answerable]),
        "mrr": _mean([r.ranking["scored"]["mrr"] for r in answerable]),
        f"ndcg_at_{NDCG_K}": _mean([r.ranking["scored"][f"ndcg_at_{NDCG_K}"] for r in answerable]),
        "evidence_precision": _mean([r.precision for r in answerable]),
        f"embedding_recall_at_{EMBEDDING_RECALL_K}": _mean([r.embedding_recall for r in answerable]),
        "candidate_recall": _mean([r.candidate_recall for r in answerable]),
        "expanded_recall": _mean([r.expanded_recall for r in answerable]),
        "context_recall": _mean([r.context_recall for r in answerable if r.context_recall is not None]),
        "mean_candidates": _mean([float(r.candidate_count) for r in rows]),
        "mean_subgraph": _mean([float(r.subgraph_size) for r in rows]),
        "primary_precision": _mean([r.primary_precision for r in answerable
                                    if r.primary_precision is not None]),
        "pack_context_recall": _mean([r.pack_context_recall for r in answerable
                                      if r.pack_context_recall is not None]),
        "mean_pack_size": _mean([float(r.pack_size) for r in answerable]),
        "mean_pack_words": _mean([float(r.pack_words) for r in answerable]),
        "abstention_rate": _mean([float(r.abstained) for r in unanswerable]),
        "false_abstention_rate": _mean([float(r.abstained) for r in answerable]),
        "confidence_auc": _auc([_conf(r) for r in rows if r.answerable and _conf(r) is not None],
                               [_conf(r) for r in unanswerable if _conf(r) is not None]),
    }


def _conf(row: CaseResult) -> Optional[float]:
    """Confidence for AUC: anchored questions are certain."""
    return 1.0 if row.anchor else row.confidence


def summarize_ranking(rows: List[CaseResult]) -> Dict[str, Dict[str, float]]:
    """Mean ranking metrics per ordering over answerable cases."""
    answerable = [r for r in rows if r.answerable and r.recall is not None]
    orders = list(dict.fromkeys(o for r in answerable for o in r.ranking))
    return {o: {m: _mean([r.ranking[o][m] for r in answerable if o in r.ranking])
                for m in RANKING_METRICS} for o in orders}


def _print_ranking(table: Dict[str, Dict[str, float]]) -> None:
    print(f"{'ordering':<12}" + "".join(f"{m:>14}" for m in RANKING_METRICS))
    for order in ["scored", "fused"] + [o for o in table if o not in ("scored", "fused")]:
        if order in table:
            print(f"{order:<12}" + "".join(f"{table[order][m]:>14}" for m in RANKING_METRICS))


def summarize_sources(rows: List[CaseResult]) -> Dict[str, Dict[int, float]]:
    """Mean recall@k per retrieval method over answerable cases."""
    answerable = [r for r in rows if r.answerable and r.recall is not None]
    methods = list(dict.fromkeys(key.split("@")[0] for r in answerable for key in r.source_recall))
    return {m: {k: _mean([r.source_recall.get(f"{m}@{k}", 0.0) for r in answerable])
                for k in SOURCE_RECALL_KS} for m in methods}


def _print_sources(table: Dict[str, Dict[int, float]]) -> None:
    print(f"{'method':<12}" + "".join(f"{'recall@' + str(k):>12}" for k in SOURCE_RECALL_KS))
    for method, by_k in table.items():
        print(f"{method:<12}" + "".join(f"{by_k[k]:>12}" for k in SOURCE_RECALL_KS))


def evaluate(cases: List[EvalCase], embedder: Optional[Embedder] = None,
             candidate_budget: Optional[int] = None, expansion_budget: Optional[int] = None,
             **options) -> Dict:
    """``options`` go to ``DocumentQA``: index options, ``weights``, ``calibration``,
    ``parser``."""
    systems: Dict[str, DocumentQA] = {}
    rows = []
    for case in cases:
        if case.document not in systems:
            systems[case.document] = DocumentQA.from_file(case.document, embedder=embedder,
                                                          candidate_budget=candidate_budget,
                                                          expansion_budget=expansion_budget,
                                                          **options)
        rows.append(evaluate_case(systems[case.document], case))
    return {"summary": summarize(rows), "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--verbose", action="store_true", help="show per-case results")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--embedder", default="tfidf", help="'tfidf', 'st:<model>' or 'auto'")
    parser.add_argument("--context", action="store_true",
                        help="prefix each node's section path to its embedding text")
    parser.add_argument("--scaling", default="minmax", choices=["raw", "minmax"])
    parser.add_argument("--compare", nargs="+", metavar="EMBEDDER",
                        help="summary table for several embedders, with and without context")
    parser.add_argument("--budgets", nargs="+", type=int, metavar="N",
                        help="candidate-budget sweep: stage recall at each budget")
    parser.add_argument("--expansion-budgets", nargs="+", type=int, metavar="N",
                        help="expansion-budget sweep: context recall vs subgraph size")
    parser.add_argument("--sources", action="store_true",
                        help="recall@k of each retrieval method alone vs the fused candidate list")
    parser.add_argument("--planner", action="store_true",
                        help="dataset is a planner set (eval/planner_v1.jsonl): score the planner only")
    parser.add_argument("--ranking", action="store_true",
                        help="hit@1 / MRR / nDCG of the scored order vs the fused order and each method")
    parser.add_argument("--weights", metavar="JSON", help="scoring weight overrides (JSON object)")
    parser.add_argument("--scorer", default="linear",
                        help="'linear', 'gnn' (models/coarse_gnn_v2/supervised) or 'gnn:<model dir>'")
    parser.add_argument("--abstain", action="store_true",
                        help="return no evidence when the calibrated confidence is insufficient")
    parser.add_argument("--fit-calibration", action="store_true",
                        help="fit the abstention calibration for --embedder on this dataset and save it")
    parser.add_argument("--format", choices=["md", "pdf", "docx", "html"],
                        help="evaluate on this rendering of each document (examples/*.<format>)")
    parser.add_argument("--parser", default="native", choices=["native", "docling"])
    parser.add_argument("--keep", type=float, default=1.0,
                        help="with --fit-calibration: share of answerable questions the threshold keeps")
    args = parser.parse_args()
    options = {"weights": load_weights(args.weights) if args.weights else None,
               "abstain": args.abstain, "parser": args.parser}
    if args.scorer != "linear":
        from .gnn.scorer import load_scorer

        options["scorer"] = load_scorer(args.scorer)
    if args.fit_calibration:
        options["calibration"] = None
    if args.planner:
        _print_planner(evaluate_planner(args.dataset, load_embedder(args.embedder)), args.verbose)
        return
    cases = load_cases(args.dataset)
    if args.format:
        for case in cases:
            case.document = str(Path(case.document).with_suffix("." + args.format))

    if args.compare:
        _compare(cases, args.compare, args.scaling)
        return
    if args.budgets or args.expansion_budgets:
        embedder = load_embedder(args.embedder)
        if args.budgets:
            keys = ["candidate_recall", "expanded_recall", "evidence_recall", "hit_at_1",
                    "evidence_precision", "mean_candidates"]
            option, budgets = "candidate_budget", args.budgets
        else:
            keys = ["expanded_recall", "context_recall", "mean_subgraph", "evidence_recall",
                    "hit_at_1", "evidence_precision"]
            option, budgets = "expansion_budget", args.expansion_budgets
        print(f"{'budget':>8}" + "".join(f"{k:>20}" for k in keys))
        for budget in budgets:
            summary = evaluate(cases, embedder, contextual=args.context, semantic_scaling=args.scaling,
                               **{option: budget}, **options)["summary"]
            print(f"{budget:>8}" + "".join(f"{summary[k]:>20}" for k in keys))
        return
    embedder = load_embedder(args.embedder)
    report = evaluate(cases, embedder, contextual=args.context, semantic_scaling=args.scaling,
                      **options)
    if args.fit_calibration:
        _fit(report["cases"], embedder.name, args.dataset, args.keep)
        return
    if args.ranking:
        _print_ranking(summarize_ranking(report["cases"]))
        print()
    if args.sources:
        _print_sources(summarize_sources(report["cases"]))
        print()
    if args.json:
        print(json.dumps({"summary": report["summary"],
                          "cases": [asdict(r) for r in report["cases"]]}, indent=2))
        return
    if args.verbose:
        for r in report["cases"]:
            flags = "" if r.type_correct else f"  [planner: {r.predicted_type}]"
            if r.answerable:
                status = f"recall={r.recall:.2f} prec={r.precision:.2f} hit@1={int(r.hit_at_1)}"
            else:
                status = "abstained" if r.abstained else f"did NOT abstain ({r.pack_size} items)"
            if r.answerable and r.abstained:
                status = "WRONGLY abstained"
            conf = f"p={r.confidence:.2f}" if r.confidence is not None else ""
            print(f"{r.id:<8} {status:<36} {r.anchor or conf:<16} {r.question}{flags}")
            for missed in r.missed:
                print(f"         missed: {missed}")
        print()
    for key, value in report["summary"].items():
        print(f"{key:<20} {value}")


def _fit(rows: List[CaseResult], embedder: str, dataset: str, keep: float) -> None:
    from .scoring import EvidenceConfidence

    confidences = [EvidenceConfidence(r.margin, r.coverage, r.anchor) for r in rows]
    calibration = fit_calibration(confidences, [r.answerable for r in rows], embedder,
                                  Path(dataset).name, keep)
    save_calibration(calibration)
    probs = [(calibration.probability(c.margin, c.coverage), r) for c, r in zip(confidences, rows)]
    kept = [p >= calibration.threshold or r.anchor for p, r in probs if r.answerable]
    abstained = [p < calibration.threshold and not r.anchor for p, r in probs if not r.answerable]
    print(f"saved calibration for {embedder}: coef={calibration.coef} "
          f"intercept={calibration.intercept:.3f} threshold={calibration.threshold}")
    print(f"on {Path(dataset).name}: keeps {sum(kept)}/{len(kept)} answerable, "
          f"abstains on {sum(abstained)}/{len(abstained)} unanswerable")


def evaluate_planner(path: str, embedder: Optional[Embedder] = None) -> Dict:
    """Score the query planner alone: type, focus term, target sections, sides."""
    from .document_graph import DocumentGraph
    from .index import GraphIndex
    from .query_planner import QUERY_TYPES, plan_query

    base = Path(path).resolve().parent
    indexes: Dict[str, GraphIndex] = {}
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        doc = str((base / case["document"]).resolve())
        if doc not in indexes:
            indexes[doc] = GraphIndex(DocumentGraph.from_file(doc), embedder)
        index = indexes[doc]
        plan = plan_query(case["question"], index.graph, index)
        numbers = {(index.graph.nodes[t].metadata.get("number") or "").lower()
                   for t in plan.target_sections}
        row = {"id": case["id"], "question": case["question"], "expected": case["query_type"],
               "predicted": plan.query_type, "strategy": plan.strategy,
               "type_ok": plan.query_type == case["query_type"]}
        if "focus" in case:
            row["focus_ok"] = plan.focus_terms[:1] == [case["focus"]]
        if "sections" in case:
            want = [s.lower() for s in case["sections"]]
            row["section_recall"] = sum(s in numbers for s in want) / len(want)
            row["section_precision"] = (sum(n in want for n in numbers) / len(numbers)) if numbers else 0.0
            row["sections"] = sorted(numbers)
        if "sides" in case:
            row["sides_ok"] = len(plan.sides) == case["sides"]
        rows.append(row)

    def mean(key, subset=None):
        values = [float(r[key]) for r in (subset or rows) if key in r]
        return round(sum(values) / len(values), 3) if values else None

    summary = {"cases": len(rows), "type_accuracy": mean("type_ok"),
               "focus_accuracy": mean("focus_ok"), "section_recall": mean("section_recall"),
               "section_precision": mean("section_precision"), "sides_accuracy": mean("sides_ok")}
    for qt in QUERY_TYPES:
        subset = [r for r in rows if r["expected"] == qt]
        if subset:
            summary[f"accuracy[{qt}]"] = mean("type_ok", subset)
    return {"summary": summary, "cases": rows}


def _print_planner(report: Dict, verbose: bool) -> None:
    if verbose:
        for r in report["cases"]:
            problems = []
            if not r["type_ok"]:
                problems.append(f"type {r['predicted']} (want {r['expected']})")
            if r.get("focus_ok") is False:
                problems.append("focus term")
            if r.get("section_recall", 1) < 1:
                problems.append(f"sections {r['sections']}")
            if r.get("sides_ok") is False:
                problems.append("sides")
            status = "ok" if not problems else "; ".join(problems)
            print(f"{r['id']:<8} {r['strategy']:<18} {status:<55} {r['question']}")
        print()
    for key, value in report["summary"].items():
        print(f"{key:<38} {value}")


def _compare(cases: List[EvalCase], specs: List[str], scaling: str) -> None:
    rows = []
    for spec in specs:
        embedder = load_embedder(spec)
        for contextual in (False, True):
            summary = evaluate(cases, embedder, contextual=contextual,
                               semantic_scaling=scaling)["summary"]
            rows.append((f"{spec}{' +context' if contextual else ''}", summary))
    keys = [k for k in rows[0][1] if k not in ("cases", "planner_accuracy")]
    width = max(len(name) for name, _ in rows) + 2
    print(" " * width + "  ".join(f"{k[:18]:>18}" for k in keys))
    for name, summary in rows:
        print(f"{name:<{width}}" + "  ".join(f"{summary[k]:>18}" for k in keys))


if __name__ == "__main__":
    main()
