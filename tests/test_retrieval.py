"""Phase 5: hybrid candidate retrieval."""

from pathlib import Path

import pytest

from gnn_pruner import DocumentGraph
from gnn_pruner.evaluation import SOURCE_RECALL_KS, evaluate, load_cases, summarize_sources
from gnn_pruner.index import GraphIndex
from gnn_pruner.query_planner import plan_query
from gnn_pruner.retrieval import (
    QUOTA_SOURCES,
    SOURCE_QUOTA,
    QueryContext,
    graph_list,
    retrieve,
    retrieve_candidates,
    structural_list,
)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample_contract.md"
DATASET = ROOT / "eval" / "contracts_v1.jsonl"

CHAPTERS = """# Handbook

## 1. Security

### 1.1 Encryption

Data at rest uses AES-256 keys rotated yearly.

### 1.2 Access Reviews

Accounts are reviewed every quarter by the owner.

## 2. Privacy

### 2.1 Retention

Logs are deleted after ninety days.

### 2.2 Encryption

Personal data is pseudonymised before export.
"""


@pytest.fixture(scope="module")
def index():
    return GraphIndex(DocumentGraph.from_file(str(SAMPLE)))


def _context(index, question):
    return QueryContext.build(index, plan_query(question, index.graph, index))


def test_structural_list_uses_the_chapter_path():
    index = GraphIndex(DocumentGraph.from_text(CHAPTERS))
    graph = index.graph
    ranked = structural_list(index, _context(index, "privacy encryption rules"), 10)
    sections = [graph.section_of(n).title for n in ranked]
    # Both "Encryption" subsections match on title; the chapter path breaks the tie.
    assert sections[0] == "2.2 Encryption"
    assert "1.1 Encryption" in sections
    assert all(graph.nodes[n].node_type == "paragraph" for n in ranked)


def test_structural_list_is_empty_without_a_title_match(index):
    assert structural_list(index, _context(index, "zebra quokka"), 10) == []


def test_graph_list_follows_references_and_definitions(index):
    graph = index.graph
    cap = graph.children(graph.lookup("4.1 Cap on Liability").id)[0]
    indemnity = graph.children(graph.lookup("5. Indemnity").id)[0]
    allowed = {"paragraph", "clause", "sentence"}
    found = graph_list(graph, [cap.id, indemnity.id], allowed)
    exclusions = graph.lookup("4.2 Exclusions from the Cap")
    assert any(graph.section_of(n).id == exclusions.id for n in found)   # cited clause
    assert graph.definitions["losses"] in found                          # used term
    assert cap.id not in found and indemnity.id not in found             # seeds excluded


def test_graph_list_lifts_a_sentence_to_its_paragraph(index):
    graph = index.graph
    sentence = next(n for n in graph.nodes.values() if n.node_type == "sentence")
    assert sentence.parent_id in graph_list(graph, [sentence.id], {"paragraph", "sentence"})


def test_candidates_are_bounded_ordered_and_explained(index):
    ctx = _context(index, "When must the Customer pay invoices?")
    candidates = retrieve_candidates(index, ctx)
    plan = ctx.plan

    pinned = {n for n, r in candidates.items() if any(x.startswith(("plan:", "definition:")) for x in r)}
    quota = {n for s, ranked in ctx.sources.items() if s in QUOTA_SOURCES for n in ranked[:SOURCE_QUOTA]}
    assert len(candidates) <= max(plan.max_candidates, len(pinned | quota))
    assert quota <= set(candidates)
    for nid, reasons in candidates.items():
        assert index.graph.nodes[nid].node_type in plan.target_types
        assert "fused" in reasons
    # Ordered by fused score, and every source's ranking is recorded.
    order = list(candidates)
    assert ctx.sources["fused"] == order
    assert [ctx.fused[n] for n in order] == sorted(ctx.fused.values(), reverse=True)
    assert {"lexical", "phrase", "semantic", "structural", "graph", "fused"} <= set(ctx.sources)


def test_planner_targets_are_pinned(index):
    trace = retrieve(index, "compare sections 4.1 and 4.2", candidate_budget=1)
    graph = index.graph
    for number in ("4.1", "4.2"):
        section = graph.section_by_number(number)
        member = graph.children(section.id)[0]
        assert member.id in trace.candidates
        assert any(r.startswith("plan:") for r in trace.candidates[member.id])


def test_definition_question_pins_the_definition(index):
    trace = retrieve(index, "What does Misconduct mean?", candidate_budget=1)
    assert index.graph.definitions["misconduct"] in trace.candidates


def test_trace_exposes_source_rankings(index):
    trace = retrieve(index, "What is the cap on liability?")
    assert trace.sources["fused"] == list(trace.candidates)


def test_fused_candidates_recall_at_least_every_single_method():
    """Phase 5 acceptance: the fused candidate set finds at least as much gold
    evidence as any single method's top ``SOURCE_QUOTA`` (guaranteed by the quota)."""
    report = evaluate(load_cases(str(DATASET)))
    table = summarize_sources(report["cases"])
    candidate_recall = report["summary"]["candidate_recall"]
    for method in QUOTA_SOURCES:
        assert candidate_recall >= table[method][SOURCE_RECALL_KS[0]], method
    assert report["summary"]["mean_candidates"] <= 50
