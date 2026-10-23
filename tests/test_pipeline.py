from pathlib import Path

import pytest

from gnn_pruner import DocumentGraph, DocumentQA
from gnn_pruner.models import CROSS_REFERENCE, DEFINITION_LINK
from gnn_pruner.parser import parse_text
from gnn_pruner.query_planner import COMPARISON, DEFINITION, SUMMARY, plan_query

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_contract.md"


@pytest.fixture(scope="module")
def qa():
    return DocumentQA.from_file(str(SAMPLE))


def test_parser_tracks_offsets_and_hierarchy():
    text = "# Title\n\nFirst sentence here. Second sentence here.\n\n2.1 Numbered Heading\n\nBody text.\n"
    nodes = {n.id: n for n in parse_text(text)}
    for node in nodes.values():
        if node.node_type in ("paragraph", "sentence"):
            assert " ".join(text[node.start:node.end].split()) == node.text
    numbered = next(n for n in nodes.values() if n.title == "2.1 Numbered Heading")
    assert numbered.metadata["number"] == "2.1"
    sentences = [n for n in nodes.values() if n.node_type == "sentence"]
    assert [s.text for s in sentences] == ["First sentence here.", "Second sentence here."]


def test_graph_links_definitions_and_cross_references():
    graph = DocumentGraph.from_text(SAMPLE.read_text())
    assert {"losses", "misconduct", "confidential information"} <= set(graph.definitions)

    cap = graph.lookup("4.1 Cap on Liability")
    cap_para = graph.children(cap.id)[0]
    refs = graph.neighbors(cap_para.id, (CROSS_REFERENCE,), "out")
    assert [n.metadata.get("number") for n, _, _ in refs] == ["4.2"]

    indemnity = graph.children(graph.lookup("5. Indemnity").id)[0]
    defs = {n.id for n, _, _ in graph.neighbors(indemnity.id, (DEFINITION_LINK,), "out")}
    assert graph.definitions["losses"] in defs
    assert graph.definitions["misconduct"] in defs


@pytest.mark.parametrize("query,expected", [
    ("what is the definition of Losses?", DEFINITION),
    ("What does Misconduct mean", DEFINITION),
    ("compare sections 4.1 and 4.2", COMPARISON),
    ("summarize the liability terms", SUMMARY),
])
def test_query_planner(query, expected):
    assert plan_query(query).query_type == expected


def test_definition_query_puts_definition_first(qa):
    result = qa.ask("what is the definition of Losses")
    assert result.evidence.items[0].text.startswith('"Losses" means')
    # Definition dependencies must be real definitions, not paragraphs that use the term.
    for item in result.evidence.items:
        if item.role == "definition":
            assert " means " in item.text


def test_pruning_keeps_referenced_clause(qa):
    result = qa.ask("what is the cap on liability")
    texts = [i.text for i in result.evidence.items]
    assert any("total liability" in t for t in texts)
    # 4.1 says "Except as set out in Section 4.2", so 4.2 must survive pruning.
    assert any("does not apply" in t for t in texts)


def test_evidence_pack_is_bounded_and_cited(qa):
    result = qa.ask("is the indemnity subject to the liability cap")
    pack = result.evidence
    assert 0 < len(pack.items) <= result.plan.max_evidence + 4
    assert all(i.section_path for i in pack.items)
    assert len({i.node_id for i in pack.items}) == len(pack.items)
    prompt = pack.render_prompt()
    assert "[E1]" in prompt and "Answer only from the evidence" in prompt


def test_answer_uses_llm_when_provided():
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return "stub answer"

    qa = DocumentQA.from_file(str(SAMPLE), llm=fake_llm)
    assert qa.ask("how long do confidentiality obligations last").answer == "stub answer"
    assert "five years" in seen["prompt"]
