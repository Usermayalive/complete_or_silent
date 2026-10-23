"""Phase 9: topology routing, token budget, rendering, citation map."""

from pathlib import Path

import pytest

from gnn_pruner import DocumentGraph, DocumentQA
from gnn_pruner.evidence import EvidenceItem, build_evidence_pack
from gnn_pruner.parser import normalize_ws
from gnn_pruner.prompt import (PATH, QA, SIBLING, SYNTHESIS, TEMPLATES, analyse, apply_budget,
                               count_tokens)
from gnn_pruner.pruning import EvidenceNode, PrunedSubgraph
from gnn_pruner.query_planner import plan_query

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample_contract.md"
SAAS = ROOT / "examples" / "saas_agreement.md"


@pytest.fixture(scope="module")
def sample():
    return DocumentGraph.from_file(str(SAMPLE))


@pytest.fixture(scope="module")
def saas():
    return DocumentGraph.from_file(str(SAAS))


def node(graph, prefix, kind="paragraph"):
    return next(n.id for n in graph.nodes.values() if n.node_type == kind and n.text.startswith(prefix))


def test_route_qa(sample):
    t = analyse(sample, [node(sample, "The Customer shall pay")])
    assert (t.route, t.cross_refs, t.siblings) == (QA, 0, 1) and t.depth == 2


def test_route_synthesis(sample):
    items = [node(sample, "Except as set out"), node(sample, "The cap in Section 4.1"),
             node(sample, "Claims under this indemnity"), node(sample, "Each party shall keep")]
    t = analyse(sample, items)
    # 4.1 <-> 4.2, the indemnity claim -> 4.2, 4.2 -> Section 3.
    assert t.cross_refs == 3 and t.route == SYNTHESIS


def test_route_sibling(saas):
    convenience = node(saas, "The Customer may terminate")
    items = [node(saas, "The Customer may terminate for convenience", "sentence"),
             node(saas, "No refund is payable", "sentence"),
             node(saas, "On termination, the Provider"), node(saas, "The Provider may suspend")]
    assert all(saas.section_of(i).id == saas.section_of(convenience).id for i in items)
    t = analyse(saas, items)
    assert t.siblings == 4 and t.cross_refs <= 2 and t.route == SIBLING


def test_route_path(saas):
    sentence = node(saas, "Fees may not increase during the initial", "sentence")  # in 9.2, under 9
    t = analyse(saas, [sentence])
    assert t.depth == 4 and t.route == PATH


def _item(label, node_id, role="primary", text="word " * 10, score=1.0, required_by=None):
    return EvidenceItem(label, node_id, role, "s", text, score, 0, 0, required_by=required_by)


def test_budget_keeps_priority_and_drops_orphaned_dependencies():
    items = [_item("E1", "a", score=0.9), _item("E2", "b", score=0.5),
             _item("E3", "d", "definition", required_by="b"), _item("E4", "e", "definition", required_by="a")]
    report = apply_budget(items, 25)  # room for a, its definition, and half of b
    assert [i.node_id for i in items] == ["a", "e"]
    assert set(report.dropped) == {"b", "d"} and report.used == 20
    assert apply_budget([_item("E1", "x")], None).used == count_tokens("word " * 10)


def test_budget_truncates_at_sentence_boundary():
    text = "First sentence here. Second sentence here. Third one here."
    items = [_item("E1", "a", text=text)]
    report = apply_budget(items, count_tokens("First sentence here. Second sentence here.") + 1)
    assert items[0].text == "First sentence here. Second sentence here. …" and items[0].truncated
    assert report.truncated == ["a"]


@pytest.fixture(scope="module")
def pack(sample):
    plan = plan_query("What is the cap on liability?", sample)
    nodes = {nid: sample.nodes[nid] for nid in (node(sample, "Except as set out"), node(sample, "The cap in Section 4.1"),
                                                 sample.definitions["losses"])}
    cap, excl, losses = nodes
    pruned = PrunedSubgraph([EvidenceNode(nodes[cap], "primary", 0.9),
                             EvidenceNode(nodes[excl], "reference", 0.5, required_by=cap),
                             EvidenceNode(nodes[losses], "definition", 0.3, required_by=excl)])
    return build_evidence_pack(sample, plan, pruned, confidence="high", budgets={})


def test_rendering(pack, sample):
    prompt = pack.render_prompt()
    assert prompt == pack.render_prompt()  # deterministic
    assert prompt.startswith("Question: What is the cap on liability?")
    assert "Retrieval confidence that the document answers this: high." in prompt
    # Primaries by score, then supporting items in document order (the definition, then 4.2).
    assert "[E2] 1. Definitions - defines a term used in E3" in prompt
    assert "[E3] 4. Liability > 4.2 Exclusions from the Cap - cited by E1" in prompt
    assert "Cites: -> E3" in prompt  # 4.1 cites 4.2, which is in the pack
    assert "Cites (not in evidence): 3. Confidentiality: Each party shall keep" in prompt  # 4.2 cites Section 3
    assert TEMPLATES[pack.route] in prompt and "Answer only from the evidence" in prompt
    assert [i.label for i in pack.items] == ["E1", "E2", "E3"]


def test_citation_map_points_back_to_the_source(pack, sample):
    cmap = pack.citation_map()
    assert set(cmap) == {"E1", "E2", "E3"}
    for label, where in cmap.items():
        item = next(i for i in pack.items if i.label == label)
        assert where["node_id"] == item.node_id
        assert normalize_ws(sample.source_text[where["start"]:where["end"]]) == item.text


def test_pipeline_packs_are_routed_and_budgeted():
    qa = DocumentQA.from_file(str(SAAS))
    result = qa.ask("What happens if availability falls below the Service Levels?")
    pack = result.evidence
    assert pack.route in (QA, SYNTHESIS, SIBLING, PATH) and pack.topology is not None
    assert pack.confidence is not None and pack.budget is not None and pack.budget.used <= 250 + 50
    again = DocumentQA.from_file(str(SAAS)).ask("What happens if availability falls below the Service Levels?")
    assert again.evidence.render_prompt() == pack.render_prompt()  # deterministic end to end
    unbudgeted = DocumentQA.from_file(str(SAAS), budgets={}).ask(
        "What happens if availability falls below the Service Levels?").evidence
    assert len(unbudgeted.items) >= len(pack.items)
