from pathlib import Path

import pytest

from gnn_pruner import DocumentGraph, DocumentQA
from gnn_pruner.index import GraphIndex
from gnn_pruner.models import (
    CROSS_REFERENCE,
    DEFINITION_LINK,
    SAME_SECTION,
    SEMANTIC_SIMILARITY,
    Edge,
)
from gnn_pruner.store import GraphStore

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SAAS = EXAMPLES / "saas_agreement.md"


def _refs(graph, node):
    return sorted(n.metadata.get("number") for n, _, _ in
                  graph.neighbors(node.id, (CROSS_REFERENCE,), "out"))


def _paragraph(graph, startswith):
    return next(n for n in graph.content_nodes(("paragraph", "clause")) if n.text.startswith(startswith))


# --------------------------------------------------------------- node types

def test_title_heading_becomes_document_title():
    graph = DocumentGraph.from_text("# Supply Agreement\n\n## 1. Scope\n\nBody text here.\n")
    assert graph.title == "Supply Agreement"
    assert [s.title for s in graph.sections()] == ["1. Scope"]
    assert graph.path(graph.content_nodes(("paragraph",))[0].id) == "1. Scope"


def test_title_heading_with_its_own_body_stays_a_section():
    graph = DocumentGraph.from_text("# Introduction\n\nIntro text.\n\n## Details\n\nMore text.\n")
    assert [s.title for s in graph.sections()] == ["Introduction", "Details"]


def test_chapter_section_subsection_kinds():
    graph = DocumentGraph.from_text(
        "# Handbook\n\n## Chapter 1 Introduction\n\n### Purpose\n\nWhy this exists.\n\n"
        "#### Details\n\nFine print.\n\n## Chapter 2 Rules\n\n### Conduct\n\nBe kind.\n\n"
        "## Glossary\n\nTerms used.\n")
    kinds = {s.title: s.node_type for s in graph.sections()}
    assert kinds == {"Chapter 1 Introduction": "chapter", "Purpose": "section",
                     "Details": "subsection", "Chapter 2 Rules": "chapter",
                     "Conduct": "section", "Glossary": "section"}


def test_numbered_paragraph_is_a_clause_and_resolves_references():
    graph = DocumentGraph.from_text(
        "## 4. Delivery\n\n4.1.1 The Supplier shall deliver the goods by the delivery date.\n\n"
        "4.1.2 Late delivery is subject to clause 4.1.1 and Sections 5 and 6.\n\n"
        "## 5. Payment\n\nPay on time.\n\n## 6. Risk\n\nRisk passes on delivery.\n")
    first = _paragraph(graph, "4.1.1")
    second = _paragraph(graph, "4.1.2")
    assert (first.node_type, first.metadata["number"]) == ("clause", "4.1.1")
    assert _refs(graph, second) == ["4.1.1", "5", "6"]


def test_inline_list_becomes_clause_children():
    graph = DocumentGraph.from_file(str(SAAS))
    para = _paragraph(graph, "The Customer shall not:")
    clauses = graph.children(para.id)
    assert [(c.node_type, c.metadata["label"], c.metadata["number"]) for c in clauses] == [
        ("clause", "a", "2(a)"), ("clause", "b", "2(b)"), ("clause", "c", "2(c)")]
    assert clauses[2].text == "(c) use the Platform to build a competing product."
    for c in clauses:
        assert " ".join(graph.source_text[c.start:c.end].split()) == c.text


def test_clause_reference_and_section_fallback():
    graph = DocumentGraph.from_file(str(SAAS))
    suspend = _paragraph(graph, "The Provider may suspend")
    target = graph.neighbors(suspend.id, (CROSS_REFERENCE,), "out")[0][0]
    assert target.metadata["number"] == "2(c)"
    assert graph.resolve_reference("2(z)").title == "2. Subscription"  # falls back
    assert graph.resolve_reference("99") is None


def test_lone_parenthetical_and_inline_reference_are_not_split():
    graph = DocumentGraph.from_text(
        "## 1. Terms\n\nThe fee (a) is due monthly.\n\n"
        "## 2. Other\n\nSee Section 1(a) and (b) for details.\n")
    assert not graph.content_nodes(("clause",))


def test_self_reference_is_ignored():
    graph = DocumentGraph.from_file(str(SAAS))
    para = _paragraph(graph, "The limitation in this Section 8")
    assert _refs(graph, para) == ["4"]


def test_definitions_inside_clauses():
    graph = DocumentGraph.from_text(
        "## 1. Definitions\n\nIn this Agreement: (a) \"Fees\" means the charges in Schedule 1; "
        "(b) \"Term\" means one year.\n\n## 2. Payment\n\nFees are due within the Term.\n")
    fees, term = graph.definitions["fees"], graph.definitions["term"]
    assert graph.nodes[fees].node_type == graph.nodes[term].node_type == "clause"
    usage = _paragraph(graph, "Fees are due")
    linked = {n.id for n, _, _ in graph.neighbors(usage.id, (DEFINITION_LINK,), "out")}
    assert linked == {fees, term}


# -------------------------------------------------------------------- edges

def test_same_section_edges_connect_section_units():
    graph = DocumentGraph.from_file(str(SAAS))
    section = graph.lookup("1. Definitions")
    units = [c.id for c in graph.children(section.id)]
    assert len(units) == 4
    pairs = {frozenset((e.source, e.target)) for e in graph.edges if e.edge_type == SAME_SECTION}
    assert sum(1 for p in pairs if p <= set(units)) == 6  # 4 choose 2


def test_comparison_pulls_whole_sections():
    result = DocumentQA.from_file(str(SAAS)).ask("Compare the renewal and termination provisions")
    texts = " ".join(i.text for i in result.evidence.items)
    assert "renews automatically" in texts
    assert "terminate for convenience on thirty days' notice" in texts


# ------------------------------------------------------------ storage / checks

@pytest.mark.parametrize("name", ["sample_contract.md", "saas_agreement.md"])
def test_examples_validate(name):
    assert DocumentGraph.from_file(str(EXAMPLES / name)).validate() == []


def test_validate_reports_corruption():
    graph = DocumentGraph.from_file(str(SAAS))
    graph.edges.append(Edge("p1", "missing", CROSS_REFERENCE))
    para = graph.content_nodes(("paragraph",))[0]
    para.end = para.start + 3
    problems = graph.validate()
    assert any("dangling" in p for p in problems)
    assert any(para.id in p and "source offsets" in p for p in problems)


def test_json_round_trip_preserves_graph_and_answers(tmp_path):
    graph = DocumentGraph.from_file(str(SAAS))
    GraphIndex(graph)  # adds semantic_similarity edges, which must survive too
    path = tmp_path / "graph.json"
    graph.save(str(path))
    loaded = DocumentGraph.load(str(path))

    assert loaded.to_dict() == graph.to_dict()
    assert loaded.stats()["edges"][SEMANTIC_SIMILARITY] > 0
    question = "Can the Provider suspend access if the Customer builds a competing product?"
    original = [i.text for i in DocumentQA(graph).ask(question).evidence.items]
    assert [i.text for i in DocumentQA(loaded).ask(question).evidence.items] == original


def test_load_rejects_unknown_version():
    data = DocumentGraph.from_file(str(SAAS)).to_dict()
    data["format_version"] = 999
    with pytest.raises(ValueError, match="version"):
        DocumentGraph.from_dict(data)


def test_graph_store_round_trip_and_queries(tmp_path):
    store = GraphStore(str(tmp_path / "graphs.db"))
    graph = DocumentGraph.from_file(str(SAAS))
    doc_id = store.put(graph, doc_id="saas")
    other = store.put(DocumentGraph.from_file(str(EXAMPLES / "sample_contract.md")))

    assert store.get("saas").to_dict() == graph.to_dict()
    assert {d["doc_id"] for d in store.list_documents()} == {doc_id, other}

    clauses = store.find_nodes("saas", node_type="clause")
    assert [c.metadata["number"] for c in clauses] == ["2(a)", "2(b)", "2(c)"]
    suspend = store.find_nodes("saas", contains="suspend access")[0]
    assert [n.id for n in store.neighbors("saas", suspend.id, CROSS_REFERENCE)] == [clauses[2].id]
    assert suspend.id in {n.id for n in store.neighbors("saas", clauses[2].id, CROSS_REFERENCE, "in")}

    store.delete("saas")
    assert store.find_nodes("saas") == []
    with pytest.raises(KeyError):
        store.get("saas")


def test_store_skips_reparsing_unchanged_files(tmp_path, monkeypatch):
    store = GraphStore(str(tmp_path / "graphs.db"))
    doc = tmp_path / "contract.md"
    doc.write_text(SAAS.read_text())
    first = DocumentQA.from_file(str(doc), store=store)

    def boom(path):
        raise AssertionError("should not re-parse an unchanged file")

    monkeypatch.setattr(DocumentGraph, "from_file", classmethod(lambda cls, p: boom(p)))
    second = DocumentQA.from_file(str(doc), store=store)
    assert second.graph.to_dict()["nodes"] == first.graph.to_dict()["nodes"]

    monkeypatch.undo()
    doc.write_text(SAAS.read_text() + "\n## 11. Notices\n\nNotices must be in writing.\n")
    changed = DocumentQA.from_file(str(doc), store=store)
    assert changed.graph.lookup("11. Notices") is not None
    assert len(store.list_documents()) == 2


def test_networkx_export():
    nx = pytest.importorskip("networkx")
    graph = DocumentGraph.from_file(str(SAAS))
    g = graph.to_networkx()
    assert g.number_of_nodes() == len(graph.nodes)
    assert g.number_of_edges() == len(graph.edges)
    assert nx.is_tree(nx.DiGraph([(e.source, e.target) for e in graph.edges
                                  if e.edge_type == "parent_child"]))
