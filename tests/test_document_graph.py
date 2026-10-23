from gnn_pruner.document_graph import DocumentGraph
from gnn_pruner.retrieval import retrieve_relevant_nodes


def test_document_graph_builds_hierarchy_and_edges():
    text = """
# Introduction

This is the introduction paragraph.

## Definitions

The term indemnity means compensation.

### Indemnity Clause

The indemnity applies to losses caused by breach.

## Liability

This section covers financial liability.
"""

    graph = DocumentGraph.from_text(text)

    assert graph.root is not None
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    assert graph.lookup("Introduction") is not None
    assert graph.lookup("Definitions") is not None
    assert graph.lookup("Indemnity Clause") is not None


def test_retrieval_prefers_structure_and_context():
    text = """
# Introduction

This document explains indemnity and liability.

## Definitions

The term indemnity means compensation for losses.

## Liability

The indemnity clause covers losses caused by breach.

## Exceptions

This section contains standard exclusions and unrelated risk items.
"""

    graph = DocumentGraph.from_text(text)
    results = retrieve_relevant_nodes(graph, "what is the definition of indemnity")

    assert results
    assert any(node.node_type == "paragraph" for node in results[:5])
    assert any("indemnity" in node.text.lower() for node in results[:5])
