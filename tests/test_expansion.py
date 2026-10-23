"""Phase 6: bounded, best-first graph expansion."""

from pathlib import Path

import pytest

from gnn_pruner import DocumentGraph
from gnn_pruner.evaluation import gold_dependencies
from gnn_pruner.index import GraphIndex
from gnn_pruner.models import Node
from gnn_pruner.query_planner import QueryPlan, plan_query
from gnn_pruner.retrieval import continues, expand_subgraph, retrieve

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample_contract.md"
SCALE = ROOT / "examples" / "scale" / "saas_agreement.md"


@pytest.fixture(scope="module")
def graph():
    return DocumentGraph.from_file(str(SAMPLE))


def _para(graph, title, i=0):
    return [c for c in graph.children(graph.lookup(title).id) if c.is_content][i]


def _plan(**overrides):
    plan = QueryPlan("q", "fact_lookup")
    for key, value in overrides.items():
        setattr(plan, key, value)
    return plan


def _added_content(graph, subgraph, anchors):
    return [n for n in subgraph if n not in anchors and graph.nodes[n].is_content]


@pytest.mark.parametrize("first,second,expected", [
    ("The Customer shall:", "Pay all fees.", True),                # list intro
    ("Fees are due monthly.", "Such fees exclude tax.", True),      # points back
    ("Fees are due monthly.", "provided that notice is given.", True),
    ("Fees are due monthly.", "(b) Taxes are extra.", True),       # list item
    ("Fees are due monthly.", "Either party may terminate.", False),
])
def test_continuation_cues(first, second, expected):
    assert continues(Node("a", "paragraph", first), Node("b", "paragraph", second)) is expected


def test_continuation_recovers_the_next_sentence(graph):
    """para-06: the answer follows a paragraph that opens 'These obligations...'."""
    secret = _para(graph, "3. Confidentiality")
    # The definition the anchor uses comes first, then the continuation.
    sub = expand_subgraph(graph, {secret.id: {"lexical"}}, _plan(max_expanded=2))
    added = _added_content(graph, sub, {secret.id})
    assert graph.definitions["confidential information"] in added
    follow = next(n for n in added if "five years" in graph.nodes[n].text)
    assert any(r.startswith("expansion:continuation:") for r in sub[follow])
    assert sub.priority[follow] < sub.priority[graph.definitions["confidential information"]]


def test_budget_caps_content_but_not_sections(graph):
    cap = _para(graph, "4.1 Cap on Liability")
    for budget in (0, 1, 3):
        sub = expand_subgraph(graph, {cap.id: {"lexical"}}, _plan(max_expanded=budget))
        assert len(_added_content(graph, sub, {cap.id})) <= budget
        assert graph.lookup("4.1 Cap on Liability").id in sub   # its section, always
    assert expand_subgraph(graph, {cap.id: {"lexical"}}, _plan(max_expanded=0)).skipped > 0


def test_references_and_definitions_beat_plain_neighbours(graph):
    cap = _para(graph, "4.2 Exclusions from the Cap")
    sub = expand_subgraph(graph, {cap.id: {"lexical"}}, _plan(max_expanded=2))
    kinds = {r.split(":")[1] for n in _added_content(graph, sub, {cap.id}) for r in sub[n]}
    assert kinds <= {"definition", "reference"}


def test_paths_lead_back_to_an_anchor_within_the_radius(graph):
    cap = _para(graph, "4.1 Cap on Liability")
    for radius in (1, 2):
        sub = expand_subgraph(graph, {cap.id: {"lexical"}},
                              _plan(expansion_radius=radius, max_expanded=50))
        for nid, path in sub.paths.items():
            assert path[0] == cap.id and path[-1] == nid
            # Content is at most `radius` hops out; a section may sit one hop beyond.
            limit = radius + (0 if graph.nodes[nid].is_content else 1)
            assert len(path) - 1 <= limit, (nid, path)
            for reason in sub[nid]:
                if len(path) > 1 and reason.startswith("expansion:"):
                    assert reason.split(":")[2] in sub
    one = expand_subgraph(graph, {cap.id: {"lexical"}}, _plan(expansion_radius=1, max_expanded=50))
    two = expand_subgraph(graph, {cap.id: {"lexical"}}, _plan(expansion_radius=2, max_expanded=50))
    assert set(one) < set(two)


def test_same_section_only_when_planned(graph):
    first = _para(graph, "1. Definitions")
    siblings = {c.id for c in graph.children(first.parent_id) if c.is_content} - {first.id}
    plain = expand_subgraph(graph, {first.id: {"lexical"}}, _plan(max_expanded=50))
    whole = expand_subgraph(graph, {first.id: {"lexical"}},
                            _plan(max_expanded=50, expand_same_section=True))
    assert siblings <= set(whole)
    assert not any("same_section" in r for rs in plain.values() for r in rs)


def test_expansion_stays_local_on_a_long_document():
    index = GraphIndex(DocumentGraph.from_file(str(SCALE)))
    trace = retrieve(index, "What is the uptime commitment?")
    content = sum(index.graph.nodes[n].is_content for n in trace.subgraph)
    assert content <= len(trace.candidates) + trace.plan.max_expanded
    assert content < len(index.ids) / 4


def test_planner_sets_expansion_budget_per_strategy():
    assert plan_query("What is the cap on liability?").max_expanded == 10
    assert plan_query("summarize the liability terms").max_expanded == 30
    assert plan_query("compare sections 4.1 and 4.2").max_expanded == 30


def test_gold_dependencies(graph):
    needed = gold_dependencies(graph, ["the cap in section 4.1 does not apply to losses"])
    exclusions = graph.lookup("4.2 Exclusions from the Cap")
    assert exclusions.id in needed                               # its section
    assert _para(graph, "4.1 Cap on Liability").id in needed     # cited clause
    assert graph.definitions["losses"] in needed                 # term it uses
    assert graph.definitions["misconduct"] in needed
