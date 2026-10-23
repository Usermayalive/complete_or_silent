"""Phase 8: adaptive pruning and dependency rules."""

from pathlib import Path

import pytest

from gnn_pruner import DocumentGraph, DocumentQA
from gnn_pruner.evaluation import evaluate, load_cases
from gnn_pruner.pruning import (CONTEXT, COUNTERPART, DEFINITION, PRIMARY, QUALIFIER, REFERENCE,
                                PruneConfig, _qualifies, conflicts, default_config, dependencies,
                                prune)
from gnn_pruner.query_planner import plan_query
from gnn_pruner.scoring import ScoredNode

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample_contract.md"
SAAS = ROOT / "examples" / "saas_agreement.md"


@pytest.fixture(scope="module")
def graph():
    return DocumentGraph.from_file(str(SAMPLE))


@pytest.fixture(scope="module")
def saas():
    return DocumentGraph.from_file(str(SAAS))


def node(graph, prefix):
    """The paragraph-level content node whose text starts with ``prefix``."""
    return next(n for n in graph.nodes.values()
                if n.is_content and n.node_type != "sentence" and n.text.startswith(prefix))


def scored(n, score, semantic=0.8, lexical=0.8, reasons=("fused", "semantic")):
    return ScoredNode(n, score, {"semantic": semantic, "lexical": lexical}, list(reasons))


def roles(pruned):
    return {e.node.id: e.role for e in pruned.evidence}


NO_DEPS = PruneConfig(max_dependencies=0)


def test_score_gap_ends_selection_and_every_drop_has_a_reason(graph):
    plan = plan_query("when must invoices be paid", graph)
    a, b, c = node(graph, "The Customer shall pay"), node(graph, "The Supplier shall provide"), \
        node(graph, "Either party may terminate")
    pruned = prune(graph, [scored(a, 1.0), scored(b, 0.95, lexical=0.3), scored(c, 0.6, lexical=0.3)],
                   plan, NO_DEPS)
    assert roles(pruned) == {a.id: PRIMARY, b.id: PRIMARY}
    assert pruned.drop_reasons[c.id].startswith("gap")
    assert "gap" in pruned.explain()


def test_floor_and_budget_are_hard_limits(graph):
    plan = plan_query("when must invoices be paid", graph)
    a, b = node(graph, "The Customer shall pay"), node(graph, "Either party may terminate")
    pruned = prune(graph, [scored(a, 1.0), scored(b, 0.4)], plan, NO_DEPS)
    assert pruned.drop_reasons[b.id].startswith("floor")
    plan.max_evidence = 1
    pruned = prune(graph, [scored(a, 1.0), scored(b, 0.99)], plan, NO_DEPS)
    assert pruned.drop_reasons[b.id].startswith("budget")


def test_agreeing_signals_pass_a_gap(graph):
    """A boosted top node (here a definition) must not hide the passage that matches
    the question on both semantic and lexical similarity."""
    plan = plan_query("How may Confidential Information be used?", graph)
    definition, usage, weak = node(graph, '"Confidential Information"'), \
        node(graph, "Each party shall keep"), node(graph, "Either party may terminate")
    pruned = prune(graph, [scored(definition, 1.0), scored(usage, 0.55, 0.71, 0.81),
                           scored(weak, 0.5, 0.7, 0.1)], plan, NO_DEPS)
    assert roles(pruned) == {definition.id: PRIMARY, usage.id: PRIMARY}
    assert pruned.drop_reasons[weak.id].startswith("gap")


def test_low_signal_and_no_provenance_are_dropped(graph):
    plan = plan_query("when must invoices be paid", graph)
    a, b, c = node(graph, "The Customer shall pay"), node(graph, "The Supplier shall provide"), \
        node(graph, "Either party may terminate")
    pruned = prune(graph, [scored(a, 1.0), scored(b, 0.95, 0.1, 0.1),
                           scored(c, 0.9, reasons=("expansion:adjacent:p7",))], plan, NO_DEPS)
    assert roles(pruned) == {a.id: PRIMARY}
    assert pruned.drop_reasons[b.id] == "low signal"
    assert pruned.drop_reasons[c.id] == "no provenance"
    strong = prune(graph, [scored(a, 1.0), scored(c, 0.9, reasons=("expansion:continuation:p7",))],
                   plan, NO_DEPS)
    assert c.id in roles(strong)


def test_redundant_siblings_unless_strong(saas):
    plan = plan_query("what happens on termination", saas)
    convenience, export, suspend = node(saas, "The Customer may terminate"), \
        node(saas, "On termination, the Provider"), node(saas, "The Provider may suspend")
    items = [scored(convenience, 1.0), scored(export, 0.95), scored(suspend, 0.85, 0.7, 0.5)]
    assert prune(saas, items, plan, NO_DEPS).drop_reasons[suspend.id] == "redundant sibling"
    items[2] = scored(suspend, 0.85, 0.97, 0.1)  # strong match on its own
    assert suspend.id in roles(prune(saas, items, plan, NO_DEPS))


def test_widening_a_sentence_is_not_a_gap(graph):
    plan = plan_query("can either party terminate immediately", graph)
    paragraph = node(graph, "Either party may terminate")
    sentence = graph.children(paragraph.id)[1]
    other = node(graph, "On termination, the Customer")
    pruned = prune(graph, [scored(sentence, 1.0), scored(paragraph, 0.7), scored(other, 0.65)],
                   plan, NO_DEPS)
    assert roles(pruned) == {paragraph.id: PRIMARY, other.id: PRIMARY}
    assert pruned.drop_reasons[sentence.id].startswith("widened")


def test_dependency_rules(graph):
    config = PruneConfig()
    confidentiality = node(graph, "Each party shall keep")
    found = {n.id: role for n, role in dependencies(graph, confidentiality, config)}
    assert found[node(graph, "These obligations survive").id] == CONTEXT
    assert found[graph.definitions["confidential information"]] == DEFINITION
    # "...does not apply to Losses arising from ... a breach of Section 3" cites
    # Section 3 but does not qualify it.
    assert node(graph, "The cap in Section 4.1").id not in found

    # 4.1 cites 4.2, and 4.2 qualifies 4.1: both ends of the reference.
    cap = node(graph, "Except as set out in Section 4.2")
    pairs = {(n.id, role) for n, role in dependencies(graph, cap, config)}
    exclusions = node(graph, "The cap in Section 4.1").id
    assert {(exclusions, REFERENCE), (exclusions, QUALIFIER)} <= pairs

    indemnity = node(graph, "The Supplier shall indemnify")
    found = {n.id: role for n, role in dependencies(graph, indemnity, config)}
    assert found[node(graph, "Claims under this indemnity").id] == QUALIFIER
    assert found[graph.definitions["losses"]] == DEFINITION
    assert found[graph.definitions["misconduct"]] == DEFINITION


def test_qualifier_chain_reaches_the_carve_out(graph):
    """Indemnity -> "subject to the exclusions in Section 4.2" -> 4.2 -> the cap it
    carves out of: kept even though only the indemnity scored."""
    plan = plan_query("What happens if the Supplier commits misconduct?", graph)
    indemnity = node(graph, "The Supplier shall indemnify")
    kept = roles(prune(graph, [scored(indemnity, 1.0)], plan))
    assert kept[node(graph, "Claims under this indemnity").id] == QUALIFIER
    assert kept[node(graph, "The cap in Section 4.1").id] == REFERENCE
    assert len(kept) <= 1 + PruneConfig().max_dependencies


@pytest.mark.parametrize("text,number,expected", [
    ("The cap in Section 4.1 does not apply to Losses arising from a breach of Section 3.", "4.1", True),
    ("The cap in Section 4.1 does not apply to Losses arising from a breach of Section 3.", "3", False),
    ("Claims under this indemnity are subject to the exclusions in Section 4.2.", "4.2", True),
    ("Except as set out in Section 4.2, each party's total liability is limited.", "4.2", True),
    ("Except as set out in Section 4.2, each party's total liability is limited.", "4", False),
    ("Fees are payable under Section 3.", "3", False),
])
def test_qualifies(text, number, expected):
    assert _qualifies(text, number) is expected


def test_conflicting_numbers_are_counterparts(saas):
    four_hours = next(n for n in saas.nodes.values() if n.text.startswith("The Provider shall respond")
                      and "four hours" in n.text)
    one_hour = node(saas, "The Provider shall respond to critical incidents within one hour")
    assert one_hour in conflicts(saas, four_hours, 0.75)
    paragraph = saas.nodes[four_hours.parent_id]
    assert one_hour in conflicts(saas, paragraph, 0.75)
    assert four_hours in conflicts(saas, one_hour, 0.75)
    assert conflicts(saas, node(saas, "The Provider shall process"), 0.75) == []
    plan = plan_query("How fast must critical incidents be handled?", saas)
    assert roles(prune(saas, [scored(one_hour, 1.0)], plan)).get(four_hours.id) == COUNTERPART \
        or roles(prune(saas, [scored(one_hour, 1.0)], plan)).get(paragraph.id) == COUNTERPART


def test_summary_keeps_its_section():
    qa = DocumentQA.from_file(str(SAMPLE))
    texts = " ".join(i.text for i in qa.ask("Summarize the liability terms").evidence.items)
    assert "total liability" in texts and "does not apply" in texts


def test_default_config_per_embedder():
    assert default_config("tfidf").gap == 0.25
    assert default_config("st:all-MiniLM-L6-v2") == PruneConfig()
    assert DocumentQA.from_file(str(SAMPLE)).prune_config.gap == 0.25


def test_pruning_metrics_and_regression():
    """Phase 8 regression guard (TF-IDF, small set): recall and context at least the
    old pruner's (0.833 / 0.841), primary items mostly gold, bounded packs."""
    summary = evaluate(load_cases(str(ROOT / "eval" / "contracts_v1.jsonl")))["summary"]
    assert summary["evidence_recall"] >= 0.833
    assert summary["pack_context_recall"] >= 0.84
    assert summary["primary_precision"] >= 0.6
    assert summary["mean_pack_words"] <= 70
