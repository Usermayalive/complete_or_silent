from pathlib import Path

import pytest

from gnn_pruner import DocumentGraph, DocumentQA
from gnn_pruner.evaluation import evaluate_planner
from gnn_pruner.query_planner import (
    COMPARISON,
    CONTRADICTION,
    DEFINITION,
    DEFINITION_FIRST,
    FACT,
    FOLLOW_REFERENCES,
    HYBRID,
    PAIRWISE,
    PER_SIDE,
    SCOPED,
    SECTION_SUMMARY,
    SUMMARY,
    PrototypeClassifier,
    match_sections,
    plan_query,
    resolve_term,
)

ROOT = Path(__file__).resolve().parent.parent
SAAS = ROOT / "examples" / "saas_agreement.md"
MSA = ROOT / "examples" / "sample_contract.md"


@pytest.fixture(scope="module")
def saas():
    return DocumentGraph.from_file(str(SAAS))


@pytest.fixture(scope="module")
def msa():
    return DocumentGraph.from_file(str(MSA))


def _numbers(graph, ids):
    return [graph.nodes[i].metadata.get("number") for i in ids]


# ------------------------------------------------------------ classification

@pytest.mark.parametrize("question,expected", [
    ("What's a Subscription Term?", DEFINITION),          # defined term -> definition
    ("What is the uptime commitment?", FACT),             # not a defined term -> fact
    ("What is the Subscription Term length?", FACT),      # mentions one, asks something else
    ("Do sections 5 and 7 contradict each other?", CONTRADICTION),  # beats comparison
    ("Renewal vs termination: how do they differ?", COMPARISON),
    ("TL;DR of the fees", SUMMARY),
])
def test_classification_uses_the_document(saas, question, expected):
    assert plan_query(question, saas).query_type == expected


def test_without_a_graph_unquoted_what_is_stays_fact():
    assert plan_query("What's a Subscription Term?").query_type == FACT
    assert plan_query("What is 'Losses'?").query_type == DEFINITION


# ---------------------------------------------------------------- resolution

def test_term_resolution(saas):
    assert resolve_term("Affiliates", saas) == "affiliate"
    assert resolve_term("the Customer Data", saas) == "customer data"
    assert resolve_term("Customer Data in this agreement", saas) == "customer data"
    assert resolve_term("platform", saas) is None


def test_definition_plan_resolves_term_from_context(saas):
    plan = plan_query("Service Levels - what does that term mean?", saas)
    assert (plan.strategy, plan.focus_terms) == (DEFINITION_FIRST, ["service levels"])
    plan = plan_query("Define Widgets", saas)
    assert plan.focus_terms == ["widgets"]
    assert any("not a defined term" in n for n in plan.notes)


def test_section_matching_prefers_specific_and_handles_word_forms(saas, msa):
    assert _numbers(saas, [s for s, _ in match_sections("notice for non-renewal", saas)]) == ["9.1"]
    assert _numbers(msa, [s for s, _ in match_sections("summarize the liability terms", msa)]) == ["4"]
    assert "4.1" in _numbers(msa, [s for s, _ in match_sections("the liability cap", msa)])
    assert _numbers(msa, [s for s, _ in match_sections("can we terminate", msa)]) == ["6"]


def test_cited_sections_and_missing_ones(saas):
    plan = plan_query("What happens if the Customer breaches Section 2(c)?", saas)
    assert (plan.strategy, _numbers(saas, plan.target_sections)) == (FOLLOW_REFERENCES, ["2(c)"])
    plan = plan_query("What does Section 42 say?", saas)
    assert plan.target_sections == [] and any("42" in n for n in plan.notes)


def test_comparison_sides(saas, msa):
    plan = plan_query("Compare sections 4.1 and 4.2", msa)
    assert plan.strategy == PER_SIDE
    assert [_numbers(msa, s.section_ids) for s in plan.sides] == [["4.1"], ["4.2"]]
    plan = plan_query("Contrast the support terms with the service levels", saas)
    assert [_numbers(saas, s.section_ids) for s in plan.sides] == [["7"], ["5"]]
    plan = plan_query("Sections 9.1 versus 9.2", saas)
    assert [_numbers(saas, s.section_ids) for s in plan.sides] == [["9.1"], ["9.2"]]


def test_summary_and_condition_plans(saas):
    plan = plan_query("Give me an overview of the fees", saas)
    assert (plan.strategy, _numbers(saas, plan.target_sections)) == (SECTION_SUMMARY, ["3"])
    plan = plan_query("What if availability drops below 99.9 percent?", saas)
    assert plan.sub_queries == ["availability drops below 99.9 percent"]


def test_fact_strategy_depends_on_citations(saas):
    assert plan_query("What does Section 8 cover regarding losses?", saas).strategy == SCOPED
    assert plan_query("Where is customer data stored?", saas).strategy == HYBRID


def test_plan_is_explainable_and_serialisable(saas):
    plan = plan_query("Compare the renewal and termination provisions", saas)
    text = plan.explain(saas)
    assert "strategy: per_side" in text and "9.1 Renewal" in text and "10. Termination" in text
    assert plan.to_dict()["sides"][0]["text"] == "renewal"


# ------------------------------------------------------ strategies in action

def test_contradiction_keeps_both_conflicting_statements():
    result = DocumentQA.from_file(str(SAAS)).ask("Are the incident response commitments consistent?")
    assert result.plan.strategy == PAIRWISE
    texts = " ".join(i.text for i in result.evidence.items)
    assert "within four hours" in texts and "within one hour" in texts


def test_comparison_keeps_evidence_from_each_side():
    qa = DocumentQA.from_file(str(SAAS))
    result = qa.ask("Contrast the support terms with the service levels")
    paths = {i.section_path for i in result.evidence.items}
    assert "7. Support" in paths and "5. Service Levels" in paths


def test_summary_covers_the_target_section():
    qa = DocumentQA.from_file(str(SAAS))
    result = qa.ask("Outline the termination provisions")
    section = qa.graph.lookup("10. Termination")
    expected = {c.text for c in qa.graph.children(section.id)}
    assert expected <= {i.text for i in result.evidence.items}


# ------------------------------------------------------------ classifier

class KeywordEmbedder:
    """Tiny fake 'dense' embedder: one dimension per marker word."""

    name, cacheable = "keywords", True
    MARKERS = ["mean", "compare", "summar", "conflict", "happen", "when", "how"]

    def fit(self, texts):
        return self

    def encode(self, texts):
        import numpy as np

        out = np.array([[1.0 if m in t.lower() else 0.0 for m in self.MARKERS] + [0.1]
                        for t in texts], dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def encode_query(self, text):
        return self.encode([text])[0]


def test_fallback_needs_a_defined_term_for_definitions(saas):
    clf = PrototypeClassifier(KeywordEmbedder(), margins={}, default_margin=0.0)
    assert plan_query("Tell me what Misconduct means", saas, classifier=clf).query_type == FACT
    msa = DocumentGraph.from_file(str(MSA))
    plan = plan_query("Tell me what Misconduct means", msa, classifier=clf)
    assert (plan.query_type, plan.focus_terms) == (DEFINITION, ["misconduct"])
    assert any("semantic fallback" in n for n in plan.notes)


def test_planner_sets_score_as_recorded():
    dev = evaluate_planner(str(ROOT / "eval" / "planner_v1.jsonl"))["summary"]
    assert dev["type_accuracy"] == 1.0 and dev["focus_accuracy"] == 1.0
    held_out = evaluate_planner(str(ROOT / "eval" / "planner_holdout_v1.jsonl"))["summary"]
    assert held_out["type_accuracy"] >= 0.56  # rules only (TF-IDF): no semantic fallback
