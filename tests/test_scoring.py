"""Phase 7: relevance scoring, explainability and the abstention signal."""

import json
from pathlib import Path

import pytest

from gnn_pruner import DocumentGraph, DocumentQA
from gnn_pruner.embeddings import load_embedder
from gnn_pruner.evaluation import (RANKING_METRICS, evaluate, load_cases, ranking_metrics,
                                   summarize_ranking)
from gnn_pruner.index import GraphIndex
from gnn_pruner.query_planner import DEFAULT_WEIGHTS
from gnn_pruner.retrieval import retrieve
from gnn_pruner.scoring import (CALIBRATION_FILE, FEATURES, Calibration, EvidenceConfidence,
                                LinearScorer, fit_calibration, load_calibration, load_weights,
                                neighbour_support, save_calibration)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample_contract.md"
CALIBRATION_SET = ROOT / "eval" / "calibration_v1.jsonl"
NEVER = Calibration("test", [0.0, 0.0], -10.0, 0.5)  # P(answerable) ~ 0 for every question


@pytest.fixture(scope="module")
def index():
    return GraphIndex(DocumentGraph.from_file(str(SAMPLE)))


def test_linear_scorer_contributions_sum_to_score():
    scorer = LinearScorer({"semantic": 0.5, "lexical": 0.25, "noise": 0.2})
    score, contributions = scorer.score({"semantic": 0.8, "lexical": 0.4, "noise": 1.0, "fusion": 1.0})
    assert contributions == pytest.approx({"semantic": 0.4, "lexical": 0.1, "noise": -0.2, "fusion": 0.0})
    assert score == pytest.approx(sum(contributions.values()))


def test_unknown_weight_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="semantc"):
        LinearScorer({"semantc": 1.0})
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"semantic": 1, "fusion": 0.5}))
    assert load_weights(str(path)) == {"semantic": 1.0, "fusion": 0.5}


def test_default_weights_only_name_known_features():
    assert set(DEFAULT_WEIGHTS) <= set(FEATURES)


def test_every_node_has_full_breakdown(index):
    trace = retrieve(index, "What is the cap on liability?")
    assert trace.scored == sorted(trace.scored, key=lambda s: -s.score)
    for s in trace.scored:
        assert set(s.features) == set(FEATURES)
        assert all(0.0 <= v <= 1.5 for v in s.features.values())
        assert s.score == pytest.approx(sum(s.contributions.values()))
        assert s.reasons
    assert trace.scored[0].explain()


def test_weights_override_changes_the_ranking(index):
    question = "What is the cap on liability?"
    only_type = {k: 0.0 for k in FEATURES}
    only_type["type_value"] = 1.0
    scored = retrieve(index, question, weights=only_type).scored
    assert all(s.score == pytest.approx(s.features["type_value"]) for s in scored)
    qa = DocumentQA.from_file(str(SAMPLE), weights=only_type)
    assert qa.ask(question).scored[0].score == pytest.approx(1.0)


def test_a_custom_scorer_plugs_in(index):
    class Shortest:
        name = "shortest"

        def score(self, features):
            return -features["type_value"], {"type_value": -features["type_value"]}

    scored = retrieve(index, "What is the cap on liability?", scorer=Shortest()).scored
    assert scored[0].node.is_section  # lowest type value ranks first under this scorer


def test_neighbour_support_rewards_the_shared_neighbour():
    graph = DocumentGraph.from_file(str(SAMPLE))
    cap = graph.children(graph.lookup("4.1 Cap on Liability").id)[0]
    indemnity = graph.children(graph.lookup("5. Indemnity").id)[0]
    losses = graph.definitions["losses"]
    exclusions = graph.children(graph.lookup("4.2 Exclusions from the Cap").id)[0]
    nodes = [cap.id, indemnity.id, losses, exclusions.id]
    # Only the two passages are relevant; both point at the Losses definition / 4.2.
    support = neighbour_support(graph, nodes, {cap.id: 1.0, indemnity.id: 1.0})
    assert max(support.values()) == pytest.approx(1.0)
    assert support[exclusions.id] > 0 and support[losses] > 0
    assert neighbour_support(graph, nodes, {}) == {n: 0.0 for n in nodes}


@pytest.mark.parametrize("question,anchor", [
    ("Compare sections 4.1 and 4.2", "cited section"),
    ("What does Misconduct mean?", "defined term"),
    ('Where does the contract say "material breach"?', "quoted phrase"),
    ('Where does the contract say "force majeure"?', ""),
    ("Which courts have jurisdiction?", ""),
])
def test_confidence_anchors(index, question, anchor):
    assert retrieve(index, question).confidence.anchor == anchor


def test_confidence_coverage_is_absolute(index):
    exact = retrieve(index, "undisputed invoices within thirty days of receipt").confidence
    unknown = retrieve(index, "zebra quokka marmalade").confidence
    assert exact.coverage == pytest.approx(1.0)
    assert unknown.coverage == 0.0
    assert exact.margin > unknown.margin
    assert exact.probability is None and exact.level == "uncalibrated"


def test_abstention_is_opt_in_and_respects_anchors():
    qa = DocumentQA.from_file(str(SAMPLE), calibration=NEVER)
    result = qa.ask("What insurance must the Supplier carry?")
    assert result.confidence.insufficient and result.confidence.level == "insufficient"
    assert result.evidence.items  # flagged, not dropped

    qa = DocumentQA.from_file(str(SAMPLE), calibration=NEVER, abstain=True)
    result = qa.ask("What insurance must the Supplier carry?")
    assert not result.evidence.items and result.answer.startswith("Insufficient evidence")
    anchored = qa.ask("Compare sections 4.1 and 4.2")  # names sections that exist
    assert anchored.evidence.items and anchored.confidence.level == "high"


def test_fit_calibration_keeps_every_answerable_question(tmp_path):
    answerable = [EvidenceConfidence(m, c) for m, c in [(0.4, 0.6), (0.3, 0.2), (0.5, 0.9), (0.25, 0.0)]]
    unanswerable = [EvidenceConfidence(m, c) for m, c in [(0.1, 0.0), (0.15, 0.2), (0.28, 0.1)]]
    anchored = [EvidenceConfidence(0.0, 0.0, anchor="cited section")]  # excluded from the fit
    cal = fit_calibration(answerable + unanswerable + anchored, [True] * 4 + [False] * 4, "toy")
    assert cal.cases == 7
    assert all(cal.probability(c.margin, c.coverage) >= cal.threshold for c in answerable)
    assert cal.probability(0.1, 0.0) < cal.threshold
    assert cal.probability(0.5, 0.5) > cal.probability(0.2, 0.5)  # monotone in margin

    path = tmp_path / "cal.json"
    save_calibration(cal, path)
    assert load_calibration("toy", path) == cal
    assert load_calibration("other", path) is None


def test_shipped_calibrations_load():
    data = json.loads(CALIBRATION_FILE.read_text())
    assert "tfidf" in data
    for name in data:
        cal = load_calibration(name)
        assert cal.fitted_on == CALIBRATION_SET.name and 0 < cal.threshold < 1


def test_tfidf_confidence_separates_the_calibration_set():
    report = evaluate(load_cases(str(CALIBRATION_SET)), load_embedder("tfidf"))
    assert report["summary"]["confidence_auc"] >= 0.8


def test_ranking_metrics():
    qa = DocumentQA.from_file(str(SAMPLE))
    graph = qa.graph
    cap = graph.children(graph.lookup("4.1 Cap on Liability").id)[0].id
    excl = graph.children(graph.lookup("4.2 Exclusions from the Cap").id)[0].id
    other = graph.children(graph.lookup("6. Termination").id)[0].id
    gold = ["total liability under this agreement", "does not apply to losses"]
    best = ranking_metrics(qa, [cap, excl, other], gold)
    assert best == {"hit_at_1": 1.0, "mrr": 1.0, "ndcg_at_10": pytest.approx(1.0), "recall_at_5": 1.0}
    late = ranking_metrics(qa, [other, cap], gold)
    assert late["hit_at_1"] == 0.0 and late["mrr"] == 0.5 and late["recall_at_5"] == 0.5
    assert 0 < late["ndcg_at_10"] < 1
    assert ranking_metrics(qa, [other], gold)["mrr"] == 0.0


def test_evaluation_compares_scored_and_fused_orders():
    report = evaluate(load_cases(str(ROOT / "eval" / "contracts_v1.jsonl")))
    table = summarize_ranking(report["cases"])
    assert {"scored", "fused", "lexical", "semantic"} <= set(table)
    for order in table.values():
        assert set(order) == set(RANKING_METRICS)
    assert report["summary"]["mrr"] == table["scored"]["mrr"]
