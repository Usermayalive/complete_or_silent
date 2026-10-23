from pathlib import Path

from gnn_pruner.evaluation import evaluate, load_cases
from gnn_pruner.parser import normalize_ws
from gnn_pruner.query_planner import (
    COMPARISON,
    CONTRADICTION,
    CROSS_REFERENCE,
    DEFINITION,
    FACT,
    SUMMARY,
)

DATASET = Path(__file__).resolve().parent.parent / "eval" / "contracts_v1.jsonl"
QUERY_TYPES = {FACT, DEFINITION, COMPARISON, SUMMARY, CROSS_REFERENCE, CONTRADICTION}


def test_dataset_is_well_formed():
    cases = load_cases(str(DATASET))
    assert len({c.id for c in cases}) == len(cases)
    for case in cases:
        assert case.query_type in QUERY_TYPES, case.id
        assert bool(case.gold_evidence) == case.answerable, case.id
        source = normalize_ws(Path(case.document).read_text()).lower()
        for snippet in case.gold_evidence:
            assert normalize_ws(snippet).lower() in source, (case.id, snippet)


def test_evaluation_runs_and_reports_metrics():
    report = evaluate(load_cases(str(DATASET)))
    summary = report["summary"]
    assert summary["cases"] == len(report["cases"])
    for key in ("planner_accuracy", "evidence_recall", "hit_at_1", "abstention_rate"):
        assert 0.0 <= summary[key] <= 1.0
