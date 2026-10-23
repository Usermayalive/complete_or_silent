"""G6: adversarial suite, structural coherence and case metrics, precomputed embedder."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gnn_pruner import DocumentGraph  # noqa: E402
from gnn_pruner.evidence import EvidenceItem  # noqa: E402
from gnn_pruner.gnn.g6 import PrecomputedEmbedder, build_adversarial, case_metrics, coherence  # noqa: E402
from gnn_pruner.layers import four_layer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample_contract.md"


def item(graph, node_id, role="primary"):
    n = graph.nodes[node_id]
    return EvidenceItem(label="E", node_id=node_id, role=role, section_path=graph.path(node_id), text=n.text,
                        score=1.0, start=n.start, end=n.end)


def para(graph, prefix):
    return next(n.id for n in graph.nodes.values() if n.node_type == "paragraph" and n.text.startswith(prefix))


@pytest.fixture(scope="module")
def graph():
    return DocumentGraph.from_file(str(SAMPLE))


def test_coherence_counts_definitions_and_cited_clauses(graph):
    indemnity = para(graph, "The Supplier shall indemnify")  # uses Losses + Misconduct
    alone = [item(graph, indemnity)]
    assert coherence(graph, alone) == 0.0
    with_defs = alone + [item(graph, graph.definitions["losses"], "definition"),
                         item(graph, graph.definitions["misconduct"], "definition")]
    assert coherence(graph, with_defs) == 1.0
    assert coherence(graph, [item(graph, graph.definitions["losses"])]) is None  # needs nothing


def test_case_metrics(graph):
    layered = four_layer(graph, "s")
    cap, excl = para(graph, "Except as set out"), para(graph, "The cap in Section 4.1")
    other = para(graph, "Either party may terminate")
    gold = ["total liability under this Agreement is limited", "does not apply to Losses arising from Misconduct"]
    m = case_metrics(graph, layered, [item(graph, cap), item(graph, excl)], [cap, excl, other], gold, 300)
    assert m["recall@10"] == m["complete@10"] == m["evidence_recall"] == m["complete"] == m["hit@1"] == 1.0
    assert m["cross_chapter"] == 0.0  # both in chapter 4
    m = case_metrics(graph, layered, [item(graph, other)], [other], gold, 300)
    assert m["evidence_recall"] == 0.0 and m["recall@10"] == 0.0 and m["hit@1"] == 0.0
    assert m["pack_share"] == pytest.approx(len(graph.nodes[other].text.split()) / 300)


def test_precomputed_embedder_reuses_vectors():
    class Base:
        name, calls = "base", 0

        def encode(self, texts):
            Base.calls += len(texts)
            return np.ones((len(texts), 3), dtype=np.float32)

        def encode_query(self, text):
            return np.zeros(3)

    e = PrecomputedEmbedder(Base(), ["a", "b"], np.eye(2, 3, dtype=np.float32))
    out = e.encode(["a", "c", "b", "c"])
    assert Base.calls == 1 and np.allclose(out[0], [1, 0, 0]) and np.allclose(out[1], [1, 1, 1])
    assert e.name == "base" and e.cacheable


def test_adversarial_suite(tmp_path):
    if not (ROOT / "data" / "corpus_v1" / "questions_val.jsonl").exists():
        pytest.skip("generated corpus not built")
    out = tmp_path / "adv.jsonl"
    n = build_adversarial(out=str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out))
    path = out if out.exists() else ROOT / str(out)
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    assert n == len(rows) == 100
    kinds = {}
    for r in rows:
        kinds.setdefault(r["tags"]["perturbation"], []).append(r)
    assert set(kinds) == {"distractor", "wrong_number", "party_swap", "typo", "paraphrase"}
    assert all(len(v) == 20 for v in kinds.values())
    for r in rows:
        assert r["gold_evidence"] and r["question"] != "" and (ROOT / "eval" / r["document"]).resolve().exists()
        if r["tags"]["perturbation"] != "paraphrase":
            assert r["question"] != r["tags"]["base_question"] or r["tags"]["perturbation"] == "party_swap"
    again = tmp_path / "again.jsonl"
    build_adversarial(out=str(again))
    assert Path(path).read_text() == again.read_text()  # deterministic
