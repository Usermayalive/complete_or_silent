"""G3: coarsening (pooling, keep rule, discard rule), FineGNN, merge, spec end-to-end path."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from test_gnn import DIM, ToyEmbedder  # noqa: E402

from gnn_pruner import DocumentGraph  # noqa: E402
from gnn_pruner.corpus import embed_layered  # noqa: E402
from gnn_pruner.gnn.features import input_dim, paragraph_vectors  # noqa: E402
from gnn_pruner.gnn.model import CoarseGNN, ModelConfig  # noqa: E402
from gnn_pruner.gnn.spec_qa import SpecQA  # noqa: E402
from gnn_pruner.gnn.two_stage import (FineGNN, StageConfig, fine_dim, fine_inputs, fine_table,  # noqa: E402
                                      load_two_stage, retain, run_stages, save_two_stage, stage_metrics,
                                      subset_for)
from gnn_pruner.layers import four_layer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAAS = ROOT / "examples" / "saas_agreement.md"


@pytest.fixture(scope="module")
def table():
    layered = four_layer(DocumentGraph.from_file(str(SAAS)), "saas")
    _, ids, matrix = embed_layered(layered, ToyEmbedder())
    return fine_table(layered, ids, matrix, pooling="mean")


def test_mean_pooling(table):
    layered = table.doc.layered
    _, ids, matrix = embed_layered(layered, ToyEmbedder())
    vectors = {nid: matrix[i] for i, nid in enumerate(ids)}
    pooled = paragraph_vectors(layered, vectors, "mean")
    multi = next(n for n in layered.layer(3) if len(layered.children(n.id)) > 1)
    mean = np.mean([vectors[c.id] for c in layered.children(multi.id)], axis=0)
    assert np.allclose(pooled[multi.id], mean / np.linalg.norm(mean), atol=1e-6)
    single = next(n for n in layered.layer(3) if layered.children(n.id)[0].virtual)
    assert np.allclose(pooled[single.id], vectors[single.id])
    assert np.allclose(paragraph_vectors(layered, vectors, "direct")[multi.id], vectors[multi.id])


def test_fine_table(table):
    for r in table.sentences:
        paragraph = table.parent[r]
        assert table.layer[paragraph] == 3
        assert table.doc.ids[table.coarse_row[r]] == table.ids[paragraph]
        assert r in table.claims[paragraph]


def test_keep_rule():
    class Doc:  # minimal stand-in: 10 paragraphs
        l3 = np.arange(10)
    probs = np.array([0.9, 0.8, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    # theta * mean: only 2 above 1.0 x mean, but at least min_keep.
    assert list(retain(Doc, probs, StageConfig(theta=1.0, cap=0.5, min_keep=3))) == [0, 1, 2]
    assert list(retain(Doc, probs, StageConfig(theta=1.0, cap=0.5, min_keep=1))) == [0, 1]
    flat = np.full(10, 0.5) + np.arange(10) * 1e-3   # all above 0.5 x mean -> the cap binds
    assert len(retain(Doc, flat, StageConfig(theta=0.5, cap=0.2, min_keep=1))) == 2


def test_discarded_paragraphs_contribute_no_sentences(table):
    kept = table.doc.l3[:3]
    rows = subset_for(table, kept)
    kept_rows = {table.row[table.doc.ids[c]] for c in kept}
    for r in rows:
        if table.layer[r] == 4:
            assert table.parent[r] in kept_rows
        if table.layer[r] == 3:
            assert r in kept_rows
    for c in kept:  # every kept paragraph brings all its sentences and its headings
        r = table.row[table.doc.ids[c]]
        assert set(table.claims[r]) <= set(rows)
        p = table.parent[r]
        while p >= 0:
            assert p in rows
            p = table.parent[p]


def test_fine_inputs(table):
    rows = subset_for(table, table.doc.l3[:4])
    q = ToyEmbedder().encode_query("availability service credits")
    x, edge_index, context = fine_inputs(table, rows, "availability service credits", q, None)
    assert x.shape == (len(rows), fine_dim(DIM))
    local = {int(r): i for i, r in enumerate(rows)}
    for r in rows:
        if table.layer[r] == 4:
            assert torch.equal(context[local[r]], x[local[table.parent[r]]])
    pairs = set(map(tuple, edge_index.t().tolist()))
    assert all((b, a) in pairs for a, b in pairs)
    sentences = [local[r] for r in rows if table.layer[r] == 4]
    assert all((a, b) in pairs for a, b in zip(sentences, sentences[1:]))  # sequential, re-linked


def test_fine_gnn_matches_spec():
    fine = FineGNN(fine_dim(DIM), 16)
    assert isinstance(fine.alpha, torch.nn.Parameter) and float(fine.alpha) == pytest.approx(0.3)
    assert fine.context_projection.in_features == fine.context_projection.out_features == fine_dim(DIM)
    x, ctx = torch.randn(5, fine_dim(DIM)), torch.randn(5, fine_dim(DIM))
    edges = torch.tensor([[0, 1], [1, 0]])
    fine(x, edges, ctx).sum().backward()
    assert fine.alpha.grad is not None and float(fine.alpha.grad) != 0.0
    no_ctx = FineGNN(fine_dim(DIM), 16, use_context=False)
    assert torch.equal(no_ctx(x, edges, ctx), no_ctx(x, edges, torch.zeros_like(ctx)))


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    torch.manual_seed(0)
    coarse = CoarseGNN(ModelConfig(in_dim=input_dim(DIM), hidden=16, skip=True))
    fine = FineGNN(fine_dim(DIM), 16)
    path = tmp_path_factory.mktemp("two_stage")
    save_two_stage(path, coarse, fine, {"embedder": ToyEmbedder.name, "pooling": "mean",
                                        "stage_config": {"theta": 1.0, "cap": 0.2, "min_keep": 3,
                                                         "merge_ratio": 0.8}})
    return path


def test_stages_and_merge(table, models):
    coarse, fine, meta = load_two_stage(str(models))
    cfg = StageConfig(**meta["stage_config"])
    q = ToyEmbedder().encode_query("service credits")
    result = run_stages(coarse, fine, table, "service credits", q, cfg)
    assert 3 <= len(result.kept_l3) <= max(3, int(np.ceil(0.2 * len(table.doc.l3))))
    assert len(result.rows) < len(table.ids) and len(result.fine_probs) == len(result.rows)
    gold_leaf = table.ids[table.sentences[0]]
    m = stage_metrics(table, result, [{gold_leaf}], [{table.ids[table.parent[table.sentences[0]]]}], cfg)
    assert {"mrr", "stage1_recall", "merged_share", "merged_recall", "fine_nodes"} <= set(m)
    assert 0 < m["merged_share"] <= 1
    flat = run_stages(None, fine, table, "service credits", q, cfg)
    assert len(flat.rows) == len(table.ids)  # the no-coarsening baseline sees every node


def test_spec_qa_end_to_end(models):
    qa = SpecQA.from_file(str(SAAS), str(models), embedder=ToyEmbedder())
    answer = qa.ask("What happens if availability falls below the Service Levels?")
    items = answer.evidence.items
    assert items and all(i.node_id in qa.graph.nodes for i in items)  # virtual L4 ids mapped to paragraphs
    assert all(not i.node_id.endswith("#0") for i in items)
    assert set(answer.timings) == {"encode_ms", "coarse_ms", "fine_ms", "merge_pack_ms"}
    assert 0 < answer.stats["merged_sentence_share"] <= 1
