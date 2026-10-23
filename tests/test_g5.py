"""G5: partitioning, boundary replication, shards, distributed parity, stitching."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("pyarrow")

from test_gnn import DIM, ToyEmbedder  # noqa: E402

from gnn_pruner import DocumentGraph  # noqa: E402
from gnn_pruner.corpus import embed_layered  # noqa: E402
from gnn_pruner.gnn.cross_attention import AttentionHead, AttentiveCoarse, save  # noqa: E402
from gnn_pruner.gnn.distributed import (DistributedPruner, ShardWorker, halo, partition,  # noqa: E402
                                        single_process_scores, stitch, write_shards)
from gnn_pruner.gnn.features import input_dim  # noqa: E402
from gnn_pruner.gnn.model import CoarseGNN, ModelConfig  # noqa: E402
from gnn_pruner.gnn.two_stage import FineGNN, fine_dim, fine_table, save_two_stage  # noqa: E402
from gnn_pruner.layers import four_layer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LONG = ROOT / "examples" / "scale" / "saas_agreement.md"
STAGE = {"theta": 1.0, "cap": 0.2, "min_keep": 3, "merge_ratio": 0.8}
WINDOW = 16


@pytest.fixture(scope="module")
def table():
    layered = four_layer(DocumentGraph.from_file(str(LONG)), "long")
    _, ids, matrix = embed_layered(layered, ToyEmbedder())
    return fine_table(layered, ids, matrix, "mean")


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    torch.manual_seed(0)
    coarse = CoarseGNN(ModelConfig(in_dim=input_dim(DIM), hidden=8, skip=True))
    fine = FineGNN(fine_dim(DIM), 16)
    plain = tmp_path_factory.mktemp("plain")
    save_two_stage(plain, coarse, fine, {"embedder": ToyEmbedder.name, "pooling": "mean", "stage_config": STAGE})
    head = AttentionHead(coarse.hidden_dim, heads=4, window=WINDOW, top_k=10)
    torch.nn.init.normal_(head.head.weight, std=0.5)  # make the attention matter for parity
    attentive = tmp_path_factory.mktemp("attentive")
    cfg = {"window": WINDOW, "top_k": 10, "use_local": True, "use_global": True, "heads": 4}
    save(attentive, AttentiveCoarse(coarse, head), fine,
         {"base": str(plain), "embedder": ToyEmbedder.name, "pooling": "mean", "stage_config": STAGE,
          "cea": {"coarse": cfg, "fine": None}})
    return {"plain": str(plain), "attentive": str(attentive)}


def test_tree_partition_keeps_chapters_whole_and_contiguous(table):
    owner = partition(table.doc, 4, "tree")
    assert set(owner.tolist()) == {0, 1, 2, 3}
    assert np.all(np.diff(owner) >= 0)  # contiguous in document order
    for nid in table.doc.ids:
        chapter = table.doc.layered.path(nid)[0]
        assert owner[table.doc.row[nid]] == owner[table.doc.row[chapter]]
    sizes = np.bincount(owner)
    assert sizes.max() < 2 * len(owner) / 4


def test_metis_partition(table):
    pytest.importorskip("pymetis")
    owner = partition(table.doc, 3, "metis")
    assert owner.shape == (len(table.doc.ids),) and set(owner.tolist()) <= {0, 1, 2}


def test_halos(table):
    owner = partition(table.doc, 3, "tree")
    spec = halo(table.doc, owner, 1, "spec")
    owned = np.where(owner == 1)[0]
    assert set(spec) == set(range(owned.min() - 3, owned.min())) | set(range(owned.max() + 1, owned.max() + 4))
    exact = halo(table.doc, owner, 1, "exact", hops=2, window=WINDOW)
    assert set(spec) <= set(exact) and not set(exact) & set(owned)
    # Every neighbour of an owned node is present.
    present = set(owned) | set(exact)
    for t, e in table.doc.edges.items():
        for a, b in e.t().tolist():
            if a in set(owned):
                assert b in present


def test_shard_files_round_trip(table, models, tmp_path):
    owner = partition(table.doc, 2, "tree")
    dirs = write_shards(table.doc, owner, str(tmp_path), "exact", 2, WINDOW)
    worker = ShardWorker(str(dirs[0]), models["plain"])
    info = worker.info()
    assert info["owned"] == int((owner == 0).sum()) and info["halo"] > 0
    q = "service credits availability"
    full = table.doc.bm25.scores(q.split())
    worker.stats(q, ToyEmbedder().encode_query(q))
    assert np.allclose(worker.bm25.scores(q.split()), full[worker.rows])  # document-level BM25 per shard


@pytest.mark.parametrize("model", ["plain", "attentive"])
@pytest.mark.parametrize("workers,method", [(1, "tree"), (3, "tree"), (4, "metis")])
def test_exact_halo_matches_single_process(table, models, tmp_path, model, workers, method):
    if method == "metis":
        pytest.importorskip("pymetis")
    owner = partition(table.doc, workers, method)
    dirs = write_shards(table.doc, owner, str(tmp_path), "exact", 2, WINDOW)
    pruner = DistributedPruner(table, dirs, models[model], backend="local")
    for q in ("What happens if availability falls below the Service Levels?", "Where is Customer Data stored?"):
        v = ToyEmbedder().encode_query(q)
        reference, _ = single_process_scores(table, models[model], q, v)
        logits, timings = pruner.coarse_scores(q, v)
        assert np.allclose(logits, reference, atol=1e-4), np.abs(logits - reference).max()
        assert {"stats_ms", "forward_ms", "finish_ms", "coarse_ms"} <= set(timings)


def test_spec_halo_runs_but_is_approximate(table, models, tmp_path):
    owner = partition(table.doc, 3, "tree")
    dirs = write_shards(table.doc, owner, str(tmp_path), "spec")
    pruner = DistributedPruner(table, dirs, models["attentive"], backend="local")
    q = "What happens if availability falls below the Service Levels?"
    v = ToyEmbedder().encode_query(q)
    reference, _ = single_process_scores(table, models["attentive"], q, v)
    logits, _ = pruner.coarse_scores(q, v)
    assert logits.shape == reference.shape and np.abs(logits - reference).max() > 0


def test_end_to_end_merge_and_stitch(table, models, tmp_path):
    owner = partition(table.doc, 3, "tree")
    pruner = DistributedPruner(table, write_shards(table.doc, owner, str(tmp_path), "exact", 2, WINDOW),
                               models["attentive"], backend="local")
    q = "How quickly must a personal data breach be notified?"
    result = pruner.run(q, ToyEmbedder().encode_query(q))
    assert result.merged and set(result.timings) >= {"coarse_ms", "fine_ms", "merge_stitch_ms", "total_ms"}
    layered = table.doc.layered
    allowed = set(result.merged)
    for nid in result.merged:
        allowed.update(layered.path(nid))
    assert result.stitched_edges and all(s in allowed and t in allowed for s, t, _ in result.stitched_edges)
    assert stitch(table, []) == []


def test_ray_backend_matches_local(table, models, tmp_path):
    ray = pytest.importorskip("ray")
    owner = partition(table.doc, 2, "tree")
    dirs = write_shards(table.doc, owner, str(tmp_path), "exact", 2, WINDOW)
    q = "Where is Customer Data stored?"
    v = ToyEmbedder().encode_query(q)
    local, _ = DistributedPruner(table, dirs, models["plain"], backend="local").coarse_scores(q, v)
    remote = DistributedPruner(table, dirs, models["plain"], backend="ray", threads_per_worker=1)
    try:
        logits, _ = remote.coarse_scores(q, v)
    finally:
        remote.shutdown()
        ray.shutdown()
    assert np.allclose(logits, local, atol=1e-5)
