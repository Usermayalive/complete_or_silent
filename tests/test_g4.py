"""G4: cross-edge attention (spec §4)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from test_gnn import DIM, ToyEmbedder  # noqa: E402

from gnn_pruner.gnn.cross_attention import (AttentionHead, AttentiveCoarse, AttentiveFine,  # noqa: E402
                                            CrossEdgeAttention, cross_chapter_examples, load, save,
                                            windowed_attention)
from gnn_pruner.gnn.features import input_dim  # noqa: E402
from gnn_pruner.gnn.model import CoarseGNN, ModelConfig  # noqa: E402
from gnn_pruner.gnn.spec_qa import SpecQA  # noqa: E402
from gnn_pruner.gnn.two_stage import FineGNN, StageConfig, eval_tables, fine_dim, run_stages, save_two_stage  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
H = 16


def test_cross_edge_attention_matches_spec():
    cea = CrossEdgeAttention(H, num_heads=4)
    assert isinstance(cea.attention, torch.nn.MultiheadAttention)
    assert cea.global_context_proj.in_features == cea.global_context_proj.out_features == H
    assert isinstance(cea.alpha, torch.nn.Parameter) and float(cea.alpha) == pytest.approx(0.3)  # D6
    out = cea(torch.randn(1, 7, H), torch.randn(1, H))
    assert out.shape == (1, 7, H)


def test_windowed_attention_is_exact_band_attention():
    torch.manual_seed(0)
    attention = torch.nn.MultiheadAttention(H, 4, batch_first=True)
    h = torch.randn(1, 50, H)
    window = 8
    i = torch.arange(50)
    band = (i.unsqueeze(1) - i.unsqueeze(0)).abs() > window // 2
    full, _ = attention(h, h, h, attn_mask=band)
    assert torch.allclose(windowed_attention(attention, h, window), full, atol=1e-5)


def test_window_is_local():
    torch.manual_seed(0)
    attention = torch.nn.MultiheadAttention(H, 4, batch_first=True)
    h = torch.randn(1, 40, H)
    far = h.clone()
    far[0, 39] += 10.0  # 39 is far outside node 0's window (half = 4)
    a, b = windowed_attention(attention, h, 8), windowed_attention(attention, far, 8)
    assert torch.allclose(a[0, :30], b[0, :30], atol=1e-6)
    assert not torch.allclose(a[0, 39], b[0, 39])


def test_global_boost_is_bounded_and_summary_can_come_from_outside():
    torch.manual_seed(0)
    head = AttentionHead(H, heads=4, window=8, top_k=5)
    h, local = torch.randn(30, H), torch.randn(30)
    out = head(h, local)
    # The correction head starts at zero, so the change is only the bounded global boost.
    assert torch.all(out - local >= 0) and torch.all(out - local <= float(head.beta) + 1e-6)
    inside = head.summary(h, local)
    assert torch.allclose(head(h, local, global_summary=inside), out)   # G5: broadcast summary
    assert not torch.allclose(head(h, local, global_summary=torch.randn(H)), out)


def test_ablations_switch_parts_off():
    torch.manual_seed(0)
    h, local = torch.randn(12, H), torch.randn(12)
    no_global = AttentionHead(H, heads=4, window=8, use_global=False)
    assert torch.allclose(no_global(h, local), local)  # zero-initialised head, no boost
    no_local = AttentionHead(H, heads=4, window=8, use_local=False)
    assert no_local(h, local).shape == local.shape


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    torch.manual_seed(0)
    coarse = CoarseGNN(ModelConfig(in_dim=input_dim(DIM), hidden=8, skip=True))
    fine = FineGNN(fine_dim(DIM), H)
    base = tmp_path_factory.mktemp("base")
    stage = {"theta": 1.0, "cap": 0.2, "min_keep": 3, "merge_ratio": 0.8}
    save_two_stage(base, coarse, fine, {"embedder": ToyEmbedder.name, "pooling": "mean", "stage_config": stage})
    head_cfg = {"window": 8, "top_k": 5, "use_local": True, "use_global": True}
    a_coarse = AttentiveCoarse(coarse, AttentionHead(coarse.hidden_dim, heads=4, **head_cfg))
    a_fine = AttentiveFine(fine, AttentionHead(fine.hidden_dim, heads=4, **head_cfg))
    path = tmp_path_factory.mktemp("cea")
    save(path, a_coarse, a_fine, {"base": str(base), "embedder": ToyEmbedder.name, "pooling": "mean",
                                  "stage_config": stage,
                                  "cea": {"coarse": {**head_cfg, "heads": 4}, "fine": {**head_cfg, "heads": 4}}})
    return path


def test_save_load_and_pipeline(models):
    coarse, fine, meta = load(str(models))
    assert isinstance(coarse, AttentiveCoarse) and isinstance(fine, AttentiveFine)
    tables, examples = eval_tables(str(ROOT / "eval" / "dev_crosschapter_v1.jsonl"), ToyEmbedder())
    cross = cross_chapter_examples(tables, examples)
    assert len(cross) == len(examples) == 10  # every dev question spans two chapters
    e = cross[0]
    result = run_stages(coarse, fine, tables[e.doc_id], e.question, e.query_vector, StageConfig(**meta["stage_config"]))
    assert len(result.fine_probs) == len(result.rows) > 0
    qa = SpecQA.from_file(str(ROOT / "examples" / "sample_contract.md"), str(models), embedder=ToyEmbedder())
    assert qa.ask("Does the liability cap apply to indemnity claims?").evidence.items
