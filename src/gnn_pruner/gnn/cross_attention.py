"""G4: cross-edge attention (spec §4).

Two systems on top of a GNN's node representations ``h`` (in document order):

- **Local sparse attention.** A sliding window of ``window`` nodes (512,
  Longformer-style): node i attends to nodes j with |i - j| <= window / 2. It is
  computed block by block, so cost is O(n · window) and it runs on 10,000-page
  documents.
- **Global summary.** ``GlobalSummary = mean(h of the top-K nodes by local score)``
  (K = 100). In G5 the aggregator computes it across workers and broadcasts it; it is
  the only cross-shard input, so ``forward`` also accepts it from outside.

``CrossEdgeAttention`` is the spec §4.4 module (``nn.MultiheadAttention`` +
``global_context_proj``), with ``self.alpha`` defined as a trainable parameter
initialised to 0.3 (decision D6: the spec code uses it without defining it).

The refined score follows spec §4.2, plus a learned correction from the attended
representation:

    final(v) = local(v) + head(attended_v) + beta * sigmoid(cos(attended_v, GlobalSummary))

``beta`` (the §4.2 "alpha", initialised to 0.3) bounds the global boost to
[0, beta]. ``AttentiveCoarse`` / ``AttentiveFine`` wrap the G3 models with the same
call signatures, so the two-stage pipeline (keep rule, merge) runs unchanged.

Training (curriculum, spec §4 / §9): base GNN frozen, attention trained; then joint.

    python -m gnn_pruner.gnn.cross_attention train --base models/two_stage_v1/joint --out models/cea_v1
    python -m gnn_pruner.gnn.cross_attention evaluate --models models/two_stage_v1/joint models/cea_v1/joint
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

WINDOW = 512
TOP_K = 100


class CrossEdgeAttention(nn.Module):
    """Spec §4.4, with ``self.alpha`` defined (D6)."""

    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.global_context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, local_emb, global_summary, mask=None, window: Optional[int] = None,
                positions: Optional[torch.Tensor] = None):
        if window is None:
            attn_local, _ = self.attention(local_emb, local_emb, local_emb, attn_mask=mask)
        else:
            attn_local = windowed_attention(self.attention, local_emb, window, positions)
        gc = self.global_context_proj(global_summary)
        gc_exp = gc.unsqueeze(1).expand(-1, attn_local.size(1), -1)
        return attn_local + self.alpha * gc_exp


def windowed_attention(attention: nn.MultiheadAttention, h: torch.Tensor, window: int,
                       positions: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Exact band attention (|pos_i - pos_j| <= window // 2), one query block of
    ``window`` nodes at a time against the keys within reach (memory O(window^2) per
    block). ``positions``: document positions of the nodes, sorted (default 0..n-1). A
    G5 worker holds non-contiguous rows, so the band must use document positions,
    not local indices."""
    n, half = h.size(1), window // 2
    pos = torch.arange(n) if positions is None else positions.long()
    blocks = []
    for start in range(0, n, window):
        end = min(n, start + window)
        ks = int(torch.searchsorted(pos, pos[start] - half, right=False))
        ke = int(torch.searchsorted(pos, pos[end - 1] + half, right=True))
        qi = pos[start:end].unsqueeze(1)
        ki = pos[ks:ke].unsqueeze(0)
        mask = (qi - ki).abs() > half  # True = may not attend
        out, _ = attention(h[:, start:end], h[:, ks:ke], h[:, ks:ke], attn_mask=mask)
        blocks.append(out)
    return torch.cat(blocks, dim=1)


class AttentionHead(nn.Module):
    """Cross-edge attention plus the §4.2 score refinement for one GNN."""

    def __init__(self, hidden_dim: int, heads: int = 8, window: int = WINDOW, top_k: int = TOP_K,
                 use_local: bool = True, use_global: bool = True):
        super().__init__()
        self.window, self.top_k = window, top_k
        self.use_local, self.use_global = use_local, use_global
        self.cea = CrossEdgeAttention(hidden_dim, heads)
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head.weight)  # start as the base model: no correction
        nn.init.zeros_(self.head.bias)
        self.beta = nn.Parameter(torch.tensor(0.3))

    def summary(self, h: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        k = min(self.top_k, h.size(0))
        return h[torch.topk(local, k).indices].mean(0)

    def forward(self, h: torch.Tensor, local: torch.Tensor,
                global_summary: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        gs = self.summary(h, local) if global_summary is None else global_summary
        if self.use_local:
            attended = self.cea(h.unsqueeze(0), gs.unsqueeze(0) if self.use_global else torch.zeros_like(gs).unsqueeze(0),
                                window=self.window, positions=positions).squeeze(0)
        else:
            gc = self.cea.global_context_proj(gs) if self.use_global else torch.zeros_like(gs)
            attended = h + self.cea.alpha * gc
        out = local + self.head(attended).squeeze(-1)
        if self.use_global:
            out = out + self.beta * torch.sigmoid(F.cosine_similarity(attended, gs.unsqueeze(0), dim=-1))
        return out


class AttentiveCoarse(nn.Module):
    """CoarseGNN + attention head; same interface as CoarseGNN."""

    def __init__(self, base, head: AttentionHead):
        super().__init__()
        self.base, self.head_ = base, head
        self.config = base.config

    def forward(self, x, edges, global_summary=None):
        h = self.base.hidden(x, edges)
        local = self.base.out(self.base.dropout(h)).squeeze(-1)
        return self.head_(h, local, global_summary)


class AttentiveFine(nn.Module):
    """FineGNN + attention head; same interface as FineGNN."""

    def __init__(self, base, head: AttentionHead):
        super().__init__()
        self.base, self.head_ = base, head
        self.original_dim, self.hidden_dim, self.use_context = base.original_dim, base.hidden_dim, base.use_context

    @property
    def alpha(self):
        return self.base.alpha

    def forward(self, x, edge_index, parent_context, global_summary=None):
        h = self.base.hidden(x, edge_index, parent_context)
        local = self.base.output_projection(h).squeeze(-1)
        return self.head_(h, local, global_summary)


def freeze(module: nn.Module, frozen: bool = True) -> None:
    for p in module.parameters():
        p.requires_grad_(not frozen)


# ---------------------------------------------------------------- artefacts

def save(path: Path, coarse, fine, meta: Dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(coarse.state_dict(), path / "coarse.pt")
    torch.save(fine.state_dict(), path / "fine.pt")
    (path / "config.json").write_text(json.dumps(meta, indent=2) + "\n")


def _resolve_base(base: str, model_path: Path) -> Path:
    """A relative base path is relative to the project root: try the working directory,
    then each ancestor of the model directory (Ray workers run elsewhere)."""
    candidate = Path(base)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    for parent in model_path.resolve().parents:
        if (parent / candidate / "config.json").exists():
            return parent / candidate
    return candidate


def load(path: str):
    """Two-stage models with attention heads (or a plain G3 directory)."""
    from .two_stage import load_two_stage

    path = Path(path)
    meta = json.loads((path / "config.json").read_text())
    if "cea" not in meta:
        return load_two_stage(str(path))
    coarse, fine, base_meta = load_two_stage(str(_resolve_base(meta["base"], path)))
    cfg = meta["cea"]
    coarse = AttentiveCoarse(coarse, AttentionHead(coarse.hidden_dim, **cfg["coarse"])) if cfg.get("coarse") else coarse
    fine = AttentiveFine(fine, AttentionHead(fine.hidden_dim, **cfg["fine"])) if cfg.get("fine") else fine
    coarse.load_state_dict(torch.load(path / "coarse.pt", weights_only=True))
    fine.load_state_dict(torch.load(path / "fine.pt", weights_only=True))
    coarse.eval()
    fine.eval()
    return coarse, fine, {**base_meta, **meta}


# ---------------------------------------------------------------- readings

def chapters_of(table, paragraph_ids) -> set:
    return {table.doc.layered.path(p)[0] for p in paragraph_ids}


def cross_chapter_examples(tables, examples) -> List:
    """Questions whose answers sit in two or more L1 chapters."""
    out = []
    for e in examples:
        table = tables[e.doc_id]
        paras = set().union(*e._gold[1]) if e._gold[1] else set()
        if len(chapters_of(table, paras)) >= 2:
            out.append(e)
    return out


def cross_chapter_metrics(coarse, fine, tables, examples, cfg) -> Dict[str, float]:
    """complete = every answer snippet found (all chapters), at each stage."""
    from .two_stage import run_stages
    from .train import rank_l3, scores_for

    rows = []
    for e in examples:
        table = tables[e.doc_id]
        leaves, paras = e._gold
        scores = scores_for(coarse, table.doc, e.question, e.query_vector)
        top10 = set(rank_l3(table.doc, scores)[:10])
        result = run_stages(coarse, fine, table, e.question, e.query_vector, cfg)
        kept = {table.doc.ids[c] for c in result.kept_l3}
        sentence_rows = [i for i, r in enumerate(result.rows) if table.layer[r] == 4]
        probs = result.fine_probs[sentence_rows]
        merged = {table.ids[result.rows[i]] for i, p in zip(sentence_rows, probs) if p > cfg.merge_ratio * probs.mean()}
        rows.append({
            "paragraph_complete@10": float(all(g & top10 for g in paras)),
            "paragraph_recall@10": float(np.mean([bool(g & top10) for g in paras])),
            "stage1_complete": float(all(g & kept for g in paras)),
            "merged_complete": float(all(g & merged for g in leaves)),
            "merged_recall": float(np.mean([bool(g & merged) for g in leaves])),
            "merged_share": len(merged) / max(len(table.sentences), 1),
            "ms": result.coarse_ms + result.fine_ms})
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]} | {"n": len(rows)} if rows else {}


def linear_cross_chapter(sources: Dict[str, str], tables, examples, embedder) -> Dict[str, float]:
    """The rule-based pipeline (explicit reference following), same paragraph metrics."""
    from ..pipeline import DocumentQA
    from .evaluate import _linear_ranking

    qas, rows = {}, []
    for e in examples:
        if e.doc_id not in qas:
            qas[e.doc_id] = DocumentQA.from_file(sources[e.doc_id], embedder=embedder)
        top10 = set(_linear_ranking(qas[e.doc_id], tables[e.doc_id].doc, e.question)[:10])
        paras = e._gold[1]
        rows.append({"paragraph_complete@10": float(all(g & top10 for g in paras)),
                     "paragraph_recall@10": float(np.mean([bool(g & top10) for g in paras]))})
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]} | {"n": len(rows)} if rows else {}


# ---------------------------------------------------------------- CLI

def _train(args) -> None:
    from ..embeddings import load_embedder
    from .train import file_hash, train_supervised
    from .two_stage import StageConfig, load_tables, load_two_stage, train_fine

    coarse, fine, base_meta = load_two_stage(args.base)
    embedder = load_embedder(base_meta["embedder"])
    pooling = base_meta.get("pooling", "mean")
    tables, train = load_tables(args.corpus, "train", embedder, pooling)
    val_tables, val = load_tables(args.corpus, "val", embedder, pooling)
    tables.update(val_tables)
    docs = {k: t.doc for k, t in tables.items()}
    cfg = StageConfig(**base_meta["stage_config"])
    head_cfg = {"window": args.window, "top_k": args.top_k, "use_local": not args.no_local,
                "use_global": not args.no_global}
    out = Path(args.out)
    meta = {"base": args.base, "embedder": base_meta["embedder"], "pooling": pooling,
            "stage_config": base_meta["stage_config"], "corpus_manifest": file_hash(Path(args.corpus) / "manifest.json"),
            "created": time.strftime("%Y-%m-%d %H:%M"), "seed": args.seed}
    torch.manual_seed(args.seed)
    if args.where in ("coarse", "both"):
        coarse = AttentiveCoarse(coarse, AttentionHead(coarse.hidden_dim, **head_cfg))
    if args.where in ("fine", "both"):
        fine = AttentiveFine(fine, AttentionHead(fine.hidden_dim, **head_cfg))
    meta["cea"] = {"coarse": head_cfg if args.where in ("coarse", "both") else None,
                   "fine": head_cfg if args.where in ("fine", "both") else None}
    results = {}
    started = time.perf_counter()
    for phase, frozen, lr in (("frozen", True, 1e-3), ("joint", False, 1e-4)):
        if isinstance(coarse, AttentiveCoarse):
            freeze(coarse.base, frozen)
            results[f"coarse_{phase}"] = train_supervised(coarse, (docs, train), (docs, val), epochs=args.epochs,
                                                          lr=lr, seed=args.seed, tau=0.1, patience=2)
        if isinstance(fine, AttentiveFine):
            freeze(fine.base, frozen)
            results[f"fine_{phase}"] = train_fine(coarse, fine, tables, train, val, cfg, epochs=args.epochs,
                                                  lr=lr, seed=args.seed, patience=2)
        save(out / phase, coarse, fine, {**meta, "phase": phase, "results": results,
                                         "train_seconds": round(time.perf_counter() - started, 1)})


def _evaluate(args) -> None:
    from ..embeddings import load_embedder
    from .two_stage import StageConfig, eval_tables, evaluate_stages, load_tables

    loaded = {m: load(m) for m in args.models}
    first_meta = next(iter(loaded.values()))[2]
    embedder = load_embedder(first_meta["embedder"])
    report = {}
    for name in args.sets:
        if name == "corpus-val":
            tables, examples = load_tables(args.corpus, "val", embedder, "mean")
            sources = {d: str(Path(args.corpus) / "docs" / "val" / f"{d}.md") for d in tables}
        else:
            tables, examples = eval_tables(name, embedder, "mean")
            sources = {d: d for d in tables}
        cross = cross_chapter_examples(tables, examples)
        summary = {}
        for path, (coarse, fine, meta) in loaded.items():
            cfg = StageConfig(**meta["stage_config"])
            summary[path] = {"all": evaluate_stages(coarse, fine, tables, examples, cfg),
                             "cross": cross_chapter_metrics(coarse, fine, tables, cross, cfg)}
        if args.linear and cross:
            summary["linear (rule-based)"] = {"cross": linear_cross_chapter(sources, tables, cross, embedder)}
        report[Path(name).stem if name != "corpus-val" else name] = summary
        print(f"== {name}  questions {len(examples)}, cross-chapter {len(cross)}")
        for method, m in summary.items():
            a, c = m.get("all", {}), m.get("cross", {})
            print(f"  {method[-40:]:<40} all: mrr {a.get('mrr', '-')} hit@1 {a.get('hit@1', '-')} "
                  f"merged_recall {a.get('merged_recall', '-')} share {a.get('merged_share', '-')} | cross: {c}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--base", default="models/two_stage_v1/joint")
    t.add_argument("--out", default="models/cea_v1")
    t.add_argument("--corpus", default="data/corpus_v1")
    t.add_argument("--where", choices=["coarse", "fine", "both"], default="both")
    t.add_argument("--window", type=int, default=WINDOW)
    t.add_argument("--top-k", type=int, default=TOP_K)
    t.add_argument("--no-local", action="store_true")
    t.add_argument("--no-global", action="store_true")
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("evaluate")
    e.add_argument("--models", nargs="+", required=True)
    e.add_argument("--sets", nargs="+", default=["corpus-val"])
    e.add_argument("--corpus", default="data/corpus_v1")
    e.add_argument("--linear", action="store_true", help="also the rule-based pipeline on cross-chapter questions")
    e.add_argument("--json")
    args = parser.parse_args()
    _train(args) if args.cmd == "train" else _evaluate(args)


if __name__ == "__main__":
    main()
