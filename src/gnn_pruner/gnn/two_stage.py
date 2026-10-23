"""G3: graph coarsening + FineGNN, the spec's three-stage scoring (spec §3).

Stage 1, coarse. Paragraph super-nodes (mean-pooled sentence embeddings, spec §3.2)
are scored by CoarseGNN. Kept: ``score > theta * mean(scores)``, capped to the top
``cap`` share of paragraphs, with at least ``min_keep`` (decision D2).

Stage 2, fine. Kept paragraphs are expanded into their sentences. The fine graph holds
those sentences, their paragraphs and heading ancestors, with hierarchical, sequential
and reference edges. ``FineGNN`` follows spec §3.4: the parent's input row goes through
a learned ``context_projection``, is added with a trainable ``alpha`` (initialised to
0.3), then 2-layer GraphSAGE and an output projection. Inputs (D4) are the node features
of G2 plus the parent paragraph's coarse probability.

Stage 3, merge. Sentences of discarded paragraphs are dropped. Sentences above
``merge_ratio * mean`` of the fine scores are kept (spec phase 6: 80% of the mean).

Training:
1. CoarseGNN on pooled super-nodes (as G2)
2. FineGNN with the coarse model frozen; training adds the silver-positive paragraphs
   to the kept set, so the fine model sees positives
3. joint fine-tune of both (spec §9 phase 3)

Baseline ("v2.0, no coarsening", spec Table 3): FineGNN over every node of the
document, with no coarse probability.

    python -m gnn_pruner.gnn.two_stage train --corpus data/corpus_v1 --out models/two_stage_v1
    python -m gnn_pruner.gnn.two_stage evaluate --model models/two_stage_v1 --sets corpus-val eval/contracts_v1.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GraphSAGE

from ..index import BM25, tokenize
from ..layers import HIERARCHICAL, REFERENCE, LayeredGraph
from .features import KINDS, DocTensors, _kind, coverage, doc_tensors, input_dim, node_features, targets
from .model import CoarseGNN, ModelConfig
from .train import (Example, file_hash, mean_metrics, paragraph_metrics, supervised_loss,
                    train_supervised)

FINE_STRUCTURE = 4 + len(KINDS) + 2   # layer one-hot (4), kind one-hot, position, log children
FINE_SCALARS = 4                       # cos, cos min-max, bm25, coverage


def fine_dim(d: int) -> int:
    return 2 * d + FINE_SCALARS + FINE_STRUCTURE + 1  # + parent coarse probability


@dataclass
class StageConfig:
    theta: float = 1.0        # keep paragraphs with p > theta * mean(p)...
    cap: float = 0.2          # ...at most this share of paragraphs (spec: top 10-20%)
    min_keep: int = 5         # ...and at least this many
    merge_ratio: float = 0.8  # stage 3: keep sentences with p > ratio * mean (spec phase 6)
    max_keep: Optional[int] = None  # optional absolute cap: the spec's "top 10-20%" grows with
                                    # the document, so the fine stage scales linearly without it


# ---------------------------------------------------------------- the table

@dataclass
class FineTable:
    """Every L1-L4 node of a document, for building fine graphs over any subset."""

    doc: DocTensors                     # coarse view (pooled super-nodes)
    ids: List[str]
    row: Dict[str, int]
    emb: np.ndarray
    layer: np.ndarray
    structure: np.ndarray
    parent: np.ndarray                  # parent row, -1 for none
    coarse_row: np.ndarray              # row in doc of the node's L3 ancestor (or itself if L1-L3)
    bm25: BM25
    claims: Dict[int, List[int]]        # L3 row -> its L4 rows
    hierarchical: np.ndarray            # (E, 2) parent, child
    reference: np.ndarray               # (E, 2)

    @property
    def sentences(self) -> np.ndarray:
        return np.where(self.layer == 4)[0]


def fine_table(layered: LayeredGraph, ids: Sequence[str], matrix: np.ndarray,
               pooling: str = "mean") -> FineTable:
    doc = doc_tensors(layered, ids, matrix, pooling)
    vectors = {nid: matrix[i] for i, nid in enumerate(ids)}
    nodes = sorted(layered.nodes.values(), key=lambda x: (x.position, x.layer, x.id))
    row = {n.id: i for i, n in enumerate(nodes)}
    emb = np.zeros((len(nodes), doc.dim), dtype=np.float32)
    for i, n in enumerate(nodes):
        emb[i] = doc.emb[doc.row[n.id]] if n.layer <= 3 else vectors.get(n.id, doc.emb[doc.row[n.parent_id]])
    total = max(len(nodes) - 1, 1)
    children: Dict[str, int] = {}
    for n in nodes:
        if n.parent_id is not None:
            children[n.parent_id] = children.get(n.parent_id, 0) + 1
    structure = np.zeros((len(nodes), FINE_STRUCTURE), dtype=np.float32)
    parent = np.full(len(nodes), -1, dtype=np.int64)
    coarse_row = np.zeros(len(nodes), dtype=np.int64)
    claims: Dict[int, List[int]] = {}
    for i, n in enumerate(nodes):
        structure[i, n.layer - 1] = 1.0
        structure[i, 4 + KINDS.index(_kind(n.node_type))] = 1.0
        structure[i, 4 + len(KINDS)] = i / total
        structure[i, 5 + len(KINDS)] = math.log1p(children.get(n.id, 0)) / 5.0
        if n.parent_id in row:
            parent[i] = row[n.parent_id]
        owner = n.id if n.layer <= 3 else n.parent_id
        coarse_row[i] = doc.row[owner]
        if n.layer == 4:
            claims.setdefault(row[n.parent_id], []).append(i)
    hier = [(row[e.source], row[e.target]) for e in layered.edges if e.edge_type == HIERARCHICAL]
    ref = [(row[e.source], row[e.target]) for e in layered.edges if e.edge_type == REFERENCE]
    texts = [(n.title or n.text) if n.layer < 3 else n.text for n in nodes]
    return FineTable(doc, [n.id for n in nodes], row, emb, np.array([n.layer for n in nodes]), structure,
                     parent, coarse_row, BM25([tokenize(t) for t in texts]), claims,
                     np.array(hier, dtype=np.int64).reshape(-1, 2), np.array(ref, dtype=np.int64).reshape(-1, 2))


def subset_for(table: FineTable, kept_l3: Sequence[int]) -> np.ndarray:
    """Fine-graph rows: kept paragraphs, their sentences and heading ancestors."""
    rows: Set[int] = set()
    for c in kept_l3:
        r = table.row[table.doc.ids[c]]
        rows.add(r)
        rows.update(table.claims.get(r, []))
        p = table.parent[r]
        while p >= 0:
            rows.add(int(p))
            p = table.parent[p]
    return np.array(sorted(rows), dtype=np.int64)


def fine_inputs(table: FineTable, rows: np.ndarray, question: str, qvec: np.ndarray,
                coarse_probs: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x, edge_index, parent_context) for the fine graph over ``rows``."""
    q = np.asarray(qvec, dtype=np.float32)
    emb = table.emb[rows]
    cos = emb @ q
    lo, hi = float(cos.min()), float(cos.max())
    cos_mm = (cos - lo) / (hi - lo) if hi > lo else np.zeros_like(cos)
    bm25 = table.bm25.scores(tokenize(question))[rows]
    bm25 = bm25 / bm25.max() if bm25.max() > 0 else bm25
    cov = coverage(table.bm25, question)[rows]
    base = np.concatenate([emb, emb * q, np.stack([cos, cos_mm, bm25, cov], 1), table.structure[rows]], 1)
    if coarse_probs is None:
        cp = torch.zeros(len(rows), 1)
    else:
        cp = coarse_probs[torch.from_numpy(table.coarse_row[rows])].unsqueeze(1)
    x = torch.cat([torch.from_numpy(np.ascontiguousarray(base, dtype=np.float32)), cp], 1)
    local = np.full(len(table.ids), -1, dtype=np.int64)
    local[rows] = np.arange(len(rows))
    parent_local = np.where(table.parent[rows] >= 0, local[np.maximum(table.parent[rows], 0)], -1)
    context = torch.zeros_like(x)
    has = np.where(parent_local >= 0)[0]
    if len(has):
        context[torch.from_numpy(has)] = x[torch.from_numpy(parent_local[has])]
    pieces = []
    for pairs in (table.hierarchical, table.reference):
        if len(pairs):
            a, b = local[pairs[:, 0]], local[pairs[:, 1]]
            ok = (a >= 0) & (b >= 0)
            pieces.append(np.stack([a[ok], b[ok]], 1))
    layers = table.layer[rows]
    for layer in (1, 2, 3, 4):  # sequential edges between consecutive kept nodes of a layer
        same = np.where(layers == layer)[0]
        if len(same) > 1:
            pieces.append(np.stack([same[:-1], same[1:]], 1))
    if pieces:
        e = np.concatenate(pieces)
        e = np.unique(np.concatenate([e, e[:, ::-1]]), axis=0)
        edge_index = torch.from_numpy(np.ascontiguousarray(e.T))
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return x, edge_index, context


# ---------------------------------------------------------------- the model

class FineGNN(nn.Module):
    """Spec §3.4, with the input dimension extended by the query features (D4)."""

    def __init__(self, original_dim: int, hidden_dim: int = 128, output_dim: int = 1,
                 use_context: bool = True):
        super().__init__()
        self.original_dim, self.hidden_dim, self.use_context = original_dim, hidden_dim, use_context
        self.context_projection = nn.Linear(original_dim, original_dim)
        self.sage = GraphSAGE(original_dim, hidden_dim, num_layers=2)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, edge_index, parent_context):
        return self.output_projection(self.hidden(x, edge_index, parent_context)).squeeze(-1)

    def hidden(self, x, edge_index, parent_context):
        """GraphSAGE output before the output projection (G4 attends over it)."""
        if self.use_context:
            context_boost = self.context_projection(parent_context)
            x = x + self.alpha * context_boost
        return self.sage(x, edge_index)


# ---------------------------------------------------------------- stages

def retain(doc: DocTensors, probs: np.ndarray, cfg: StageConfig) -> np.ndarray:
    """Stage-1 keep rule (D2) over L3 super-nodes: rows of ``doc``, best first."""
    l3 = doc.l3
    p = probs[l3]
    order = l3[np.argsort(-p, kind="stable")]
    above = int((p > cfg.theta * p.mean()).sum())
    cap = max(cfg.min_keep, math.ceil(cfg.cap * len(l3)))
    keep = min(max(above, min(cfg.min_keep, len(l3))), cap)
    if cfg.max_keep:
        keep = min(keep, cfg.max_keep)
    return order[:keep]


@dataclass
class StageResult:
    kept_l3: np.ndarray
    rows: np.ndarray
    fine_probs: np.ndarray
    coarse_ms: float
    fine_ms: float


@torch.no_grad()
def run_stages(coarse: Optional[CoarseGNN], fine: FineGNN, table: FineTable, question: str,
               qvec: np.ndarray, cfg: StageConfig) -> StageResult:
    """Inference. ``coarse=None`` is the no-coarsening baseline: every node, one pass."""
    started = time.perf_counter()
    if coarse is not None:
        coarse.eval()
        x = node_features(table.doc, question, qvec, coarse.config.use_query)
        probs = torch.sigmoid(coarse(x, table.doc.edges))
        kept = retain(table.doc, probs.numpy(), cfg)
        rows = subset_for(table, kept)
    else:
        probs, kept = None, table.doc.l3
        rows = np.arange(len(table.ids))
    coarse_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    fine.eval()
    x, edge_index, context = fine_inputs(table, rows, question, qvec, probs)
    fine_probs = torch.sigmoid(fine(x, edge_index, context)).numpy()
    return StageResult(kept, rows, fine_probs, coarse_ms, (time.perf_counter() - started) * 1000)


def stage_metrics(table: FineTable, result: StageResult, gold_leaves: Sequence[Set[str]],
                  gold_paras: Sequence[Set[str]], cfg: StageConfig) -> Dict[str, float]:
    sentence_rows = [i for i, r in enumerate(result.rows) if table.layer[r] == 4]
    probs = result.fine_probs[sentence_rows]
    ranked = [table.ids[result.rows[sentence_rows[i]]] for i in np.argsort(-probs, kind="stable")]
    out = paragraph_metrics(ranked, gold_leaves)
    kept_ids = {table.doc.ids[c] for c in result.kept_l3}
    out["stage1_recall"] = float(np.mean([bool(g & kept_ids) for g in gold_paras])) if gold_paras else 0.0
    merged = {ranked[i] for i, p in enumerate(sorted(probs, reverse=True)) if p > cfg.merge_ratio * probs.mean()}
    out["merged_share"] = len(merged) / max(len(table.sentences), 1)
    out["merged_recall"] = float(np.mean([bool(g & merged) for g in gold_leaves])) if gold_leaves else 0.0
    out["fine_nodes"] = float(len(result.rows))
    out["coarse_ms"], out["fine_ms"] = result.coarse_ms, result.fine_ms
    return out


# ---------------------------------------------------------------- training

@dataclass
class FineExample:
    base: Example
    gold_leaves: List[Set[str]] = field(default_factory=list)


def _gold(table: FineTable, snippets: Sequence[str]) -> Tuple[List[Set[str]], List[Set[str]]]:
    from ..silver import gold_nodes

    found = [(h, p) for h, p in gold_nodes(table.doc.layered, snippets) if h]
    return [h for h, _ in found], [p & set(table.doc.row) for _, p in found]


def fine_loss(logits: torch.Tensor, table: FineTable, rows: np.ndarray, positives: Dict[str, float],
              tau: float = 0.1) -> torch.Tensor:
    y = torch.tensor([positives.get(table.ids[r], 0.0) for r in rows], dtype=torch.float32)
    sentence = torch.from_numpy(table.layer[rows] == 4)
    loss = F.binary_cross_entropy_with_logits(logits[sentence], y[sentence], pos_weight=torch.tensor(5.0)) * 0.5
    t = y[sentence]
    if t.sum() > 0:
        p = torch.where(t > 0, torch.exp(t / tau), torch.zeros_like(t))
        loss = loss - (p / p.sum() * F.log_softmax(logits[sentence], 0)).sum()
    return loss


def train_fine(coarse: Optional[CoarseGNN], fine: FineGNN, tables: Dict[str, FineTable],
               train: Sequence[Example], val: Sequence[Example], cfg: StageConfig, joint: bool = False,
               epochs: int = 10, lr: float = 3e-4, batch: int = 8, patience: int = 3, seed: int = 0,
               log: Callable[[str], None] = print) -> Dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    params = list(fine.parameters()) + (list(coarse.parameters()) if joint and coarse is not None else [])
    optimiser = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    best, history, stale = -1.0, [], 0
    best_state = (copy.deepcopy(fine.state_dict()), copy.deepcopy(coarse.state_dict()) if coarse else None)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        fine.train()
        if coarse is not None:
            coarse.train(joint)
        order = list(train)
        rng.shuffle(order)
        total = 0.0
        optimiser.zero_grad()
        for i, e in enumerate(order, 1):
            table = tables[e.doc_id]
            doc = table.doc
            if coarse is not None:
                x = node_features(doc, e.question, e.query_vector, coarse.config.use_query)
                with torch.set_grad_enabled(joint):
                    logits_c = coarse(x, doc.edges)
                probs = torch.sigmoid(logits_c)
                kept = retain(doc, probs.detach().numpy(), cfg)
                positive_l3 = [doc.row[n] for n in e.positives if n in doc.row and doc.layer[doc.row[n]] == 3]
                rows = subset_for(table, sorted(set(kept.tolist()) | set(positive_l3)))
            else:
                probs, rows = None, np.arange(len(table.ids))
            x_f, edge_index, context = fine_inputs(table, rows, e.question, e.query_vector, probs)
            loss = fine_loss(fine(x_f, edge_index, context), table, rows, e.positives)
            if joint and coarse is not None:
                loss = loss + supervised_loss(logits_c, targets(doc, e.positives), doc, tau=0.1)
            (loss / batch).backward()
            total += loss.item()
            if i % batch == 0 or i == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimiser.step()
                optimiser.zero_grad()
        metrics = evaluate_stages(coarse, fine, tables, val, cfg)
        history.append({"epoch": epoch, "loss": round(total / len(order), 4), **metrics,
                        "alpha": round(float(fine.alpha), 4), "seconds": round(time.perf_counter() - started, 1)})
        log(f"{'joint' if joint else 'fine'} epoch {epoch:>2} loss {history[-1]['loss']:.4f} alpha "
            f"{float(fine.alpha):.3f} val {_short(metrics)}")
        if metrics["mrr"] > best:
            best, stale = metrics["mrr"], 0
            best_state = (copy.deepcopy(fine.state_dict()), copy.deepcopy(coarse.state_dict()) if coarse else None)
        else:
            stale += 1
            if stale >= patience:
                break
    fine.load_state_dict(best_state[0])
    if coarse is not None and best_state[1] is not None:
        coarse.load_state_dict(best_state[1])
    return {"history": history, "best_val_mrr": best}


def _short(m: Dict) -> Dict:
    keys = ("recall@10", "mrr", "hit@1", "stage1_recall", "merged_share", "merged_recall", "fine_nodes")
    return {k: m[k] for k in keys if k in m}


def evaluate_stages(coarse: Optional[CoarseGNN], fine: FineGNN, tables: Dict[str, FineTable],
                    examples: Sequence[Example], cfg: StageConfig) -> Dict:
    rows = []
    for e in examples:
        table = tables[e.doc_id]
        leaves, paras = getattr(e, "_gold", None) or _gold(table, e.__dict__.get("snippets", []))
        result = run_stages(coarse, fine, table, e.question, e.query_vector, cfg)
        rows.append(stage_metrics(table, result, leaves, paras, cfg))
    return mean_metrics(rows)


# ---------------------------------------------------------------- data

def load_tables(corpus: str, split: str, embedder, pooling: str = "mean"
                ) -> Tuple[Dict[str, FineTable], List[Example]]:
    from ..parquet_store import read_document, read_embeddings

    root = Path(corpus)
    records = [json.loads(l) for l in (root / f"silver_{split}.jsonl").read_text().splitlines() if l]
    tables: Dict[str, FineTable] = {}
    examples = []
    for r in records:
        if r["doc_id"] not in tables:
            shard = root / "parquet" / r["doc_id"]
            _, ids, matrix = read_embeddings(str(shard))
            tables[r["doc_id"]] = fine_table(read_document(str(shard)), ids, matrix, pooling)
        table = tables[r["doc_id"]]
        e = Example(r["question_id"], r["doc_id"], r["question"],
                    np.asarray(embedder.encode_query(r["question"]), dtype=np.float32), r["positives"], [],
                    r.get("tags", {}))
        e._gold = _gold(table, r["gold_evidence"])
        e.gold = e._gold[1]
        examples.append(e)
    return tables, examples


def eval_tables(path: str, embedder, pooling: str = "mean") -> Tuple[Dict[str, FineTable], List[Example]]:
    """A dev/test set as tables + examples (nothing written; never used for training)."""
    from ..corpus import embed_layered
    from ..document_graph import DocumentGraph
    from ..evaluation import load_cases
    from ..layers import four_layer

    tables: Dict[str, FineTable] = {}
    examples = []
    for case in load_cases(path):
        if not case.answerable:
            continue
        if case.document not in tables:
            layered = four_layer(DocumentGraph.from_file(case.document), case.document)
            _, ids, matrix = embed_layered(layered, embedder)
            tables[case.document] = fine_table(layered, ids, matrix, pooling)
        e = Example(case.id, case.document, case.question,
                    np.asarray(embedder.encode_query(case.question), dtype=np.float32), {}, [], case.tags)
        e._gold = _gold(tables[case.document], case.gold_evidence)
        e.gold = e._gold[1]
        examples.append(e)
    return tables, examples


# ---------------------------------------------------------------- artefacts

def save_two_stage(path: Path, coarse: Optional[CoarseGNN], fine: FineGNN, meta: Dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if coarse is not None:
        torch.save(coarse.state_dict(), path / "coarse.pt")
    torch.save(fine.state_dict(), path / "fine.pt")
    cfg = {"coarse": coarse.config.to_dict() if coarse is not None else None,
           "fine": {"original_dim": fine.original_dim, "hidden_dim": fine.hidden_dim,
                    "use_context": fine.use_context}, **meta}
    (path / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def load_two_stage(path: str) -> Tuple[Optional[CoarseGNN], FineGNN, Dict]:
    path = Path(path)
    meta = json.loads((path / "config.json").read_text())
    coarse = None
    if meta.get("coarse"):
        coarse = CoarseGNN(ModelConfig(**meta["coarse"]))
        coarse.load_state_dict(torch.load(path / "coarse.pt", weights_only=True))
        coarse.eval()
    fine = FineGNN(**meta["fine"])
    fine.load_state_dict(torch.load(path / "fine.pt", weights_only=True))
    fine.eval()
    return coarse, fine, meta


# ---------------------------------------------------------------- CLI

def _train(args) -> None:
    from ..embeddings import load_embedder
    from .train import load_split

    root = Path(args.corpus)
    manifest = json.loads((root / "manifest.json").read_text())
    embedder = load_embedder(manifest["embedder"])
    cfg = StageConfig(theta=args.theta, cap=args.cap, min_keep=args.min_keep)
    out = Path(args.out)
    t0 = time.perf_counter()
    tables, train = load_tables(args.corpus, "train", embedder, args.pooling)
    val_tables, val = load_tables(args.corpus, "val", embedder, args.pooling)
    tables.update(val_tables)
    print(f"loaded {len(train)} / {len(val)} questions in {time.perf_counter() - t0:.1f}s")
    d = next(iter(tables.values())).doc.dim
    meta = {"embedder": manifest["embedder"], "pooling": args.pooling, "stage_config": asdict(cfg),
            "corpus_manifest": file_hash(root / "manifest.json"),
            "silver_train": file_hash(root / "silver_train.jsonl"), "seed": args.seed,
            "created": time.strftime("%Y-%m-%d %H:%M")}
    if args.flat:
        fine = FineGNN(fine_dim(d), args.hidden)
        t = time.perf_counter()
        result = train_fine(None, fine, tables, train, val, cfg, epochs=args.epochs, seed=args.seed)
        save_two_stage(out, None, fine, {**meta, "kind": "flat", **result,
                                          "train_seconds": round(time.perf_counter() - t, 1)})
        return
    # 1. CoarseGNN on pooled super-nodes (G2 recipe).
    coarse = CoarseGNN(ModelConfig(in_dim=input_dim(d), hidden=args.hidden, skip=True))
    docs = {k: t.doc for k, t in tables.items()}
    t = time.perf_counter()
    coarse_result = train_supervised(coarse, (docs, train), (docs, val), epochs=args.epochs, lr=3e-4,
                                     seed=args.seed, tau=0.1)
    coarse_seconds = time.perf_counter() - t
    # 2. FineGNN with the coarse model frozen.
    fine = FineGNN(fine_dim(d), args.hidden, use_context=not args.no_context)
    t = time.perf_counter()
    fine_result = train_fine(coarse, fine, tables, train, val, cfg, epochs=args.epochs, seed=args.seed)
    fine_seconds = time.perf_counter() - t
    save_two_stage(out / "staged", coarse, fine, {**meta, "kind": "two_stage", "coarse_result": coarse_result,
                                                  "fine_result": fine_result,
                                                  "train_seconds": round(coarse_seconds + fine_seconds, 1)})
    # 3. Joint fine-tune.
    t = time.perf_counter()
    joint_result = train_fine(coarse, fine, tables, train, val, cfg, joint=True, epochs=max(3, args.epochs // 2),
                              lr=1e-4, seed=args.seed)
    save_two_stage(out / "joint", coarse, fine, {**meta, "kind": "two_stage_joint", "coarse_result": coarse_result,
                                                 "fine_result": fine_result, "joint_result": joint_result,
                                                 "train_seconds": round(coarse_seconds + fine_seconds
                                                                        + time.perf_counter() - t, 1)})


def _evaluate(args) -> None:
    from ..embeddings import load_embedder

    report = {}
    loaded = {m: load_two_stage(m) for m in args.models}
    embedder = load_embedder(next(iter(loaded.values()))[2]["embedder"])
    for name in args.sets:
        by_pooling = {}
        for pooling in sorted({meta.get("pooling", "mean") for _, _, meta in loaded.values()} | {"mean"}):
            by_pooling[pooling] = (load_tables(args.corpus, "val", embedder, pooling) if name == "corpus-val"
                                   else eval_tables(name, embedder, pooling))
        tables, examples = by_pooling["mean"]
        summary = {}
        for path, (coarse, fine, meta) in loaded.items():
            cfg = StageConfig(**{**meta["stage_config"], **({"theta": args.theta} if args.theta else {}),
                                 **({"cap": args.cap} if args.cap else {})})
            t, ex = by_pooling[meta.get("pooling", "mean")]
            summary[path] = evaluate_stages(coarse, fine, t, ex, cfg)
        # Baselines at sentence level.
        rows: Dict[str, List[Dict]] = {"bm25": [], "dense": []}
        for e in examples:
            table = tables[e.doc_id]
            leaves = e._gold[0]
            s = table.sentences
            bm25 = table.bm25.scores(tokenize(e.question))[s]
            dense = table.emb[s] @ e.query_vector
            for name_, scores in (("bm25", bm25), ("dense", dense)):
                ranked = [table.ids[s[i]] for i in np.argsort(-scores, kind="stable")]
                rows[name_].append(paragraph_metrics(ranked, leaves))
        summary.update({k: mean_metrics(v) for k, v in rows.items()})
        summary["_docs"] = {"documents": len(tables), "questions": len(examples),
                            "mean_nodes": round(float(np.mean([len(t.ids) for t in tables.values()])), 1),
                            "mean_sentences": round(float(np.mean([len(t.sentences) for t in tables.values()])), 1)}
        report[Path(name).stem if name != "corpus-val" else name] = summary
    keys = ["recall@10", "mrr", "hit@1", "stage1_recall", "merged_share", "merged_recall", "fine_nodes",
            "coarse_ms", "fine_ms", "n"]
    for name, summary in report.items():
        print(f"== {name} {summary['_docs']}")
        print(f"{'method':<34}" + "".join(f"{k[:13]:>14}" for k in keys))
        for method, m in summary.items():
            if not method.startswith("_"):
                print(f"{method[-34:]:<34}" + "".join(f"{m.get(k, ''):>14}" for k in keys))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--corpus", default="data/corpus_v1")
    t.add_argument("--out", default="models/two_stage_v1")
    t.add_argument("--pooling", default="mean", choices=["mean", "direct"])
    t.add_argument("--flat", action="store_true", help="no-coarsening baseline (FineGNN over every node)")
    t.add_argument("--no-context", action="store_true", help="ablation: no parent context injection")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--theta", type=float, default=1.0)
    t.add_argument("--cap", type=float, default=0.2)
    t.add_argument("--min-keep", type=int, default=5)
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("evaluate")
    e.add_argument("--models", nargs="+", required=True)
    e.add_argument("--sets", nargs="+", default=["corpus-val"])
    e.add_argument("--corpus", default="data/corpus_v1")
    e.add_argument("--theta", type=float)
    e.add_argument("--cap", type=float)
    e.add_argument("--json")
    args = parser.parse_args()
    _train(args) if args.cmd == "train" else _evaluate(args)


if __name__ == "__main__":
    main()
