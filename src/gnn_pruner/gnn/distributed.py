"""G5: distributed coarse scoring with boundary replication (spec §2.3, §5 phases 2-6).

**Partitioning** (coarse graph: L1-L3)
- ``tree``: hierarchical tree-cut. Whole chapters (L1 subtrees) are assigned to shards
  contiguously, balanced by node count.
- ``metis``: METIS (``pymetis``) on the undirected coarse graph, the spec's partitioner.

**Boundary replication** (halo nodes are computed by a worker but not owned by it)
- ``spec``: the spec as written. The last / first N = 3 nodes (document order) at each
  edge of a contiguous run of owned nodes are copied onto the worker.
- ``exact``: a 2-layer GNN reads neighbours 2 hops away, and the attention window
  reaches ``window / 2`` positions, so this mode also replicates the ±window/2
  positions around the owned nodes and the GNN-depth-hop neighbourhood of all of that.
  Owned scores then equal single-process scores (parity test).

**Protocol** (aggregator ↔ workers; workers never talk to each other)
1. ``stats``: each worker returns min / max cosine and max BM25 over its owned nodes;
   the aggregator reduces them, so features are normalised exactly as in a single
   process.
2. ``forward``: the GNN pass on shard + halo. Each worker returns local logits for its
   owned nodes and its top-K candidates (hidden state, logit).
3. ``finish``: the aggregator takes the global top-K, computes the GlobalSummary (G4) and
   broadcasts it; workers apply cross-edge attention and return final owned logits.
4. Aggregator: keep rule over all paragraphs (D2), FineGNN on the kept paragraphs
   (small, so central), merge (> 0.8 × global mean), and boundary stitching (the merged
   nodes plus their ancestors, with every original edge between them).

Shards are Parquet directories, read memory-mapped:
- ``nodes.parquet``: global row, owned flag, embedding, structure
- ``edges.parquet``: local source, target, type
- ``bm25.parquet``: COO of the document-level BM25 weights for the shard's rows
- ``meta.json``

Workers run as Ray actors, or in process (``backend="local"``, used for parity tests).

    python -m gnn_pruner.gnn.distributed bench --pages 1000 --workers 1 2 4
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..index import tokenize
from .features import EDGE_TYPES, DocTensors, coverage
from .two_stage import FineTable, StageConfig, fine_inputs, retain, subset_for

SPEC_BOUNDARY = 3


# ---------------------------------------------------------------- partitioning

def partition(doc: DocTensors, n_shards: int, method: str = "tree", seed: int = 0) -> np.ndarray:
    """Owner shard of every coarse row."""
    n = len(doc.ids)
    if n_shards <= 1:
        return np.zeros(n, dtype=np.int64)
    if method == "tree":
        unit = np.arange(n)
        for i, nid in enumerate(doc.ids):
            path = doc.layered.path(nid)
            unit[i] = doc.row[path[0]] if path[0] in doc.row else i
        units = sorted(set(unit.tolist()))  # unit ids are rows of chapter headings: document order
        size = {u: 0 for u in units}
        for u in unit:
            size[int(u)] += 1
        owner_of_unit, shard, filled, target = {}, 0, 0, n / n_shards
        for u in units:
            if filled >= target * (shard + 1) and shard < n_shards - 1:
                shard += 1
            owner_of_unit[u] = shard
            filled += size[u]
        return np.array([owner_of_unit[int(u)] for u in unit], dtype=np.int64)
    if method == "metis":
        import pymetis

        adjacency = [set() for _ in range(n)]
        for t in EDGE_TYPES:
            for a, b in doc.edges[t].t().tolist():
                if a != b:
                    adjacency[a].add(b)
        _, parts = pymetis.part_graph(n_shards, adjacency=[sorted(a) for a in adjacency])
        return np.array(parts, dtype=np.int64)
    raise ValueError(f"unknown partition method {method!r}")


def _neighbours(doc: DocTensors) -> List[np.ndarray]:
    adjacency = [[] for _ in range(len(doc.ids))]
    for t in EDGE_TYPES:
        for a, b in doc.edges[t].t().tolist():
            adjacency[a].append(b)
    return [np.unique(np.array(a, dtype=np.int64)) for a in adjacency]


def halo(doc: DocTensors, owner: np.ndarray, shard: int, mode: str = "exact", hops: int = 2,
         window: Optional[int] = None, adjacency: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """Rows a worker computes but does not own."""
    owned = np.where(owner == shard)[0]
    n = len(doc.ids)
    if mode == "spec":
        rows = set()
        is_owned = owner == shard
        starts = owned[np.r_[True, np.diff(owned) > 1]]
        ends = owned[np.r_[np.diff(owned) > 1, True]]
        for s, e in zip(starts, ends):
            rows.update(range(max(0, s - SPEC_BOUNDARY), s))
            rows.update(range(e + 1, min(n, e + 1 + SPEC_BOUNDARY)))
        return np.array(sorted(r for r in rows if not is_owned[r]), dtype=np.int64)
    if mode != "exact":
        raise ValueError(f"unknown halo mode {mode!r}")
    adjacency = adjacency if adjacency is not None else _neighbours(doc)
    area = np.zeros(n, dtype=bool)
    area[owned] = True
    if window:  # the attention window around owned rows (document order)
        half = window // 2
        marks = np.zeros(n + 1, dtype=np.int64)
        np.add.at(marks, np.maximum(owned - half, 0), 1)
        np.add.at(marks, np.minimum(owned + half + 1, n), -1)
        area |= np.cumsum(marks)[:n] > 0
    frontier = np.where(area)[0]
    for _ in range(hops):  # GNN receptive field of everything above
        if not len(frontier):
            break
        reached = np.unique(np.concatenate([adjacency[r] for r in frontier])) if len(frontier) else frontier
        new = reached[~area[reached]]
        area[new] = True
        frontier = new
    area[owned] = False
    return np.where(area)[0]


# ---------------------------------------------------------------- shard files

def write_shards(doc: DocTensors, owner: np.ndarray, out: str, mode: str = "exact", hops: int = 2,
                 window: Optional[int] = None) -> List[Path]:
    """One Parquet directory per shard. Returns the shard paths."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out)
    adjacency = _neighbours(doc) if mode == "exact" else None
    weights = doc.bm25.weights.tocsr()
    paths = []
    for shard in range(int(owner.max()) + 1):
        extra = halo(doc, owner, shard, mode, hops, window, adjacency)
        rows = np.sort(np.concatenate([np.where(owner == shard)[0], extra]))
        local = {int(r): i for i, r in enumerate(rows)}
        path = out / f"shard_{shard:03d}"
        path.mkdir(parents=True, exist_ok=True)
        emb = np.ascontiguousarray(doc.emb[rows], dtype=np.float32)
        structure = np.ascontiguousarray(doc.structure[rows], dtype=np.float32)
        pq.write_table(pa.table({
            "row": pa.array(rows, pa.int64()),
            "owned": owner[rows] == shard,
            "emb": pa.FixedSizeListArray.from_arrays(pa.array(emb.reshape(-1)), emb.shape[1]),
            "structure": pa.FixedSizeListArray.from_arrays(pa.array(structure.reshape(-1)), structure.shape[1]),
        }), path / "nodes.parquet")
        src, dst, kind = [], [], []
        for t in EDGE_TYPES:
            for a, b in doc.edges[t].t().tolist():
                if a in local and b in local:
                    src.append(local[a]), dst.append(local[b]), kind.append(t)
        pq.write_table(pa.table({"source": pa.array(src, pa.int64()), "target": pa.array(dst, pa.int64()),
                                 "edge_type": kind}), path / "edges.parquet")
        sub = weights[rows].tocoo()
        pq.write_table(pa.table({"row": pa.array(sub.row, pa.int64()), "col": pa.array(sub.col, pa.int64()),
                                 "value": pa.array(sub.data, pa.float32())}), path / "bm25.parquet")
        vocab = sorted(doc.bm25.vocab, key=doc.bm25.vocab.get)
        (path / "meta.json").write_text(json.dumps({
            "shard": shard, "rows": len(rows), "owned": int((owner == shard).sum()), "halo": len(extra),
            "mode": mode, "n_coarse": len(doc.ids), "vocab": vocab, "idf": doc.bm25.idf.tolist(),
            "max_idf": doc.bm25.max_idf}))
        paths.append(path)
    return paths


class SliceBM25:
    """Document-level BM25 restricted to a shard's rows (same scores as the full index)."""

    def __init__(self, coo_rows, coo_cols, values, n_rows: int, vocab: Sequence[str], idf, max_idf: float):
        from scipy import sparse

        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.idf = np.asarray(idf, dtype=np.float32)
        self.max_idf = max_idf
        self.n_docs = n_rows
        self.weights = sparse.csc_matrix((values, (coo_rows, coo_cols)), shape=(n_rows, max(len(vocab), 1)))

    def scores(self, query_tokens) -> np.ndarray:
        cols = [self.vocab[t] for t in set(query_tokens) if t in self.vocab]
        if not cols or not self.n_docs:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self.weights[:, cols].sum(axis=1)).ravel().astype(np.float32)


# ---------------------------------------------------------------- worker

class ShardWorker:
    """Holds one shard and the coarse model. Stateless between questions except the
    cache of the question in flight."""

    def __init__(self, shard_dir: str, model_dir: str, threads: Optional[int] = None):
        import pyarrow.parquet as pq

        from .cross_attention import load

        started = time.perf_counter()
        if threads:
            torch.set_num_threads(threads)
        path = Path(shard_dir)
        nodes = pq.read_table(path / "nodes.parquet", memory_map=True)
        meta = json.loads((path / "meta.json").read_text())
        self.rows = nodes.column("row").to_numpy()
        self.owned = nodes.column("owned").to_numpy(zero_copy_only=False)
        dim = nodes.schema.field("emb").type.list_size
        self.emb = nodes.column("emb").combine_chunks().flatten().to_numpy().reshape(-1, dim)
        sdim = nodes.schema.field("structure").type.list_size
        self.structure = nodes.column("structure").combine_chunks().flatten().to_numpy().reshape(-1, sdim)
        e = pq.read_table(path / "edges.parquet", memory_map=True).to_pydict()
        self.edges = {}
        for t in EDGE_TYPES:
            pairs = [(s, d) for s, d, k in zip(e["source"], e["target"], e["edge_type"]) if k == t]
            self.edges[t] = torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs \
                else torch.zeros((2, 0), dtype=torch.long)
        b = pq.read_table(path / "bm25.parquet", memory_map=True)
        self.bm25 = SliceBM25(b.column("row").to_numpy(), b.column("col").to_numpy(), b.column("value").to_numpy(),
                              len(self.rows), meta["vocab"], meta["idf"], meta["max_idf"])
        self.coarse, _, _ = load(model_dir)
        self.coarse.eval()
        self.meta = meta
        self.load_seconds = time.perf_counter() - started
        self._cache: Dict = {}

    def info(self) -> Dict:
        return {"shard": self.meta["shard"], "rows": int(len(self.rows)), "owned": int(self.owned.sum()),
                "halo": int((~self.owned).sum()), "load_seconds": self.load_seconds}

    def stats(self, question: str, qvec: np.ndarray) -> Tuple[float, float, float]:
        started = time.perf_counter()
        q = np.asarray(qvec, dtype=np.float32)
        cos = self.emb @ q
        bm25 = self.bm25.scores(tokenize(question))
        self._cache = {"question": question, "q": q, "cos": cos, "bm25": bm25, "t": time.perf_counter() - started}
        own = self.owned
        return float(cos[own].min()), float(cos[own].max()), float(bm25[own].max()) if own.any() else 0.0

    @torch.no_grad()
    def forward(self, cos_min: float, cos_max: float, bm25_max: float, top_k: int = 100):
        started = time.perf_counter()
        c = self._cache
        cos, q = c["cos"], c["q"]
        cos_mm = (cos - cos_min) / (cos_max - cos_min) if cos_max > cos_min else np.zeros_like(cos)
        bm25 = c["bm25"] / bm25_max if bm25_max > 0 else c["bm25"]
        cov = coverage(self.bm25, c["question"])
        use_query = self.coarse.config.use_query
        product = self.emb * q if use_query else np.zeros_like(self.emb)
        scalars = np.stack([cos, cos_mm, bm25, cov], 1) if use_query else np.zeros((len(cos), 4), np.float32)
        x = torch.from_numpy(np.ascontiguousarray(np.concatenate([self.emb, product, scalars, self.structure], 1),
                                                  dtype=np.float32))
        base = getattr(self.coarse, "base", self.coarse)
        h = base.hidden(x, self.edges)
        local = base.out(h).squeeze(-1)
        c.update(h=h, local=local)
        own = np.where(self.owned)[0]
        k = min(top_k, len(own))
        top = own[np.argsort(-local.numpy()[own], kind="stable")[:k]]
        c["t"] += time.perf_counter() - started
        return (self.rows[own], local.numpy()[own], self.rows[top], local.numpy()[top], h.numpy()[top])

    @torch.no_grad()
    def finish(self, global_summary: Optional[np.ndarray]):
        started = time.perf_counter()
        c = self._cache
        if hasattr(self.coarse, "head_") and global_summary is not None:
            final = self.coarse.head_(c["h"], c["local"], torch.from_numpy(global_summary),
                                      positions=torch.from_numpy(self.rows))
        else:
            final = c["local"]
        own = self.owned
        c["t"] += time.perf_counter() - started
        return self.rows[own], final.numpy()[own], c["t"]


# ---------------------------------------------------------------- aggregator

@dataclass
class DistributedResult:
    coarse_logits: np.ndarray            # per coarse row (document order)
    kept_l3: np.ndarray
    rows: np.ndarray                     # fine graph rows (FineTable)
    fine_probs: np.ndarray
    merged: List[str]
    stitched_edges: List[Tuple[str, str, str]]
    timings: Dict[str, float] = field(default_factory=dict)


class DistributedPruner:
    def __init__(self, table: FineTable, shard_dirs: Sequence[str], model_dir: str, backend: str = "local",
                 stage_config: Optional[StageConfig] = None, threads_per_worker: Optional[int] = None):
        from .cross_attention import load

        self.table = table
        coarse, self.fine, meta = load(model_dir)
        self.config = stage_config or StageConfig(**meta["stage_config"])
        self.backend = backend
        # The global summary must use the same K as the model (G4: 100 by default).
        self.top_k = coarse.head_.top_k if hasattr(coarse, "head_") else 100
        started = time.perf_counter()
        if backend == "ray":
            import ray

            if not ray.is_initialized():
                # Ray's uv integration ships the working directory to workers; keep the
                # (large) data and model directories out of it. Workers get absolute paths.
                ray.init(num_cpus=max(len(shard_dirs), 1), include_dashboard=False, log_to_driver=False,
                         runtime_env={"excludes": ["data/", "models/", "examples/", "*.parquet", ".pytest_cache/"]})
            Remote = ray.remote(ShardWorker)
            model_abs = str(Path(model_dir).resolve())
            self.workers = [Remote.remote(str(Path(d).resolve()), model_abs, threads_per_worker) for d in shard_dirs]
            self._get = ray.get
            self.infos = ray.get([w.info.remote() for w in self.workers])
        elif backend == "local":
            self.workers = [ShardWorker(str(d), model_dir, threads_per_worker) for d in shard_dirs]
            self.infos = [w.info() for w in self.workers]
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.startup_seconds = time.perf_counter() - started

    def _call(self, method: str, *args_per_worker):
        if self.backend == "ray":
            return self._get([getattr(w, method).remote(*a) for w, a in zip(self.workers, args_per_worker)])
        return [getattr(w, method)(*a) for w, a in zip(self.workers, args_per_worker)]

    def coarse_scores(self, question: str, qvec: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        t0 = time.perf_counter()
        n = len(self.workers)
        stats = self._call("stats", *[(question, qvec)] * n)
        cos_min = min(s[0] for s in stats)
        cos_max = max(s[1] for s in stats)
        bm25_max = max(s[2] for s in stats)
        t1 = time.perf_counter()
        forwards = self._call("forward", *[(cos_min, cos_max, bm25_max, self.top_k)] * n)
        t2 = time.perf_counter()
        # Global summary: mean hidden state of the global top-K by local logit.
        cand_logit = np.concatenate([f[3] for f in forwards])
        cand_h = np.concatenate([f[4] for f in forwards])
        top = np.argsort(-cand_logit, kind="stable")[:self.top_k]
        summary = cand_h[top].mean(0).astype(np.float32) if len(top) else None
        t3 = time.perf_counter()
        finals = self._call("finish", *[(summary,)] * n)
        t4 = time.perf_counter()
        logits = np.zeros(len(self.table.doc.ids), dtype=np.float32)
        for rows, values, _ in finals:
            logits[rows] = values
        worker_seconds = [f[2] for f in finals]
        return logits, {"stats_ms": (t1 - t0) * 1000, "forward_ms": (t2 - t1) * 1000,
                        "summary_ms": (t3 - t2) * 1000, "finish_ms": (t4 - t3) * 1000,
                        "coarse_ms": (t4 - t0) * 1000, "max_worker_ms": max(worker_seconds) * 1000,
                        "sum_worker_ms": sum(worker_seconds) * 1000}

    @torch.no_grad()
    def run(self, question: str, qvec: np.ndarray) -> DistributedResult:
        logits, timings = self.coarse_scores(question, qvec)
        t0 = time.perf_counter()
        probs = 1 / (1 + np.exp(-logits))
        kept = retain(self.table.doc, probs, self.config)
        rows = subset_for(self.table, kept)
        x, edge_index, context = fine_inputs(self.table, rows, question, qvec, torch.from_numpy(probs))
        self.fine.eval()
        fine_probs = torch.sigmoid(self.fine(x, edge_index, context)).numpy()
        t1 = time.perf_counter()
        table = self.table
        sentence = [i for i, r in enumerate(rows) if table.layer[r] == 4]
        p = fine_probs[sentence]
        merged = [table.ids[rows[i]] for i, v in zip(sentence, p) if v > self.config.merge_ratio * p.mean()]
        stitched = stitch(table, merged)
        t2 = time.perf_counter()
        timings.update(fine_ms=(t1 - t0) * 1000, merge_stitch_ms=(t2 - t1) * 1000,
                       total_ms=timings["coarse_ms"] + (t2 - t0) * 1000)
        return DistributedResult(logits, kept, rows, fine_probs, merged, stitched, timings)

    def shutdown(self) -> None:
        if self.backend == "ray":
            import ray

            for w in self.workers:
                ray.kill(w)


def stitch(table: FineTable, merged: Sequence[str]) -> List[Tuple[str, str, str]]:
    """Boundary stitching: the merged nodes plus their ancestors, with every original
    edge between them, whichever shard each end came from."""
    layered = table.doc.layered
    keep = set(merged)
    for nid in merged:
        keep.update(layered.path(nid))
    return [(e.source, e.target, e.edge_type) for e in layered.edges if e.source in keep and e.target in keep]


# ---------------------------------------------------------------- bench

def single_process_scores(table: FineTable, model_dir: str, question: str, qvec: np.ndarray):
    from .cross_attention import load
    from .features import node_features

    coarse, _, _ = load(model_dir)
    started = time.perf_counter()
    with torch.no_grad():
        x = node_features(table.doc, question, qvec, coarse.config.use_query)
        logits = coarse(x, table.doc.edges).numpy()
    return logits, (time.perf_counter() - started) * 1000


def _bench(args) -> None:
    import tempfile

    from ..corpus import embed_layered, generate_contract
    from ..document_graph import DocumentGraph
    from ..embeddings import load_embedder
    from ..layers import four_layer
    from ..parquet_store import read_document, read_embeddings, write_document
    from ..silver import gold_nodes
    from .cross_attention import load
    from .two_stage import fine_table

    _, _, meta = load(args.model)
    embedder = load_embedder(meta["embedder"])
    window = None
    coarse, _, _ = load(args.model)
    if hasattr(coarse, "head_"):
        window = coarse.head_.window
    hops = getattr(getattr(coarse, "base", coarse).config, "layers", 2)
    results = []
    for pages in args.pages:
        cache = Path(args.cache) / f"bench{pages}"
        doc = generate_contract(f"bench{pages}", pages, seed=21, questions_per_doc=args.questions)
        if not (cache / f"bench{pages}" / "embeddings.parquet").exists():
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "doc.md"
                p.write_text(doc.text)
                layered = four_layer(DocumentGraph.from_file(str(p)), f"bench{pages}")
            write_document(layered, str(cache), embed_layered(layered, embedder))
        t = time.perf_counter()
        layered = read_document(str(cache / f"bench{pages}"))
        _, ids, matrix = read_embeddings(str(cache / f"bench{pages}"))
        table = fine_table(layered, ids, matrix, meta.get("pooling", "mean"))
        table_s = time.perf_counter() - t
        qs = [(q, np.asarray(embedder.encode_query(q.question), dtype=np.float32)) for q in doc.questions]
        single = [single_process_scores(table, args.model, q.question, v) for q, v in qs]
        row = {"pages": pages, "coarse_nodes": len(table.doc.ids), "table_s": round(table_s, 1),
               "single_coarse_ms": round(float(np.mean([s[1] for s in single])), 1), "configs": []}
        for method in args.partition:
            for mode in args.halo:
                for workers in args.workers:
                    owner = partition(table.doc, workers, method)
                    shard_root = Path(tempfile.mkdtemp(prefix=f"shards{pages}_"))
                    t = time.perf_counter()
                    dirs = write_shards(table.doc, owner, str(shard_root), mode, hops, window)
                    write_s = time.perf_counter() - t
                    threads = max(1, (args.cpus // workers))
                    pruner = DistributedPruner(table, dirs, args.model, args.backend, threads_per_worker=threads)
                    timings, diffs, stage1, merged_recall, agree = [], [], [], [], []
                    for (q, v), (ref, _) in zip(qs, single):
                        result = pruner.run(q.question, v)
                        timings.append(result.timings)
                        diffs.append(float(np.abs(result.coarse_logits - ref).max()))
                        l3 = table.doc.l3
                        top_ref = set(l3[np.argsort(-ref[l3])[:50]].tolist())
                        top_dist = set(l3[np.argsort(-result.coarse_logits[l3])[:50]].tolist())
                        agree.append(len(top_ref & top_dist) / 50)
                        found = [(h, p) for h, p in gold_nodes(layered, q.gold_evidence) if h]
                        kept = {table.doc.ids[k] for k in result.kept_l3}
                        stage1.append(np.mean([bool(p & kept) for _, p in found]))
                        merged_recall.append(np.mean([bool(h & set(result.merged)) for h, _ in found]))
                    pruner.shutdown()
                    halo_share = sum(i["halo"] for i in pruner.infos) / len(table.doc.ids)
                    cfg = {"partition": method, "halo": mode, "workers": workers, "threads_per_worker": threads,
                           "write_shards_s": round(write_s, 1), "startup_s": round(pruner.startup_seconds, 1),
                           "shard_load_s_max": round(max(i["load_seconds"] for i in pruner.infos), 2),
                           "halo_share": round(halo_share, 3),
                           "max_abs_logit_diff": round(float(np.max(diffs)), 6),
                           "top50_agreement": round(float(np.mean(agree)), 4),
                           "stage1_recall": round(float(np.mean(stage1)), 4),
                           "merged_recall": round(float(np.mean(merged_recall)), 4)}
                    for key in timings[0]:
                        cfg[key] = round(float(np.mean([t[key] for t in timings])), 1)
                    row["configs"].append(cfg)
                    print(json.dumps({"pages": pages, **cfg}), flush=True)
        results.append(row)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--model", default="models/two_stage_v1/joint")
    b.add_argument("--pages", nargs="+", type=int, default=[1000])
    b.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    b.add_argument("--partition", nargs="+", default=["tree"], choices=["tree", "metis"])
    b.add_argument("--halo", nargs="+", default=["exact"], choices=["exact", "spec"])
    b.add_argument("--backend", default="ray", choices=["ray", "local"])
    b.add_argument("--cpus", type=int, default=8)
    b.add_argument("--questions", type=int, default=8)
    b.add_argument("--cache", default="data/bench")
    b.add_argument("--json")
    args = parser.parse_args()
    _bench(args)


if __name__ == "__main__":
    main()
