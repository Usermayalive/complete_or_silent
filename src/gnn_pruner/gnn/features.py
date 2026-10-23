"""Coarse-graph tensors and per-question node features for CoarseGNN.

The coarse graph is the four-layer DAG without Layer 4: L1 sections, L2 sub-sections,
L3 paragraphs. L3 embeddings come straight from the stored paragraph vectors. An L1/L2
heading's embedding is the normalised mean of the paragraphs below it, since headings
have no vector of their own. (G3 replaces L3 vectors with mean-pooled sentence vectors,
spec §3.2.)

Edges: hierarchical, sequential and reference edges, each in both directions. A
reference from a sentence counts as one from its paragraph.

Node input (decision D4: the spec's formulas have no query input, so every question
would score the same):

    embedding (d)                     what the node says
    embedding * query (d)             which dimensions agree with the question
    cos, cos min-max, bm25 / max,     how well it matches (semantic, lexical)
    term coverage
    layer (3), kind (4), position,    where it sits in the document
    log #children

``use_query=False`` zeroes the query-dependent parts (an ablation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..index import BM25, tokenize
from ..layers import HIERARCHICAL, REFERENCE, SEQUENTIAL, LayeredGraph

EDGE_TYPES = (HIERARCHICAL, SEQUENTIAL, REFERENCE)
KINDS = ("heading", "paragraph", "table", "figure")
SCALARS = ("cos", "cos_minmax", "bm25", "coverage")
STRUCTURE = 3 + len(KINDS) + 2  # layer one-hot, kind one-hot, position, log children


def _kind(node_type: str) -> str:
    if node_type in ("table", "figure"):
        return node_type
    return "paragraph" if node_type in ("paragraph", "clause", "sentence") else "heading"


@dataclass
class DocTensors:
    doc_id: str
    layered: LayeredGraph
    ids: List[str]                    # coarse nodes (L1-L3), document order
    row: Dict[str, int]
    emb: np.ndarray                   # (N, d) float32, L2-normalised
    layer: np.ndarray                 # (N,) 1..3
    structure: np.ndarray             # (N, STRUCTURE)
    edges: Dict[str, torch.Tensor]    # edge type -> (2, E), both directions
    bm25: BM25
    l3: np.ndarray = field(default=None)  # indices of L3 rows

    @property
    def dim(self) -> int:
        return self.emb.shape[1]


def paragraph_vectors(layered: LayeredGraph, vectors: Dict[str, np.ndarray],
                      pooling: str = "direct") -> Dict[str, np.ndarray]:
    """L3 vectors. ``direct``: the paragraph's own embedding. ``mean`` (spec §3.2):
    normalised mean of its Layer-4 sentence embeddings (a virtual L4 node shares the
    paragraph's vector)."""
    out = {}
    for n in layered.nodes.values():
        if n.layer != 3:
            continue
        if pooling == "direct":
            out[n.id] = vectors[n.id]
        elif pooling == "mean":
            claims = [vectors.get(c.id, vectors[n.id]) for c in layered.children(n.id) if c.layer == 4]
            v = np.mean(claims, axis=0) if claims else vectors[n.id]
            out[n.id] = v / (np.linalg.norm(v) or 1.0)
        else:
            raise ValueError(f"unknown pooling {pooling!r}")
    return out


def doc_tensors(layered: LayeredGraph, ids: Sequence[str], matrix: np.ndarray,
                pooling: str = "direct") -> DocTensors:
    """``ids`` / ``matrix``: stored vectors (L3 and L4 rows, as in the Parquet shards).
    ``pooling``: how L3 super-node vectors are formed (``direct`` or ``mean``)."""
    vectors = {nid: matrix[i] for i, nid in enumerate(ids)}
    vectors.update(paragraph_vectors(layered, vectors, pooling))
    coarse = [n for n in sorted(layered.nodes.values(), key=lambda x: (x.position, x.layer, x.id))
              if n.layer <= 3]
    row = {n.id: i for i, n in enumerate(coarse)}
    dim = matrix.shape[1]
    emb = np.zeros((len(coarse), dim), dtype=np.float32)
    for n in coarse:
        if n.layer == 3:
            emb[row[n.id]] = vectors[n.id]
    # Headings: mean of the paragraphs below them (deepest first, so L1 sees L2's).
    below: Dict[str, List[int]] = {}
    for n in coarse:
        if n.layer == 3:
            for a in layered.ancestors(n.id):
                below.setdefault(a.id, []).append(row[n.id])
    for n in coarse:
        if n.layer < 3 and below.get(n.id):
            v = emb[below[n.id]].mean(axis=0)
            emb[row[n.id]] = v / (np.linalg.norm(v) or 1.0)

    total = max(len(coarse) - 1, 1)
    structure = np.zeros((len(coarse), STRUCTURE), dtype=np.float32)
    children: Dict[str, int] = {}
    for n in layered.nodes.values():
        if n.parent_id is not None:
            children[n.parent_id] = children.get(n.parent_id, 0) + 1
    for i, n in enumerate(coarse):
        structure[i, n.layer - 1] = 1.0
        structure[i, 3 + KINDS.index(_kind(n.node_type))] = 1.0
        structure[i, 3 + len(KINDS)] = i / total
        structure[i, 4 + len(KINDS)] = math.log1p(children.get(n.id, 0)) / 5.0

    def lift(nid: str) -> Optional[str]:
        node = layered.nodes.get(nid)
        while node is not None and node.layer > 3:
            node = layered.nodes.get(node.parent_id) if node.parent_id else None
        return node.id if node is not None else None

    pairs: Dict[str, set] = {t: set() for t in EDGE_TYPES}
    for e in layered.edges:
        a, b = lift(e.source), lift(e.target)
        if a is None or b is None or a == b or a not in row or b not in row:
            continue
        pairs[e.edge_type].add((row[a], row[b]))
        pairs[e.edge_type].add((row[b], row[a]))
    edges = {t: (torch.tensor(sorted(p), dtype=torch.long).t().contiguous() if p
                 else torch.zeros((2, 0), dtype=torch.long)) for t, p in pairs.items()}
    texts = [(n.title or n.text) if n.layer < 3 else n.text for n in coarse]
    bm25 = BM25([tokenize(t) for t in texts])
    layer = np.array([n.layer for n in coarse], dtype=np.int64)
    return DocTensors(layered.doc_id, layered, [n.id for n in coarse], row, emb, layer, structure,
                      edges, bm25, np.where(layer == 3)[0])


def coverage(bm25: BM25, question: str) -> np.ndarray:
    """Per node: share of the question's IDF-weighted terms the node contains."""
    tokens = list(dict.fromkeys(tokenize(question)))
    out = np.zeros(bm25.n_docs, dtype=np.float32)
    if not tokens or not bm25.n_docs:
        return out
    weights = np.array([bm25.idf[bm25.vocab[t]] if t in bm25.vocab else bm25.max_idf for t in tokens])
    known = [(bm25.vocab[t], w) for t, w in zip(tokens, weights) if t in bm25.vocab]
    if known:
        present = (bm25.weights[:, [c for c, _ in known]] > 0).astype(np.float32)
        out = np.asarray(present @ np.array([w for _, w in known], dtype=np.float32)).ravel()
        out = out / float(weights.sum())
    return out.astype(np.float32)


def node_features(doc: DocTensors, question: str, query_vector: np.ndarray,
                  use_query: bool = True) -> torch.Tensor:
    """(N, 2d + 4 + STRUCTURE) input matrix for one question."""
    q = np.asarray(query_vector, dtype=np.float32)
    cos = doc.emb @ q
    lo, hi = float(cos.min()), float(cos.max())
    cos_mm = (cos - lo) / (hi - lo) if hi > lo else np.zeros_like(cos)
    bm25 = doc.bm25.scores(tokenize(question))
    bm25 = bm25 / bm25.max() if bm25.max() > 0 else bm25
    scalars = np.stack([cos, cos_mm, bm25, coverage(doc.bm25, question)], axis=1).astype(np.float32)
    product = doc.emb * q
    if not use_query:
        product = np.zeros_like(product)
        scalars = np.zeros_like(scalars)
    x = np.concatenate([doc.emb, product, scalars, doc.structure], axis=1)
    return torch.from_numpy(np.ascontiguousarray(x))


def feature_groups(dim: int) -> Dict[str, slice]:
    """Column ranges of the input, for explanations."""
    return {
        "embedding": slice(0, dim),
        "query_match": slice(dim, 2 * dim),
        "semantic": slice(2 * dim, 2 * dim + 2),
        "lexical": slice(2 * dim + 2, 2 * dim + 4),
        "structure": slice(2 * dim + 4, 2 * dim + 4 + STRUCTURE),
    }


def input_dim(dim: int) -> int:
    return 2 * dim + len(SCALARS) + STRUCTURE


def targets(doc: DocTensors, positives: Dict[str, float]) -> torch.Tensor:
    """Graded silver target per coarse node (0 outside the Silver Positive Set)."""
    y = np.zeros(len(doc.ids), dtype=np.float32)
    for nid, grade in positives.items():
        if nid in doc.row:
            y[doc.row[nid]] = grade
    return torch.from_numpy(y)
