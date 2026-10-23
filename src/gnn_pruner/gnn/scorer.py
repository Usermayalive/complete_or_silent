"""GNNScorer: CoarseGNN behind the Phase 7 ``Scorer`` interface.

The GNN scores the whole coarse graph of the document (spec path), not only the
retrieved subgraph. Each subgraph node then gets:
- L1-L3 nodes: ``sigmoid(logit)``
- L4 sentences / clauses: ``0.95 * paragraph score + 0.05 * own semantic feature``
  (G3's FineGNN replaces this)

Explanations are gradient × input on the node's logit, summed per feature group
(embedding, query_match, semantic, lexical, structure), plus ``neighbours`` = what
reached it through message passing. They explain the logit, so unlike LinearScorer
they do not sum to the score.

No trained model, or a model trained with a different embedder: use ``LinearScorer``
(``load_scorer`` falls back).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ..layers import four_layer
from .features import doc_tensors, feature_groups, node_features
from .train import load

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "coarse_gnn_v2" / "supervised"
SENTENCE_SHARE = 0.05


class GNNScorer:
    name = "gnn"

    def __init__(self, path: str = str(DEFAULT_MODEL), explain_top: int = 15):
        self.path = path
        self.model, self.meta = load(path)
        self.explain_top = explain_top
        self._docs: Dict[int, object] = {}

    def _doc(self, index):
        key = id(index)
        if key not in self._docs:
            if index.embedder.name != self.meta["embedder"]:
                raise ValueError(f"model trained with {self.meta['embedder']}, index uses {index.embedder.name}")
            layered = four_layer(index.graph, "doc")
            l3 = [n.id for n in layered.nodes.values() if n.layer == 3]
            ids = [nid for nid in l3 if nid in index.position]
            matrix = np.vstack([index.matrix[index.position[nid]] for nid in ids])
            self._docs[key] = doc_tensors(layered, ids, matrix)
        return self._docs[key]

    def score_nodes(self, index, ctx, features: Dict[str, Dict[str, float]]
                    ) -> Dict[str, Tuple[float, Dict[str, float]]]:
        doc = self._doc(index)
        x = node_features(doc, ctx.plan.query, ctx.query_vector, self.model.config.use_query)
        x.requires_grad_(True)
        logits = self.model(x, doc.edges)
        probs = torch.sigmoid(logits).detach().numpy()
        out: Dict[str, Tuple[float, Dict[str, float]]] = {}
        for nid, f in features.items():
            if nid in doc.row:
                out[nid] = (float(probs[doc.row[nid]]), {})
                continue
            node = doc.layered.nodes.get(nid)
            parent = node.parent_id if node is not None else None
            if parent in doc.row:
                score = (1 - SENTENCE_SHARE) * probs[doc.row[parent]] + SENTENCE_SHARE * f.get("semantic", 0.0)
                out[nid] = (float(score), {"paragraph": float(probs[doc.row[parent]])})
            else:
                out[nid] = (0.0, {})
        groups = feature_groups(doc.dim)
        top = sorted((nid for nid in out if nid in doc.row), key=lambda n: -out[n][0])[:self.explain_top]
        for nid in top:
            r = doc.row[nid]
            grad, = torch.autograd.grad(logits[r], x, retain_graph=True)
            gi = (grad * x).detach().numpy()
            own = {g: round(float(gi[r, s].sum()), 4) for g, s in groups.items()}
            own["neighbours"] = round(float(gi.sum() - gi[r].sum()), 4)
            out[nid] = (out[nid][0], own)
        return out


def load_scorer(spec: Optional[str]):
    """``None`` / ``"linear"`` -> None (LinearScorer); ``"gnn"`` -> default model;
    ``"gnn:<dir>"`` -> that model."""
    if not spec or spec == "linear":
        return None
    if spec == "gnn":
        return GNNScorer()
    if spec.startswith("gnn:"):
        return GNNScorer(spec[4:])
    raise ValueError(f"unknown scorer {spec!r}")
