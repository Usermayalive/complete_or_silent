"""G1: Apache Parquet storage for four-layer graphs (spec §2.1).

One directory per document (a "shard" in G5):

    <root>/<doc_id>/nodes.parquet        one row per LayerNode
    <root>/<doc_id>/edges.parquet        source, target, edge_type, subtype
    <root>/<doc_id>/embeddings.parquet   node_id + fixed-size float32 vector; model in metadata

Reads use memory-mapped files (``pyarrow.memory_map``), as in the spec. Local disk
stands in for S3/MinIO; the layout is the same.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .layers import LayerEdge, LayeredGraph, LayerNode

NODE_COLUMNS = ("node_id", "layer", "node_type", "text", "parent_id", "position", "title",
                "number", "page", "start", "end", "virtual")


def _pa():
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa, pq


def write_document(layered: LayeredGraph, root: str,
                   embeddings: Optional[Tuple[str, List[str], np.ndarray]] = None) -> Path:
    """Write ``layered`` (and optionally ``(model, node_ids, matrix)``) under ``root``."""
    pa, pq = _pa()
    out = Path(root) / layered.doc_id
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(layered.nodes.values(), key=lambda n: (n.position, n.layer, n.id))
    nodes = pa.table({
        "node_id": [n.id for n in rows],
        "layer": pa.array([n.layer for n in rows], pa.int8()),
        "node_type": [n.node_type for n in rows],
        "text": [n.text for n in rows],
        "parent_id": [n.parent_id for n in rows],
        "position": pa.array([n.position for n in rows], pa.int32()),
        "title": [n.title for n in rows],
        "number": [n.number for n in rows],
        "page": pa.array([n.page for n in rows], pa.int32()),
        "start": pa.array([n.start for n in rows], pa.int64()),
        "end": pa.array([n.end for n in rows], pa.int64()),
        "virtual": [n.virtual for n in rows],
    }).replace_schema_metadata({"doc_id": layered.doc_id, "title": layered.title,
                                "repairs": json.dumps(layered.repairs)})
    pq.write_table(nodes, out / "nodes.parquet")
    edges = pa.table({
        "source": [e.source for e in layered.edges],
        "target": [e.target for e in layered.edges],
        "edge_type": [e.edge_type for e in layered.edges],
        "subtype": [e.subtype for e in layered.edges],
    })
    pq.write_table(edges, out / "edges.parquet")
    if embeddings is not None:
        model, ids, matrix = embeddings
        matrix = np.asarray(matrix, dtype=np.float32)
        vectors = pa.FixedSizeListArray.from_arrays(pa.array(matrix.reshape(-1)), matrix.shape[1])
        table = pa.table({"node_id": ids, "vector": vectors}).replace_schema_metadata(
            {"model": model, "dim": str(matrix.shape[1])})
        pq.write_table(table, out / "embeddings.parquet")
    return out


def read_document(path: str, memory_map: bool = True) -> LayeredGraph:
    _, pq = _pa()
    path = Path(path)
    table = pq.read_table(path / "nodes.parquet", memory_map=memory_map)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    cols = table.to_pydict()
    nodes: Dict[str, LayerNode] = {}
    for i, nid in enumerate(cols["node_id"]):
        nodes[nid] = LayerNode(nid, int(cols["layer"][i]), cols["node_type"][i], cols["text"][i],
                               cols["parent_id"][i], int(cols["position"][i]), cols["title"][i],
                               cols["number"][i], cols["page"][i], int(cols["start"][i]),
                               int(cols["end"][i]), bool(cols["virtual"][i]))
    e = pq.read_table(path / "edges.parquet", memory_map=memory_map).to_pydict()
    edges = [LayerEdge(s, t, et, st) for s, t, et, st in
             zip(e["source"], e["target"], e["edge_type"], e["subtype"])]
    return LayeredGraph(meta.get("doc_id", path.name), meta.get("title", ""), nodes, edges,
                        json.loads(meta.get("repairs", "[]")))


def read_embeddings(path: str, memory_map: bool = True) -> Tuple[str, List[str], np.ndarray]:
    """(model, node_ids, matrix). The matrix is a zero-copy view when memory-mapped."""
    _, pq = _pa()
    table = pq.read_table(Path(path) / "embeddings.parquet", memory_map=memory_map)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    dim = int(meta["dim"])
    flat = table.column("vector").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    return meta["model"], table.column("node_id").to_pylist(), flat.reshape(-1, dim)
