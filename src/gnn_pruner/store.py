"""Local graph + vector store: many document graphs in one SQLite file.

Nodes and edges are real tables, so the store can be queried directly
(``find_nodes``, ``neighbors``) without loading a whole graph, and graphs are
keyed by a content hash so re-ingesting an unchanged file is free.

Embeddings live alongside them:
- ``node_embeddings``: one vector per (document, node, model), joined to the
  node's type / text / metadata. ``search`` runs vector search across documents.
- ``embedding_cache``: vectors keyed by (model, text hash), so a pretrained model
  never embeds the same text twice (``GraphStore`` is a ``VectorCache``).

    store = GraphStore("graphs.db")
    graph = store.load_or_ingest("contract.pdf")
    index = GraphIndex(graph, CachedEmbedder(load_embedder("st"), store))
    store.put_embeddings(doc_id, index)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .document_graph import DocumentGraph
from .models import Edge, Node

if TYPE_CHECKING:
    from .index import GraphIndex


@dataclass
class SearchHit:
    doc_id: str
    node: Node
    score: float


def _blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    title        TEXT,
    source       TEXT,
    content_hash TEXT,
    source_text  TEXT,
    warnings     TEXT,
    definitions  TEXT,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS nodes (
    doc_id    TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    position  INTEGER NOT NULL,
    node_id   TEXT NOT NULL,
    node_type TEXT NOT NULL,
    parent_id TEXT,
    level     INTEGER,
    title     TEXT,
    text      TEXT,
    start     INTEGER,
    "end"     INTEGER,
    page      INTEGER,
    metadata  TEXT,
    PRIMARY KEY (doc_id, node_id)
);
CREATE TABLE IF NOT EXISTS edges (
    doc_id    TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    source    TEXT NOT NULL,
    target    TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight    REAL
);
CREATE TABLE IF NOT EXISTS node_embeddings (
    doc_id     TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    node_id    TEXT NOT NULL,
    model      TEXT NOT NULL,
    contextual INTEGER NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    PRIMARY KEY (doc_id, node_id, model)
);
CREATE TABLE IF NOT EXISTS embedding_cache (
    model     TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL,
    PRIMARY KEY (model, text_hash)
);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(doc_id, node_type);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(doc_id, source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(doc_id, target);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
"""


def file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GraphStore:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # -------------------------------------------------------------- writes

    def put(self, graph: DocumentGraph, doc_id: Optional[str] = None,
            source: Optional[str] = None, content_hash: Optional[str] = None) -> str:
        """Insert or replace a graph; returns its doc_id."""
        content_hash = content_hash or hashlib.sha256(graph.source_text.encode()).hexdigest()
        doc_id = doc_id or content_hash[:16]
        with self.conn:
            self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self.conn.execute(
                "INSERT INTO documents (doc_id, title, source, content_hash, source_text, "
                "warnings, definitions) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, graph.title, source or (graph.root.metadata.get("source") if graph.root else None),
                 content_hash, graph.source_text, json.dumps(graph.warnings),
                 json.dumps(graph.definitions)))
            self.conn.executemany(
                'INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                [(doc_id, i, n.id, n.node_type, n.parent_id, n.level, n.title, n.text,
                  n.start, n.end, n.page, json.dumps(n.metadata))
                 for i, n in enumerate(graph.nodes[nid] for nid in graph.order)])
            self.conn.executemany(
                "INSERT INTO edges VALUES (?, ?, ?, ?, ?)",
                [(doc_id, e.source, e.target, e.edge_type, e.weight) for e in graph.edges])
        return doc_id

    def delete(self, doc_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    # --------------------------------------------------------------- reads

    @staticmethod
    def _node(row: sqlite3.Row) -> Node:
        return Node(id=row["node_id"], node_type=row["node_type"], text=row["text"],
                    parent_id=row["parent_id"], level=row["level"], title=row["title"],
                    start=row["start"], end=row["end"], page=row["page"],
                    metadata=json.loads(row["metadata"] or "{}"))

    def get(self, doc_id: str) -> DocumentGraph:
        doc = self.conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if doc is None:
            raise KeyError(doc_id)
        nodes = [self._node(r) for r in self.conn.execute(
            "SELECT * FROM nodes WHERE doc_id = ? ORDER BY position", (doc_id,))]
        edges = [Edge(r["source"], r["target"], r["edge_type"], r["weight"])
                 for r in self.conn.execute("SELECT * FROM edges WHERE doc_id = ? ORDER BY rowid",
                                            (doc_id,))]
        return DocumentGraph(nodes, source_text=doc["source_text"],
                             warnings=json.loads(doc["warnings"] or "[]"), edges=edges,
                             definitions=json.loads(doc["definitions"] or "{}"))

    def find_by_hash(self, content_hash: str) -> Optional[str]:
        row = self.conn.execute("SELECT doc_id FROM documents WHERE content_hash = ?",
                                (content_hash,)).fetchone()
        return row["doc_id"] if row else None

    def list_documents(self) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT d.doc_id, d.title, d.source, d.created_at, "
            "(SELECT COUNT(*) FROM nodes n WHERE n.doc_id = d.doc_id) AS node_count "
            "FROM documents d ORDER BY d.created_at, d.doc_id")
        return [dict(r) for r in rows]

    def find_nodes(self, doc_id: str, node_type: Optional[str] = None,
                   contains: Optional[str] = None) -> List[Node]:
        sql, args = "SELECT * FROM nodes WHERE doc_id = ?", [doc_id]
        if node_type:
            sql += " AND node_type = ?"
            args.append(node_type)
        if contains:
            sql += " AND text LIKE ?"
            args.append(f"%{contains}%")
        return [self._node(r) for r in self.conn.execute(sql + " ORDER BY position", args)]

    def neighbors(self, doc_id: str, node_id: str, edge_type: Optional[str] = None,
                  direction: str = "out") -> List[Node]:
        near, far = ("source", "target") if direction == "out" else ("target", "source")
        sql = (f"SELECT n.* FROM edges e JOIN nodes n ON n.doc_id = e.doc_id AND n.node_id = e.{far} "
               f"WHERE e.doc_id = ? AND e.{near} = ?")
        args = [doc_id, node_id]
        if edge_type:
            sql += " AND e.edge_type = ?"
            args.append(edge_type)
        return [self._node(r) for r in self.conn.execute(sql + " ORDER BY e.rowid", args)]

    # ------------------------------------------------------------- vectors

    def put_embeddings(self, doc_id: str, index: "GraphIndex") -> int:
        """Persist every node vector of ``index`` for ``doc_id`` (replacing that model's)."""
        if self.conn.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone() is None:
            raise KeyError(doc_id)
        records = index.records()
        model = index.embedder.name
        with self.conn:
            self.conn.execute("DELETE FROM node_embeddings WHERE doc_id = ? AND model = ?",
                              (doc_id, model))
            self.conn.executemany(
                "INSERT INTO node_embeddings VALUES (?, ?, ?, ?, ?, ?)",
                [(doc_id, r.node_id, model, int(index.contextual), len(r.vector), _blob(r.vector))
                 for r in records])
        return len(records)

    def get_embeddings(self, doc_id: str, model: str) -> Dict[str, np.ndarray]:
        rows = self.conn.execute(
            "SELECT node_id, vector FROM node_embeddings WHERE doc_id = ? AND model = ?",
            (doc_id, model))
        return {r["node_id"]: _vector(r["vector"]) for r in rows}

    def search(self, query_vector: np.ndarray, model: str, k: int = 10,
               node_types: Optional[Iterable[str]] = None,
               doc_ids: Optional[Iterable[str]] = None) -> List[SearchHit]:
        """Exact vector search across stored documents, filtered by type / document."""
        sql = ("SELECT e.doc_id, e.vector, n.* FROM node_embeddings e JOIN nodes n "
               "ON n.doc_id = e.doc_id AND n.node_id = e.node_id WHERE e.model = ?")
        args: list = [model]
        for column, values in (("n.node_type", node_types), ("e.doc_id", doc_ids)):
            if values is not None:
                values = list(values)
                sql += f" AND {column} IN ({','.join('?' * len(values))})"
                args.extend(values)
        rows = self.conn.execute(sql, args).fetchall()
        if not rows:
            return []
        matrix = np.stack([_vector(r["vector"]) for r in rows])
        scores = matrix @ np.asarray(query_vector, dtype=np.float32)
        order = np.argsort(-scores)[:k]
        return [SearchHit(rows[i]["doc_id"], self._node(rows[i]), float(scores[i])) for i in order]

    # VectorCache protocol (see embeddings.CachedEmbedder)
    def get_vectors(self, model: str, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        found: Dict[str, np.ndarray] = {}
        keys = list(dict.fromkeys(keys))
        for start in range(0, len(keys), 500):
            chunk = keys[start:start + 500]
            rows = self.conn.execute(
                f"SELECT text_hash, vector FROM embedding_cache WHERE model = ? "
                f"AND text_hash IN ({','.join('?' * len(chunk))})", [model, *chunk])
            found.update({r["text_hash"]: _vector(r["vector"]) for r in rows})
        return found

    def put_vectors(self, model: str, items: Dict[str, np.ndarray]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache VALUES (?, ?, ?, ?)",
                [(model, key, len(v), _blob(v)) for key, v in items.items()])

    # ------------------------------------------------------------ convenience

    def load_or_ingest(self, path: str) -> DocumentGraph:
        """Parse ``path`` only if this exact file content is not already stored."""
        digest = file_hash(path)
        doc_id = self.find_by_hash(digest)
        if doc_id is not None:
            return self.get(doc_id)
        graph = DocumentGraph.from_file(path)
        self.put(graph, source=str(path), content_hash=digest)
        return graph
