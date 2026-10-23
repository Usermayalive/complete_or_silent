"""Lexical (BM25) and dense indexes over graph nodes."""

from __future__ import annotations

import copy
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .document_graph import DocumentGraph
from .embeddings import Embedder, TfidfEmbedder
from .parser import normalize_ws
from .textutil import normalize_numbers, stem
from .models import CONTENT_TYPES, PARAGRAPH, SECTION_TYPES, SEMANTIC_SIMILARITY
from .vector_index import ids_of_types, make_vector_index

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by does do for from has have how if in is it its of on or "
    "that the this to was were what when where which who why will with shall any all "
    "me tell about there their they".split()
)


def tokenize(text: str) -> List[str]:
    """BM25 / phrase tokens: numbers normalised, lower-cased, stopwords removed,
    plurals singularised."""
    return [stem(t) for t in _TOKEN.findall(normalize_numbers(text).lower()) if t not in STOPWORDS]


SEMANTIC_QUANTILE = 0.9  # adaptive threshold: top 10% of paragraph pairs...
SEMANTIC_FLOOR = 0.15    # ...but never below this similarity
SEMANTIC_MODEL_KEY = "semantic_edges_model"


class BM25:
    """Okapi BM25 over a sparse term matrix: per-document term weights are
    precomputed, so a query is a column sum (fast on long documents)."""

    def __init__(self, docs: Sequence[List[str]], k1: float = 1.5, b: float = 0.75):
        from scipy import sparse

        self.k1, self.b = k1, b
        self.vocab: Dict[str, int] = {}
        rows, cols, counts = [], [], []
        for i, doc in enumerate(docs):
            for term, count in Counter(doc).items():
                rows.append(i)
                cols.append(self.vocab.setdefault(term, len(self.vocab)))
                counts.append(count)
        n_docs = len(docs)
        tf = sparse.csr_matrix((np.array(counts, dtype=np.float32), (rows, cols)),
                               shape=(n_docs, max(len(self.vocab), 1)))
        lengths = np.array([len(d) for d in docs], dtype=np.float32)
        avgdl = float(lengths.mean()) if n_docs else 0.0
        df = np.bincount(cols, minlength=tf.shape[1]).astype(np.float32)
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        norm = self.k1 * (1 - self.b + self.b * lengths / max(avgdl, 1e-9))
        weights = tf.tocoo()
        data = idf[weights.col] * weights.data * (self.k1 + 1) / (weights.data + norm[weights.row])
        self.weights = sparse.csc_matrix((data, (weights.row, weights.col)), shape=tf.shape)
        self.n_docs = n_docs
        self.idf = idf
        self.max_idf = float(np.log(1 + (n_docs + 0.5) / 0.5))  # a term no node contains

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        cols = [self.vocab[t] for t in set(query_tokens) if t in self.vocab]
        if not cols or not self.n_docs:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self.weights[:, cols].sum(axis=1)).ravel().astype(np.float32)


_QUOTED = re.compile(r"[\"\u201c\u2018']([^\"\u201c\u201d\u2018\u2019']{3,}?)[\"\u201d\u2019']")


class PhraseIndex:
    """Exact phrase evidence: quoted phrases in the question, and adjacent content-word
    pairs ("sole remedy", "breach notification") shared by question and node."""

    def __init__(self, texts: Sequence[str]):
        self.normalized = [normalize_ws(normalize_numbers(t)).lower() for t in texts]
        self.postings: Dict[Tuple[str, str], List[int]] = {}
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            for pair in set(zip(tokens, tokens[1:])):
                self.postings.setdefault(pair, []).append(i)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.normalized), dtype=np.float32)
        tokens = tokenize(query)
        for pair in set(zip(tokens, tokens[1:])):
            for i in self.postings.get(pair, ()):
                out[i] += 1.0
        for phrase in _QUOTED.findall(query):
            needle = normalize_ws(normalize_numbers(phrase)).lower()
            for i, text in enumerate(self.normalized):
                if needle in text:
                    out[i] += 2.0  # an exact quoted phrase outweighs a shared word pair
        return out

    def quoted_found(self, query: str) -> bool:
        """True if the question quotes a phrase that occurs verbatim in some node."""
        needles = [normalize_ws(normalize_numbers(p)).lower() for p in _QUOTED.findall(query)]
        return any(n in text for n in needles for text in self.normalized)


@dataclass
class EmbeddingRecord:
    """What gets stored per embedded node (Phase 3 schema)."""

    node_id: str
    vector: np.ndarray
    node_type: str
    text: str
    metadata: Dict[str, Any]


class GraphIndex:
    """Embeds content nodes (+ a summary vector per section) and indexes them for
    dense and BM25 retrieval.

    ``contextual=True`` embeds each node as "<section path>. <text>" so that a
    clause like "Fees may not increase during the initial term" carries the
    meaning of its heading ("9.2 Price Changes"). Off by default: on the eval set it
    helped TF-IDF slightly but hurt both dense models (see docs/USE_CASE.md).
    """

    def __init__(self, graph: DocumentGraph, embedder: Optional[Embedder] = None,
                 contextual: bool = False, vector_index: str = "auto",
                 semantic_scaling: str = "minmax",
                 semantic_edges: bool = True, semantic_threshold: Optional[float] = None,
                 semantic_k: int = 2):
        """``semantic_threshold=None`` adapts to the embedder: similarity scales differ
        widely (TF-IDF paragraph pairs rarely exceed 0.4; dense models often sit at 0.3-0.8)."""
        self.graph = graph
        self.contextual = contextual
        self.semantic_scaling = semantic_scaling
        content = graph.content_nodes(CONTENT_TYPES)
        sections = graph.content_nodes(SECTION_TYPES)
        self.ids = [n.id for n in content]
        self.section_ids = [s.id for s in sections]
        self.position: Dict[str, int] = {nid: i for i, nid in enumerate(self.ids)}
        self.section_position: Dict[str, int] = {sid: i for i, sid in enumerate(self.section_ids)}

        section_texts = [self._section_summary(s.id) for s in sections]
        content_texts = [self.embedding_text(n.id) for n in content]

        embedder = embedder or TfidfEmbedder()
        # Corpus-fitted embedders (TF-IDF) are copied so one instance shared across
        # documents cannot refit and silently corrupt another document's index.
        self.embedder = embedder if embedder.cacheable else copy.deepcopy(embedder)
        self.embedder.fit(content_texts + section_texts)
        self.matrix = self._encode(content_texts)
        self.section_matrix = self._encode(section_texts)
        self.vectors = make_vector_index(self.ids, self.matrix, vector_index)
        self.bm25 = BM25([tokenize(n.text) for n in content])
        self.phrases = PhraseIndex([n.text for n in content])

        if semantic_edges:
            self._link_semantic(semantic_threshold, semantic_k)

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return self.embedder.encode(texts)

    def embedding_text(self, node_id: str) -> str:
        node = self.graph.nodes[node_id]
        if not self.contextual:
            return node.text
        section = self.graph.section_of(node_id)
        return f"{self.graph.path(section.id)}. {node.text}" if section else node.text

    def _section_summary(self, section_id: str) -> str:
        node = self.graph.nodes[section_id]
        body = [c.text for c in self.graph.children(section_id) if c.node_type in CONTENT_TYPES]
        return " ".join([node.title or "", node.title or ""] + body)  # title counted twice

    def _link_semantic(self, threshold: Optional[float], k: int) -> None:
        """Add semantic_similarity edges, rebuilding them if another model made them."""
        root_meta = self.graph.root.metadata if self.graph.root else {}
        existing = any(e.edge_type == SEMANTIC_SIMILARITY for e in self.graph.edges)
        if existing and root_meta.get(SEMANTIC_MODEL_KEY) == self.embedder.name:
            return
        if existing:
            self.graph.remove_edges(SEMANTIC_SIMILARITY)
        self.graph.add_semantic_edges(self._semantic_pairs(threshold, k))
        root_meta[SEMANTIC_MODEL_KEY] = self.embedder.name

    def _semantic_pairs(self, threshold: Optional[float], k: int):
        para_idx = [i for i, nid in enumerate(self.ids)
                    if self.graph.nodes[nid].node_type == PARAGRAPH]
        if len(para_idx) < 2:
            return []
        sub = self.matrix[para_idx]
        sims = sub @ sub.T
        np.fill_diagonal(sims, -1.0)
        if threshold is None:
            off_diagonal = sims[~np.eye(len(para_idx), dtype=bool)]
            threshold = max(SEMANTIC_FLOOR, float(np.quantile(off_diagonal, SEMANTIC_QUANTILE)))
        pairs = set()
        for row, i in enumerate(para_idx):
            for col in np.argsort(-sims[row])[:k]:
                if sims[row, col] >= threshold:
                    a, b = sorted((self.ids[i], self.ids[para_idx[col]]))
                    pairs.add((a, b, float(sims[row, col])))
        return sorted(pairs)

    # ----------------------------------------------------------------- access

    def encode_query(self, query: str) -> np.ndarray:
        return self.embedder.encode_query(query)

    def search(self, query_vector: np.ndarray, k: int,
               node_types: Optional[Iterable[str]] = None) -> List[Tuple[str, float]]:
        allowed = None
        if node_types is not None:
            allowed = ids_of_types((self.graph.nodes[i] for i in self.ids), node_types)
        return self.vectors.search(query_vector, k, allowed)

    def records(self) -> List[EmbeddingRecord]:
        """node_id, vector, type, text and citation metadata for every embedded node."""
        out = []
        for i, nid in enumerate(self.ids):
            node = self.graph.nodes[nid]
            out.append(EmbeddingRecord(nid, self.matrix[i], node.node_type, node.text, {
                "section_path": self.graph.path(nid),
                "page": node.page,
                "number": node.metadata.get("number"),
                "model": self.embedder.name,
                "contextual": self.contextual,
            }))
        return out
