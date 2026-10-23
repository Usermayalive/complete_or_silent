"""Vector search over node embeddings.

``NumpyVectorIndex`` is exact and dependency-free; it is right for a single
document (hundreds to thousands of vectors). ``FaissVectorIndex`` (``pip install
faiss-cpu``) uses exact inner-product search below ``hnsw_threshold`` vectors and
an approximate HNSW graph above it, for corpus-scale search.

Both take an optional ``allowed`` id filter (node types, documents) applied
before results are cut to ``k``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Hit = Tuple[str, float]


class NumpyVectorIndex:
    kind = "numpy"

    def __init__(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        self.ids = list(ids)
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: np.ndarray, k: int, allowed: Optional[Set[str]] = None) -> List[Hit]:
        if not self.ids:
            return []
        scores = self.vectors @ np.asarray(query, dtype=np.float32)
        order = np.argsort(-scores)
        hits = []
        for i in order:
            if allowed is None or self.ids[i] in allowed:
                hits.append((self.ids[i], float(scores[i])))
                if len(hits) == k:
                    break
        return hits


class FaissVectorIndex:
    kind = "faiss"

    def __init__(self, ids: Sequence[str], vectors: np.ndarray, hnsw_threshold: int = 10_000,
                 hnsw_m: int = 32, ef_search: int = 128) -> None:
        import sys

        import faiss

        if sys.platform == "darwin" and "torch" in sys.modules:
            # FAISS and torch each bundle an OpenMP runtime on macOS; multithreaded
            # FAISS calls abort or segfault once torch has run. One thread is safe.
            faiss.omp_set_num_threads(1)
        self.ids = list(ids)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        dim = vectors.shape[1] if vectors.ndim == 2 and vectors.shape[0] else 1
        if len(self.ids) >= hnsw_threshold:
            self._index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efSearch = ef_search
            self.approximate = True
        else:
            self._index = faiss.IndexFlatIP(dim)
            self.approximate = False
        if len(self.ids):
            self._index.add(vectors)

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: np.ndarray, k: int, allowed: Optional[Set[str]] = None) -> List[Hit]:
        if not self.ids:
            return []
        q = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        fetch = k
        while True:
            # Over-fetch when filtering so k allowed hits survive.
            fetch = min(len(self.ids), fetch if allowed is None else max(fetch, 4 * k))
            scores, idx = self._index.search(q, fetch)
            hits = [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0])
                    if i >= 0 and (allowed is None or self.ids[i] in allowed)]
            if len(hits) >= k or fetch >= len(self.ids):
                return hits[:k]
            fetch *= 4


def make_vector_index(ids: Sequence[str], vectors: np.ndarray, kind: str = "auto"):
    """``kind``: "numpy", "faiss", or "auto" (FAISS if installed, else numpy).

    On macOS, "auto" uses numpy once torch is loaded: FAISS and torch ship separate
    OpenMP runtimes and the process aborts when both run (seen with the GNN scorer).
    Both indexes are exact below FAISS's HNSW size, so results are identical."""
    import sys

    if kind == "numpy" or (kind == "auto" and sys.platform == "darwin" and "torch" in sys.modules):
        return NumpyVectorIndex(ids, vectors)
    if kind in ("faiss", "auto"):
        try:
            return FaissVectorIndex(ids, vectors)
        except ImportError:
            if kind == "faiss":
                raise
    return NumpyVectorIndex(ids, vectors)


def ids_of_types(nodes: Iterable, types: Iterable[str]) -> Set[str]:
    wanted = set(types)
    return {n.id for n in nodes if n.node_type in wanted}
