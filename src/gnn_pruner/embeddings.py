"""Phase 3: semantic representations of nodes.

Embedders share one interface. ``fit`` is a no-op for pretrained models;
``encode`` embeds documents and ``encode_query`` embeds questions, so models
that use asymmetric prompts (e5/bge: "query: " / "passage: ") work correctly.
All vectors are L2-normalised, so a dot product is the cosine similarity.

- ``TfidfEmbedder``: the zero-download default. It is fitted on the document, so
  its vectors are not reusable across documents and are never cached.
- ``SentenceTransformerEmbedder``: pretrained dense model (``pip install
  sentence-transformers``). Stateless, so vectors are cached by text hash.
- ``CachedEmbedder``: wraps a stateless embedder with a persistent vector cache.

``load_embedder("tfidf" | "st:<model name>" | "auto")`` builds one from a spec
string; "auto" picks the default dense model when sentence-transformers is
installed and falls back to TF-IDF otherwise.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    name: str          # stable identifier: cache key and provenance of stored vectors
    cacheable: bool    # True if a text's vector does not depend on the corpus

    def fit(self, texts: Sequence[str]) -> "Embedder": ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed document texts: an (n, d) array of L2-normalised vectors."""
        ...

    def encode_query(self, text: str) -> np.ndarray:
        """Embed a question: a (d,) L2-normalised vector."""
        ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class TfidfEmbedder:
    name = "tfidf"
    cacheable = False

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        from .textutil import stemmed_words

        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), stop_words="english",
            tokenizer=stemmed_words, token_pattern=None,
        )

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        self._vectorizer.fit(list(texts) or [""])
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return l2_normalize(self._vectorizer.transform(list(texts)).toarray())

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class SentenceTransformerEmbedder:
    cacheable = True

    def __init__(self, model_name: str = DEFAULT_ST_MODEL, query_prefix: str = "",
                 document_prefix: str = "", batch_size: int = 32,
                 device: Optional[str] = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        prefixes = f"|q={query_prefix}|d={document_prefix}" if query_prefix or document_prefix else ""
        self.name = f"st:{model_name}{prefixes}"

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        return self

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(texts, batch_size=self.batch_size,
                                     convert_to_numpy=True, show_progress_bar=False)
        return l2_normalize(vectors)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.document_prefix + t for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]


class VectorCache(Protocol):
    def get_vectors(self, model: str, keys: Sequence[str]) -> Dict[str, np.ndarray]: ...

    def put_vectors(self, model: str, items: Dict[str, np.ndarray]) -> None: ...


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedEmbedder:
    """Only texts not already in ``cache`` reach the wrapped model."""

    def __init__(self, embedder: Embedder, cache: VectorCache) -> None:
        if not embedder.cacheable:
            raise ValueError(f"{embedder.name} vectors depend on the corpus; caching them is unsafe")
        self.inner = embedder
        self.cache = cache
        self.name = embedder.name
        self.cacheable = True
        self.computed = 0  # texts actually embedded, for observability

    def fit(self, texts: Sequence[str]) -> "CachedEmbedder":
        self.inner.fit(texts)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        keys = [text_key(t) for t in texts]
        found = self.cache.get_vectors(self.name, keys)
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in found]
        if missing:
            vectors = self.inner.encode([t for _, t in missing])
            new = {k: v for (k, _), v in zip(missing, vectors)}
            self.cache.put_vectors(self.name, new)
            found.update(new)
            self.computed += len(missing)
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([found[k] for k in keys]).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.inner.encode_query(text)


def load_embedder(spec: str = "tfidf", **kwargs) -> Embedder:
    """``"tfidf"``, ``"st"`` (default model), ``"st:<huggingface model name>"`` or ``"auto"``."""
    if spec == "auto":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return TfidfEmbedder()
        return SentenceTransformerEmbedder(DEFAULT_ST_MODEL, **kwargs)
    if spec == "tfidf":
        return TfidfEmbedder()
    if spec == "st" or spec.startswith("st:"):
        model = spec.partition(":")[2] or DEFAULT_ST_MODEL
        return SentenceTransformerEmbedder(model, **kwargs)
    raise ValueError(f"unknown embedder spec {spec!r}; use 'tfidf' or 'st:<model>'")
