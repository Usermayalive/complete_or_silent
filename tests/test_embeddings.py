import os
import re
import zlib
from pathlib import Path

import numpy as np
import pytest

from gnn_pruner import DocumentGraph, DocumentQA
from gnn_pruner.embeddings import CachedEmbedder, TfidfEmbedder, l2_normalize, load_embedder
from gnn_pruner.index import SEMANTIC_MODEL_KEY, GraphIndex
from gnn_pruner.models import SEMANTIC_SIMILARITY
from gnn_pruner.store import GraphStore
from gnn_pruner.vector_index import FaissVectorIndex, NumpyVectorIndex, make_vector_index

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SAAS = EXAMPLES / "saas_agreement.md"
MSA = EXAMPLES / "sample_contract.md"


class HashingEmbedder:
    """Deterministic, corpus-independent bag-of-words embedder for tests."""

    cacheable = True

    def __init__(self, dim=256, name="hashing"):
        self.dim, self.name, self.calls = dim, name, 0

    def fit(self, texts):
        return self

    def encode(self, texts):
        self.calls += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z]+", text.lower()):
                out[row, zlib.crc32(token.encode()) % self.dim] += 1
        return l2_normalize(out)

    def encode_query(self, text):
        return self.encode([text])[0]


# ---------------------------------------------------------------- embedders

def test_load_embedder_specs():
    assert load_embedder("tfidf").name == "tfidf"
    with pytest.raises(ValueError):
        load_embedder("word2vec")


def test_vectors_are_normalised():
    index = GraphIndex(DocumentGraph.from_file(str(SAAS)), HashingEmbedder())
    assert np.allclose(np.linalg.norm(index.matrix, axis=1), 1.0, atol=1e-5)


def test_tfidf_cannot_be_cached():
    with pytest.raises(ValueError, match="corpus"):
        CachedEmbedder(TfidfEmbedder(), GraphStore())


def test_shared_tfidf_instance_does_not_corrupt_other_documents():
    shared = TfidfEmbedder()
    first = GraphIndex(DocumentGraph.from_file(str(SAAS)), shared)
    before = first.search(first.encode_query("enterprise plan monthly fee"), 3)
    GraphIndex(DocumentGraph.from_file(str(MSA)), shared)  # would refit a shared vectorizer
    assert first.search(first.encode_query("enterprise plan monthly fee"), 3) == before


def test_cache_embeds_each_text_once_and_persists(tmp_path):
    db = str(tmp_path / "store.db")
    model = HashingEmbedder()
    cached = CachedEmbedder(model, GraphStore(db))
    GraphIndex(DocumentGraph.from_file(str(SAAS)), cached)
    first_calls = model.calls
    assert first_calls > 0

    reopened = CachedEmbedder(model, GraphStore(db))  # new process, same database
    index = GraphIndex(DocumentGraph.from_file(str(SAAS)), reopened)
    assert model.calls == first_calls and reopened.computed == 0
    direct = GraphIndex(DocumentGraph.from_file(str(SAAS)), HashingEmbedder())
    assert np.allclose(index.matrix, direct.matrix)


# -------------------------------------------------------------------- index

def test_contextual_embedding_text_includes_section_path():
    graph = DocumentGraph.from_file(str(SAAS))
    para = next(n for n in graph.content_nodes(("paragraph",)) if n.text.startswith("The Provider must give"))
    assert GraphIndex(graph, HashingEmbedder(), contextual=True).embedding_text(para.id) == \
        f"9. Term and Renewal > 9.2 Price Changes. {para.text}"
    assert GraphIndex(graph, HashingEmbedder()).embedding_text(para.id) == para.text  # default off


def test_records_carry_vector_type_text_and_metadata():
    graph = DocumentGraph.from_file(str(SAAS))
    index = GraphIndex(graph, HashingEmbedder())
    records = {r.node_id: r for r in index.records()}
    clause = next(n for n in graph.content_nodes(("clause",)) if n.metadata["number"] == "2(c)")
    record = records[clause.id]
    assert record.node_type == "clause" and record.text == clause.text
    assert record.metadata["section_path"] == "2. Subscription"
    assert record.metadata["number"] == "2(c)" and record.metadata["model"] == "hashing"
    assert record.vector.shape == (256,)


def test_semantic_edges_follow_the_embedder():
    graph = DocumentGraph.from_file(str(SAAS))
    GraphIndex(graph, HashingEmbedder(name="model-a"))
    edges_a = [e for e in graph.edges if e.edge_type == SEMANTIC_SIMILARITY]
    assert graph.root.metadata[SEMANTIC_MODEL_KEY] == "model-a"

    GraphIndex(graph, HashingEmbedder(name="model-a"))  # same model: kept as is
    assert [e for e in graph.edges if e.edge_type == SEMANTIC_SIMILARITY] == edges_a

    GraphIndex(graph, HashingEmbedder(dim=64, name="model-b"))  # new model: rebuilt
    assert graph.root.metadata[SEMANTIC_MODEL_KEY] == "model-b"
    edges_b = [e for e in graph.edges if e.edge_type == SEMANTIC_SIMILARITY]
    assert edges_b and edges_b != edges_a
    assert all(e.edge_type != SEMANTIC_SIMILARITY or e in edges_b for e in graph.edges)
    assert graph.validate() == []


def test_search_filters_node_types():
    index = GraphIndex(DocumentGraph.from_file(str(SAAS)), HashingEmbedder())
    hits = index.search(index.encode_query("build a competing product"), 3, ["clause"])
    assert hits and all(index.graph.nodes[nid].node_type == "clause" for nid, _ in hits)


# ------------------------------------------------------------- vector index

def test_numpy_index_ranks_and_filters():
    vectors = l2_normalize(np.eye(4, dtype=np.float32) + 0.1)
    index = NumpyVectorIndex(["a", "b", "c", "d"], vectors)
    assert index.search(vectors[2], 1) == [("c", pytest.approx(1.0))]
    assert [i for i, _ in index.search(vectors[2], 2, allowed={"a", "b"})] in (["a", "b"], ["b", "a"])


@pytest.mark.parametrize("n", [500, 12_000])  # exact, then HNSW
def test_faiss_matches_exact_search(n):
    pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    vectors = l2_normalize(rng.normal(size=(n, 32)))
    ids = [f"n{i}" for i in range(n)]
    exact, faiss_index = NumpyVectorIndex(ids, vectors), FaissVectorIndex(ids, vectors)
    assert faiss_index.approximate == (n >= 10_000)
    overlap = []
    for q in vectors[:20] + 0.05:
        want = {i for i, _ in exact.search(q, 10)}
        overlap.append(len(want & {i for i, _ in faiss_index.search(q, 10)}) / 10)
    assert np.mean(overlap) >= (1.0 if n < 10_000 else 0.9)
    allowed = set(ids[:50])
    assert all(i in allowed for i, _ in faiss_index.search(vectors[0], 5, allowed))


def test_make_vector_index_kinds():
    vectors = l2_normalize(np.ones((3, 4)))
    assert make_vector_index(["a", "b", "c"], vectors, "numpy").kind == "numpy"
    assert make_vector_index(["a", "b", "c"], vectors, "auto").kind in ("numpy", "faiss")


# -------------------------------------------------------------------- store

def test_store_persists_embeddings_and_searches_across_documents(tmp_path):
    store = GraphStore(str(tmp_path / "store.db"))
    model = HashingEmbedder()
    ids = {}
    for name, path in (("saas", SAAS), ("msa", MSA)):
        graph = DocumentGraph.from_file(str(path))
        ids[name] = store.put(graph, doc_id=name)
        index = GraphIndex(graph, model)
        assert store.put_embeddings(name, index) == len(index.ids)

    saved = store.get_embeddings("saas", "hashing")
    graph = store.get("saas")
    assert np.allclose(saved[graph.content_nodes(("table",))[0].id],
                       GraphIndex(graph, model).matrix[GraphIndex(graph, model).position[
                           graph.content_nodes(("table",))[0].id]])

    query = model.encode_query("indemnify the Customer against Losses from Misconduct")
    hits = store.search(query, "hashing", k=3)
    assert hits[0].doc_id == "msa" and "indemnify" in hits[0].node.text
    only_saas = store.search(query, "hashing", k=3, doc_ids=["saas"])
    assert {h.doc_id for h in only_saas} == {"saas"}
    clauses = store.search(query, "hashing", k=5, node_types=["clause"])
    assert clauses and all(h.node.node_type == "clause" for h in clauses)

    store.delete("msa")
    assert {h.doc_id for h in store.search(query, "hashing", k=50)} == {"saas"}
    with pytest.raises(KeyError):
        store.put_embeddings("missing", index)


def test_pipeline_runs_with_custom_embedder_and_options():
    qa = DocumentQA.from_file(str(SAAS), embedder=HashingEmbedder(), contextual=False,
                              vector_index="numpy", semantic_scaling="minmax")
    result = qa.ask("How much does the Enterprise plan cost?")
    assert any("4,000 USD" in i.text for i in result.evidence.items)


# ------------------------------------------------- real model (if available)

def _cached_model(name):
    pytest.importorskip("sentence_transformers")
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if not (cache / f"models--{name.replace('/', '--')}").exists():
        pytest.skip(f"{name} not downloaded")
    os.environ["HF_HUB_OFFLINE"] = "1"
    return load_embedder(f"st:{name}")


def test_sentence_transformer_semantics():
    embedder = _cached_model("sentence-transformers/all-MiniLM-L6-v2")
    index = GraphIndex(DocumentGraph.from_file(str(SAAS)), embedder)
    assert np.allclose(np.linalg.norm(index.matrix, axis=1), 1.0, atol=1e-4)
    # Paraphrase with no shared content words: dense embeddings should still find it.
    top = index.search(index.encode_query("Can they charge us more next year?"), 3)
    texts = " ".join(index.graph.nodes[nid].text for nid, _ in top)
    assert "increase" in texts
