"""G1: four-layer graph, GVRM repair, Parquet shards, generated corpus, Silver Paths."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from gnn_pruner import DocumentGraph
from gnn_pruner.corpus import TOPICS, build_corpus, generate_contract
from gnn_pruner.layers import (HIERARCHICAL, REFERENCE, SEQUENTIAL, LayeredGraph, LayerNode, check,
                               four_layer, validate_and_repair)
from gnn_pruner.parquet_store import read_document, read_embeddings, write_document
from gnn_pruner.silver import LeafIndex, label_quality, silver_path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample_contract.md"


@pytest.fixture(scope="module")
def layered():
    return four_layer(DocumentGraph.from_file(str(SAMPLE)), "sample")


def test_four_layer_mapping(layered):
    assert check(layered) == [] and layered.repairs == []
    by_text = {n.text[:20]: n for n in layered.nodes.values() if not n.virtual}
    assert by_text["1. Definitions"].layer == 1
    assert next(n for n in layered.nodes.values() if n.title == "4.1 Cap on Liability").layer == 2
    assert by_text['"Losses" means all d'].layer == 3
    sentence = next(n for n in layered.nodes.values() if n.node_type == "sentence" and not n.virtual)
    assert sentence.layer == 4 and layered.nodes[sentence.parent_id].layer == 3
    # A paragraph with no sentences gets one virtual L4 claim with its text.
    losses = by_text['"Losses" means all d']
    claims = layered.children(losses.id)
    assert len(claims) == 1 and claims[0].virtual and claims[0].text == losses.text
    # Paths run L1 -> L4; a subsection paragraph has an L2 on its path.
    cap = next(n for n in layered.layer(3) if n.text.startswith("Except as set out"))
    assert [layered.nodes[p].layer for p in layered.path(cap.id)] == [1, 2, 3]


def test_four_layer_edges(layered):
    kinds = {e.edge_type for e in layered.edges}
    assert kinds == {HIERARCHICAL, SEQUENTIAL, REFERENCE}
    for e in layered.edges:
        a, b = layered.nodes[e.source], layered.nodes[e.target]
        if e.edge_type == SEQUENTIAL:
            assert a.layer == b.layer and a.position <= b.position
        if e.edge_type == HIERARCHICAL:
            assert b.parent_id == a.id and a.layer < b.layer
    assert {e.subtype for e in layered.edges if e.edge_type == REFERENCE} == {"cross_reference", "definition_link"}


def test_gvrm_repairs_broken_layering():
    nodes = {
        "s1": LayerNode("s1", 1, "section", "1. A", None, 0),
        "p1": LayerNode("p1", 3, "paragraph", "Text.", "s1", 1),
        "c1": LayerNode("c1", 4, "sentence", "Text.", "s1", 2),    # L4 under an L1
        "c2": LayerNode("c2", 4, "sentence", "More.", "gone", 3),  # missing parent
        "x": LayerNode("x", 2, "subsection", "1.1", "p1", 4, end=-1),  # parent on a lower layer
    }
    graph = LayeredGraph("t", "T", nodes, [])
    repairs = validate_and_repair(graph)
    graph.invalidate()
    assert len(repairs) >= 4
    assert nodes["c1"].parent_id == "p1" and nodes["c2"].parent_id == "p1"
    assert nodes["x"].parent_id == "s1" and nodes["x"].end >= nodes["x"].start
    assert check(graph) == []


def test_parquet_round_trip(layered, tmp_path):
    ids = [n.id for n in layered.nodes.values() if n.layer >= 3]
    matrix = np.random.default_rng(0).normal(size=(len(ids), 8)).astype(np.float32)
    write_document(layered, str(tmp_path), ("toy-model", ids, matrix))
    loaded = read_document(str(tmp_path / "sample"))
    assert loaded.nodes == layered.nodes and set(loaded.edges) == set(layered.edges)
    assert loaded.doc_id == "sample" and loaded.title == layered.title
    model, got_ids, got = read_embeddings(str(tmp_path / "sample"))
    assert model == "toy-model" and got_ids == ids and np.array_equal(got, matrix)


def test_generated_contract_is_deterministic_and_answerable():
    a, b = generate_contract("t1", 6, seed=3), generate_contract("t1", 6, seed=3)
    assert a.text == b.text and [q.question for q in a.questions] == [q.question for q in b.questions]
    assert a.words >= 6 * 500 * 0.9
    for q in a.questions:
        for snippet in q.gold_evidence:
            assert a.text.count(snippet) == 1, snippet   # every answer exists, exactly once
    cross = [q for q in a.questions if q.tags["cross_chapter"]]
    assert cross and all(len(q.gold_evidence) == 2 for q in cross)
    for q in a.questions:
        if q.tags["paraphrase"]:  # paraphrases avoid the topic's own words
            topic = next(t for t in TOPICS if t.lower() in " ".join(q.gold_evidence).lower())
            assert topic.lower() not in q.question.lower()


def test_generated_graph_is_a_valid_four_layer_dag(tmp_path):
    doc = generate_contract("t2", 4, seed=5)
    path = tmp_path / "t2.md"
    path.write_text(doc.text)
    layered = four_layer(DocumentGraph.from_file(str(path)), "t2")
    assert check(layered) == []
    stats = layered.stats()
    assert all(stats[f"L{n}"] > 0 for n in (1, 2, 3, 4)) and stats["reference"] > 0


def test_no_test_set_leakage():
    """Generated documents share no sentence with any test or dev document, and no
    generated question repeats an eval question."""
    texts = [generate_contract(f"leak{i}", 8, seed=100 + i).text for i in range(3)]
    corpus = "\n".join(texts)
    for doc in list((ROOT / "examples").glob("*.md")) + list((ROOT / "examples" / "scale").glob("*.md")):
        for sentence in re.split(r"(?<=[.;:])\s+", doc.read_text()):
            sentence = sentence.strip()
            if len(sentence.split()) >= 6 and not sentence.startswith("#"):
                assert sentence not in corpus, (doc.name, sentence)
    # Eval sets over real documents only: the G6 adversarial suite is built from the
    # generated validation split on purpose (its documents live in data/corpus_v1).
    eval_questions = {q["question"].lower() for f in (ROOT / "eval").glob("*.jsonl")
                      for q in map(json.loads, filter(str.strip, f.read_text().splitlines()))
                      if "data/corpus" not in q.get("document", "")}
    generated = {q.question.lower() for i in range(3)
                 for q in generate_contract(f"leak{i}", 8, seed=100 + i).questions}
    assert not eval_questions & generated


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    manifest = build_corpus(str(out), docs=4, seed=11, min_pages=2, max_pages=6, val_fraction=0.25,
                            embedder_spec="tfidf", questions_per_doc=10)
    return out, manifest


def test_build_corpus_splits_by_document(corpus):
    out, manifest = corpus
    splits = {d["doc_id"]: d["split"] for d in manifest["docs"]}
    assert set(splits.values()) == {"train", "val"}
    for split in ("train", "val"):
        for line in (out / f"questions_{split}.jsonl").read_text().splitlines():
            q = json.loads(line)
            doc_id = Path(q["document"]).stem
            assert splits[doc_id] == split and (out / q["document"]).exists()
            assert (out / "parquet" / doc_id / "embeddings.parquet").exists()


def test_silver_paths(corpus):
    out, manifest = corpus
    from gnn_pruner.embeddings import load_embedder

    q = json.loads((out / "questions_train.jsonl").read_text().splitlines()[0])
    doc_id = Path(q["document"]).stem
    layered = read_document(str(out / "parquet" / doc_id))
    _, ids, matrix = read_embeddings(str(out / "parquet" / doc_id))
    index = LeafIndex.from_embeddings(layered, ids, matrix)
    embedder = load_embedder("tfidf").fit([layered.nodes[i].text for i in ids])
    vector = embedder.encode_query(q["question"])
    for mode in ("bm25", "fallback", "hybrid"):
        sp = silver_path(index, q["id"], q["question"], vector, mode)
        leaves = [leaf["node_id"] for leaf in sp.leaves]
        assert 0 < len(leaves) <= 10 and all(layered.nodes[n].layer == 4 for n in leaves)
        # Silver Positive Set = leaves plus every ancestor up to Layer 1.
        for leaf in leaves:
            assert all(a in sp.positives for a in layered.path(leaf))
        assert any(layered.nodes[n].layer == 1 for n in sp.positives)
        assert all(0.4 <= leaf["grade"] <= 1.0 for leaf in sp.leaves)
        assert not set(sp.negatives) & set(sp.positives)
        assert sp.used_dense == (mode == "hybrid" or (mode == "fallback" and sp.coverage < 0.5))
    quality = label_quality(silver_path(index, q["id"], q["question"], vector, "hybrid"),
                            layered, q["gold_evidence"])
    assert set(quality) >= {"leaf_recall", "path_recall", "precision", "mrr"}


def test_low_coverage_questions_follow_dense(corpus):
    out, _ = corpus
    doc_id = json.loads((out / "manifest.json").read_text())["docs"][0]["doc_id"]
    layered = read_document(str(out / "parquet" / doc_id))
    _, ids, matrix = read_embeddings(str(out / "parquet" / doc_id))
    index = LeafIndex.from_embeddings(layered, ids, matrix)
    vector = matrix[ids.index(index.ids[3])]  # a query vector equal to leaf 3's
    sp = silver_path(index, "q", "zebra quokka marmalade", vector, "hybrid")
    assert sp.coverage == 0.0 and sp.leaves[0]["node_id"] in {
        nid for nid in index.ids if np.allclose(index.matrix[index.ids.index(nid)], vector)}


def test_docling_parser_matches_native_structure():
    pytest.importorskip("docling")
    docx = ROOT / "examples" / "sample_contract.docx"
    native = DocumentGraph.from_file(str(docx))
    docling = DocumentGraph.from_file(str(docx), parser="docling")
    titles = lambda g: [n.title for n in g.nodes.values() if n.is_section]
    assert titles(docling) == titles(native)
