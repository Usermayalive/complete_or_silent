"""G2: CoarseGNN features, training, PPO policy and GNNScorer (needs torch + torch_geometric)."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gnn_pruner import DocumentGraph, DocumentQA  # noqa: E402
from gnn_pruner.embeddings import l2_normalize  # noqa: E402
from gnn_pruner.gnn.features import (EDGE_TYPES, doc_tensors, feature_groups, input_dim,  # noqa: E402
                                     node_features, targets)
from gnn_pruner.gnn.model import CoarseGNN, ModelConfig  # noqa: E402
from gnn_pruner.gnn.scorer import GNNScorer, load_scorer  # noqa: E402
from gnn_pruner.gnn.train import (Example, layer_rewards, paragraph_metrics, policy_keep,  # noqa: E402
                                  rank_l3, save, scores_for, train_supervised)
from gnn_pruner.index import tokenize  # noqa: E402
from gnn_pruner.layers import four_layer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample_contract.md"
DIM = 32


class ToyEmbedder:
    """Deterministic hashed bag of words: fast, and the same vector for the same text."""

    name = "toy-hash"
    cacheable = True

    def fit(self, texts):
        return self

    def encode(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokenize(text):
                out[i, int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
        return l2_normalize(out)

    def encode_query(self, text):
        return self.encode([text])[0]


@pytest.fixture(scope="module")
def doc():
    layered = four_layer(DocumentGraph.from_file(str(SAMPLE)), "sample")
    ids = [n.id for n in layered.nodes.values() if n.layer >= 3]
    return doc_tensors(layered, ids, ToyEmbedder().encode([layered.nodes[i].text for i in ids]))


def test_coarse_graph(doc):
    layers = {doc.layered.nodes[nid].layer for nid in doc.ids}
    assert layers == {1, 2, 3}
    # A heading's vector is the normalised mean of the paragraphs below it.
    section = next(n for n in doc.layered.layer(1) if n.title == "4. Liability")
    below = [doc.row[n.id] for n in doc.layered.layer(3) if section.id in doc.layered.path(n.id)]
    mean = doc.emb[below].mean(0)
    assert np.allclose(doc.emb[doc.row[section.id]], mean / np.linalg.norm(mean), atol=1e-6)
    for t in EDGE_TYPES:  # every edge appears in both directions
        pairs = set(map(tuple, doc.edges[t].t().tolist()))
        assert pairs and all((b, a) in pairs for a, b in pairs)


def test_node_features(doc):
    q = ToyEmbedder().encode_query("cap on liability")
    x = node_features(doc, "cap on liability", q)
    assert x.shape == (len(doc.ids), input_dim(DIM))
    groups = feature_groups(DIM)
    no_query = node_features(doc, "cap on liability", q, use_query=False)
    assert torch.all(no_query[:, groups["query_match"]] == 0)
    assert torch.all(no_query[:, groups["semantic"]] == 0) and torch.all(no_query[:, groups["lexical"]] == 0)
    assert torch.equal(no_query[:, groups["embedding"]], x[:, groups["embedding"]])


def test_paragraph_metrics():
    m = paragraph_metrics(["a", "b", "c"], [{"b"}, {"z"}], k=2)
    assert m == {"recall@2": 0.5, "mrr": 0.5, "ndcg@2": pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3))),
                 "hit@1": 0.0}
    assert paragraph_metrics(["a"], []) == {}


def _example(doc, question, answer_prefix):
    target = next(n for n in doc.layered.layer(3) if n.text.startswith(answer_prefix))
    positives = {nid: 1.0 for nid in doc.layered.path(target.id)}
    return Example("q", doc.doc_id, question, ToyEmbedder().encode_query(question), positives,
                   [{target.id}])


def test_supervised_training_fits_and_is_seeded(doc):
    examples = [_example(doc, "cap on liability", "Except as set out"),
                _example(doc, "invoices thirty days", "The Customer shall pay")]
    data = ({doc.doc_id: doc}, examples)

    def run():
        model = CoarseGNN(ModelConfig(in_dim=input_dim(DIM), hidden=32, dropout=0.0))
        torch.manual_seed(0)
        model = CoarseGNN(ModelConfig(in_dim=input_dim(DIM), hidden=32, dropout=0.0))
        result = train_supervised(model, data, data, epochs=30, lr=5e-3, batch=1, patience=30,
                                  log=lambda _: None)
        return model, result

    model, result = run()
    assert result["best_val_mrr"] == 1.0
    for e in examples:
        assert rank_l3(doc, scores_for(model, doc, e.question, e.query_vector))[0] in e.gold[0]
    again, _ = run()
    for a, b in zip(model.state_dict().values(), again.state_dict().values()):
        assert torch.equal(a, b)


def test_policy_respects_hierarchy(doc):
    rng = np.random.default_rng(0)
    probs = rng.random(len(doc.ids))
    kept, decisions = policy_keep(doc, probs, rng)
    for i in np.where(kept)[0]:
        parent = doc.layered.nodes[doc.ids[i]].parent_id
        assert parent is None or kept[doc.row[parent]]
    assert [doc.layer[r] for r, _ in decisions] == sorted(doc.layer[r] for r, _ in decisions)
    everything, _ = policy_keep(doc, np.ones(len(doc.ids)))
    y = targets(doc, {doc.ids[doc.l3[0]]: 1.0}).numpy()
    rewards = layer_rewards(doc, everything, y, size_penalty=0.5)
    assert rewards[2] == pytest.approx(1.0 - 0.5)


@pytest.fixture(scope="module")
def toy_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("model")
    torch.manual_seed(0)
    model = CoarseGNN(ModelConfig(in_dim=input_dim(DIM), hidden=16))
    save(model, path, {"embedder": ToyEmbedder.name, "stage": "test"})
    return path


def test_gnn_scorer_in_the_pipeline(toy_model):
    qa = DocumentQA.from_file(str(SAMPLE), embedder=ToyEmbedder(), scorer=GNNScorer(str(toy_model)))
    result = qa.ask("What is the cap on liability?")
    assert result.evidence.items
    scored = result.scored
    assert scored == sorted(scored, key=lambda s: -s.score)
    assert all(0.0 <= s.score <= 1.0 for s in scored)
    explained = [s for s in scored if "neighbours" in s.contributions]
    assert explained and set(explained[0].contributions) >= {"embedding", "query_match", "lexical", "neighbours"}
    assert all(set(s.features) >= {"semantic", "lexical"} for s in scored)  # pruning still has its features


def test_gnn_scorer_rejects_another_embedder(toy_model):
    qa = DocumentQA.from_file(str(SAMPLE), scorer=GNNScorer(str(toy_model)))  # TF-IDF index
    with pytest.raises(ValueError, match="toy-hash"):
        qa.ask("What is the cap on liability?")


def test_load_scorer(toy_model):
    assert load_scorer(None) is None and load_scorer("linear") is None
    assert isinstance(load_scorer(f"gnn:{toy_model}"), GNNScorer)
    with pytest.raises(ValueError):
        load_scorer("bogus")
