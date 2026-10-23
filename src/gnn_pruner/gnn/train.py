"""Train CoarseGNN on Silver Paths: supervised pre-training, then PPO (decision D5).

**Stage 1, supervised.** Per training question, the targets are the graded Silver
Positive Set on the coarse graph. The loss is a listwise term (softmax over L3
paragraphs vs the silver grade distribution) plus graded BCE on all coarse nodes.
Early stopping uses the validation documents' known answers (paragraph MRR); these
are generated questions, never the dev or test sets.

**Stage 2, PPO.** The same network, as a top-down keep/drop policy
``p = sigmoid(logit)``:
1. decide on L1 sections
2. then on the L2 sub-sections of kept sections
3. then on the L3 paragraphs of kept parents

Reward per layer ``l`` (dense intermediate rewards, spec §2.2):
``r_l = silver grade kept / silver grade available - size_penalty * share of layer kept``.
A decision on layer ``l`` earns ``sum_{l' >= l} w_l' r_l'``. Updates use the clipped
PPO objective with an entropy bonus and a small supervised anchor, which stops the
ranking from collapsing.

    python -m gnn_pruner.gnn.train --corpus data/corpus_v1 --out models/coarse_gnn_v1
    python -m gnn_pruner.gnn.train --corpus data/corpus_v1 --out models/ablate_no_edges --ablation no_edges --stage supervised
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .features import DocTensors, doc_tensors, input_dim, node_features, targets
from .model import CoarseGNN, ModelConfig

ABLATIONS = {"none": {}, "no_edges": {"use_edges": False}, "no_query": {"use_query": False},
             "untyped": {"typed": False}, "skip": {"skip": True}}


@dataclass
class Example:
    question_id: str
    doc_id: str
    question: str
    query_vector: np.ndarray
    positives: Dict[str, float]
    gold: List[Set[str]]                 # per answer snippet: acceptable L3 paragraphs
    tags: Dict = field(default_factory=dict)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_split(corpus: str, split: str, embedder, labels: str = "silver"
               ) -> Tuple[Dict[str, DocTensors], List[Example]]:
    """``labels="silver"``: Silver Positive Sets (the spec). ``labels="gold"``: the
    generated questions' known answers (path to each answer paragraph, grade 1); an
    upper-bound reading of what silver noise costs, still on training documents only."""
    from ..parquet_store import read_document, read_embeddings
    from ..silver import gold_nodes

    root = Path(corpus)
    records = [json.loads(l) for l in (root / f"silver_{split}.jsonl").read_text().splitlines() if l]
    docs: Dict[str, DocTensors] = {}
    examples = []
    for r in records:
        if r["doc_id"] not in docs:
            shard = root / "parquet" / r["doc_id"]
            _, ids, matrix = read_embeddings(str(shard))
            docs[r["doc_id"]] = doc_tensors(read_document(str(shard)), ids, matrix)
        doc = docs[r["doc_id"]]
        gold = [paras & set(doc.row) for _, paras in gold_nodes(doc.layered, r["gold_evidence"])]
        positives = r["positives"]
        if labels == "gold":
            positives = {nid: 1.0 for g in gold for p in g for nid in doc.layered.path(p)}
        examples.append(Example(r["question_id"], r["doc_id"], r["question"],
                                np.asarray(embedder.encode_query(r["question"]), dtype=np.float32),
                                positives, [g for g in gold if g], r.get("tags", {})))
    return docs, examples


# ------------------------------------------------------------------ metrics

def rank_l3(doc: DocTensors, scores: np.ndarray) -> List[str]:
    order = doc.l3[np.argsort(-scores[doc.l3], kind="stable")]
    return [doc.ids[i] for i in order]


def paragraph_metrics(ranked: Sequence[str], gold: Sequence[Set[str]], k: int = 10) -> Dict[str, float]:
    """Recall@k, MRR, nDCG@k and hit@1 at paragraph level. Each answer snippet counts
    once, at its first acceptable paragraph."""
    if not gold:
        return {}
    position = {nid: r for r, nid in enumerate(ranked)}
    ranks = sorted(min((position[p] for p in g if p in position), default=10 ** 9) for g in gold)
    first = ranks[0]
    dcg = sum(1 / np.log2(r + 2) for r in ranks if r < k)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return {f"recall@{k}": sum(r < k for r in ranks) / len(gold),
            "mrr": 1 / (first + 1) if first < 10 ** 9 else 0.0,
            f"ndcg@{k}": dcg / ideal, "hit@1": float(first == 0)}


def mean_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    rows = [r for r in rows if r]
    keys = rows[0].keys() if rows else []
    out = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
    out["n"] = len(rows)
    return out


@torch.no_grad()
def scores_for(model: CoarseGNN, doc: DocTensors, question: str, qvec: np.ndarray) -> np.ndarray:
    model.eval()
    x = node_features(doc, question, qvec, model.config.use_query)
    return model(x, doc.edges).numpy()


def evaluate_model(model: CoarseGNN, docs: Dict[str, DocTensors], examples: Sequence[Example]) -> Dict:
    rows = [paragraph_metrics(rank_l3(docs[e.doc_id], scores_for(model, docs[e.doc_id], e.question,
                                                                e.query_vector)), e.gold)
            for e in examples]
    return mean_metrics(rows)


# --------------------------------------------------------------- supervised

def supervised_loss(logits: torch.Tensor, y: torch.Tensor, doc: DocTensors,
                    pos_weight: float = 5.0, tau: Optional[float] = None) -> torch.Tensor:
    """Graded BCE on all coarse nodes plus a listwise term over L3. The listwise target is
    proportional to the silver grade, or with ``tau`` to ``exp(grade / tau)`` over the
    positives. Grades fall only from 1.0 to 0.4 across the 10 silver leaves, so a plain
    target spreads almost evenly; ``tau`` concentrates it on the best-ranked ones."""
    l3 = torch.from_numpy(doc.l3)
    t = y[l3]
    loss = F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=torch.tensor(pos_weight)) * 0.5
    if t.sum() > 0:
        if tau:
            p = torch.where(t > 0, torch.exp(t / tau), torch.zeros_like(t))
            p = p / p.sum()
        else:
            p = t / t.sum()
        loss = loss - (p * F.log_softmax(logits[l3], dim=0)).sum()
    return loss


def train_supervised(model: CoarseGNN, train: Tuple[Dict, List[Example]], val: Tuple[Dict, List[Example]],
                     epochs: int = 20, lr: float = 1e-3, batch: int = 8, patience: int = 4, seed: int = 0,
                     tau: Optional[float] = None, log: Callable[[str], None] = print) -> Dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    docs, examples = train
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, history, stale = -1.0, None, [], 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        order = list(examples)
        rng.shuffle(order)
        total = 0.0
        optimiser.zero_grad()
        for i, e in enumerate(order, 1):
            doc = docs[e.doc_id]
            x = node_features(doc, e.question, e.query_vector, model.config.use_query)
            loss = supervised_loss(model(x, doc.edges), targets(doc, e.positives), doc, tau=tau) / batch
            loss.backward()
            total += loss.item() * batch
            if i % batch == 0 or i == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                optimiser.zero_grad()
        metrics = evaluate_model(model, *val)
        history.append({"epoch": epoch, "loss": round(total / len(order), 4), **metrics,
                        "seconds": round(time.perf_counter() - started, 1)})
        log(f"epoch {epoch:>2} loss {history[-1]['loss']:.4f} val {metrics}")
        if metrics["mrr"] > best:
            best, best_state, stale = metrics["mrr"], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return {"history": history, "best_val_mrr": best}


# ---------------------------------------------------------------------- PPO

@dataclass
class PPOConfig:
    iterations: int = 30
    batch: int = 32
    update_epochs: int = 3
    clip: float = 0.2
    lr: float = 3e-4
    size_penalty: float = 1.0
    layer_weights: Tuple[float, float, float] = (0.25, 0.25, 1.0)
    entropy: float = 0.01
    anchor: float = 0.1


def parent_rows(doc: DocTensors) -> np.ndarray:
    out = np.full(len(doc.ids), -1, dtype=np.int64)
    for i, nid in enumerate(doc.ids):
        parent = doc.layered.nodes[nid].parent_id
        if parent is not None and parent in doc.row:
            out[i] = doc.row[parent]
    return out


def policy_keep(doc: DocTensors, probs: np.ndarray, rng: Optional[np.random.Generator] = None
                ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Top-down decisions. ``rng`` samples; without it, keep where p > 0.5. Returns the
    kept mask and the (row, action) decisions taken, in layer order."""
    parents = doc.__dict__.setdefault("_parents", parent_rows(doc))
    kept = np.zeros(len(doc.ids), dtype=bool)
    decisions = []
    for layer in (1, 2, 3):
        for i in np.where(doc.layer == layer)[0]:
            p = parents[i]
            if p >= 0 and not kept[p]:
                continue
            action = int(rng.random() < probs[i]) if rng is not None else int(probs[i] > 0.5)
            kept[i] = bool(action)
            decisions.append((int(i), action))
    return kept, decisions


def layer_rewards(doc: DocTensors, kept: np.ndarray, y: np.ndarray, size_penalty: float) -> np.ndarray:
    rewards = np.zeros(3)
    for layer in (1, 2, 3):
        rows = doc.layer == layer
        if not rows.any():
            continue
        available = y[rows].sum()
        gained = y[rows & kept].sum() / available if available > 0 else 0.0
        rewards[layer - 1] = gained - size_penalty * kept[rows].mean()
    return rewards


def train_ppo(model: CoarseGNN, train: Tuple[Dict, List[Example]], val: Tuple[Dict, List[Example]],
              config: PPOConfig = PPOConfig(), seed: int = 0, log: Callable[[str], None] = print) -> Dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    docs, examples = train
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    w = np.array(config.layer_weights)
    history, best, best_state = [], -1e9, copy.deepcopy(model.state_dict())
    for it in range(1, config.iterations + 1):
        started = time.perf_counter()
        batch = [examples[i] for i in rng.choice(len(examples), size=min(config.batch, len(examples)),
                                                 replace=False)]
        model.eval()  # no dropout: old and new log-probs must come from the same function
        rollouts = []
        with torch.no_grad():
            for e in batch:
                doc = docs[e.doc_id]
                x = node_features(doc, e.question, e.query_vector, model.config.use_query)
                logits = model(x, doc.edges)
                kept, decisions = policy_keep(doc, torch.sigmoid(logits).numpy(), rng)
                y = targets(doc, e.positives)
                r = layer_rewards(doc, kept, y.numpy(), config.size_penalty)
                returns = np.array([(w * r)[l - 1:].sum() for l in (1, 2, 3)])
                rows = torch.tensor([d[0] for d in decisions], dtype=torch.long)
                actions = torch.tensor([d[1] for d in decisions], dtype=torch.float32)
                old = -F.binary_cross_entropy_with_logits(logits[rows], actions, reduction="none")
                rollouts.append((e, doc, x, y, rows, actions, old, returns[doc.layer[rows.numpy()] - 1]))
        all_returns = np.concatenate([ro[7] for ro in rollouts])
        mean, std = all_returns.mean(), all_returns.std() + 1e-8
        for _ in range(config.update_epochs):
            optimiser.zero_grad()
            total = 0.0
            for e, doc, x, y, rows, actions, old, ret in rollouts:
                logits = model(x, doc.edges)
                advantage = torch.from_numpy(((ret - mean) / std).astype(np.float32))
                new = -F.binary_cross_entropy_with_logits(logits[rows], actions, reduction="none")
                ratio = torch.exp(new - old)
                pg = -torch.min(ratio * advantage, torch.clamp(ratio, 1 - config.clip, 1 + config.clip)
                                * advantage).mean()
                p = torch.sigmoid(logits[rows])
                entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).mean()
                loss = pg - config.entropy * entropy + config.anchor * supervised_loss(logits, y, doc)
                (loss / len(rollouts)).backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        val_policy = evaluate_policy(model, *val, size_penalty=config.size_penalty)
        val_rank = evaluate_model(model, *val)
        history.append({"iteration": it, "train_return": round(float(all_returns.mean()), 4),
                        **{f"val_{k}": v for k, v in val_policy.items()}, "val_mrr": val_rank["mrr"],
                        "seconds": round(time.perf_counter() - started, 1)})
        log(f"ppo {it:>3} return {all_returns.mean():.3f} val policy {val_policy} mrr {val_rank['mrr']}")
        if val_policy["objective"] > best:
            best, best_state = val_policy["objective"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {"history": history, "best_val_objective": best}


@torch.no_grad()
def evaluate_policy(model: CoarseGNN, docs: Dict[str, DocTensors], examples: Sequence[Example],
                    size_penalty: float = 1.0) -> Dict[str, float]:
    """Deterministic top-down policy (keep where p > 0.5): share of L3 paragraphs kept,
    share of answer snippets whose paragraph is kept, and recall - penalty * share."""
    kept_share, recall = [], []
    for e in examples:
        doc = docs[e.doc_id]
        probs = 1 / (1 + np.exp(-scores_for(model, doc, e.question, e.query_vector)))
        kept, _ = policy_keep(doc, probs)
        kept_ids = {doc.ids[i] for i in np.where(kept)[0]}
        kept_share.append(kept[doc.l3].mean())
        if e.gold:
            recall.append(np.mean([bool(g & kept_ids) for g in e.gold]))
    out = {"kept_l3_share": round(float(np.mean(kept_share)), 4),
           "gold_recall_kept": round(float(np.mean(recall)), 4)}
    out["objective"] = round(out["gold_recall_kept"] - size_penalty * out["kept_l3_share"], 4)
    return out


# ---------------------------------------------------------------- artefacts

def save(model: CoarseGNN, path: Path, meta: Dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "config.json").write_text(json.dumps({"model": model.config.to_dict(), **meta}, indent=2) + "\n")


def load(path: str) -> Tuple[CoarseGNN, Dict]:
    path = Path(path)
    meta = json.loads((path / "config.json").read_text())
    model = CoarseGNN(ModelConfig(**meta["model"]))
    model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
    model.eval()
    return model, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="data/corpus_v1")
    parser.add_argument("--out", default="models/coarse_gnn_v1")
    parser.add_argument("--stage", choices=["supervised", "ppo", "both"], default="both")
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="none")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--labels", choices=["silver", "gold"], default="silver",
                        help="gold = generated corpus answers (upper-bound reading, not the spec)")
    parser.add_argument("--tau", type=float, default=None, help="listwise target sharpness (None: proportional)")
    parser.add_argument("--ppo-iterations", type=int, default=30)
    parser.add_argument("--size-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from ..embeddings import load_embedder

    torch.set_num_threads(max(1, torch.get_num_threads()))
    root = Path(args.corpus)
    manifest = json.loads((root / "manifest.json").read_text())
    embedder = load_embedder(manifest["embedder"])
    started = time.perf_counter()
    train = load_split(args.corpus, "train", embedder, args.labels)
    val = load_split(args.corpus, "val", embedder)
    print(f"loaded {len(train[1])} train / {len(val[1])} val questions in {time.perf_counter() - started:.1f}s")
    dim = next(iter(train[0].values())).dim
    meta = {"embedder": manifest["embedder"], "corpus": str(root), "seed": args.seed,
            "ablation": args.ablation, "labels": args.labels, "corpus_manifest": file_hash(root / "manifest.json"),
            "silver_train": file_hash(root / "silver_train.jsonl"),
            "created": time.strftime("%Y-%m-%d %H:%M")}
    out = Path(args.out)
    if args.stage in ("supervised", "both"):
        model = CoarseGNN(ModelConfig(in_dim=input_dim(dim), hidden=args.hidden, layers=args.layers,
                                      dropout=args.dropout, **ABLATIONS[args.ablation]))
        t = time.perf_counter()
        result = train_supervised(model, train, val, epochs=args.epochs, lr=args.lr, seed=args.seed,
                                  tau=args.tau)
        meta["hyper"] = {"hidden": args.hidden, "layers": args.layers, "lr": args.lr,
                         "dropout": args.dropout, "tau": args.tau}
        save(model, out / "supervised", {**meta, "stage": "supervised", **result,
                                         "train_seconds": round(time.perf_counter() - t, 1)})
        print(f"supervised: best val MRR {result['best_val_mrr']:.4f}")
    if args.stage in ("ppo", "both"):
        model, _ = load(str(out / "supervised"))
        t = time.perf_counter()
        before = evaluate_policy(model, *val, size_penalty=args.size_penalty)
        cfg = PPOConfig(iterations=args.ppo_iterations, size_penalty=args.size_penalty)
        result = train_ppo(model, train, val, cfg, seed=args.seed)
        save(model, out / "ppo", {**meta, "stage": "ppo", "ppo": cfg.__dict__, "policy_before": before,
                                  **result, "train_seconds": round(time.perf_counter() - t, 1)})
        print(f"ppo: policy before {before}, best val objective {result['best_val_objective']}")


if __name__ == "__main__":
    main()
