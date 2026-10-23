"""G1: Silver Path weak supervision (spec §2.2).

For each training question:

1. BM25 ranks the Layer-4 leaves (atomic claims) of the question's document.
2. A dense retriever supplements BM25 for questions that rely on meaning rather than
   shared words. Three modes:
   - ``bm25``: BM25 only
   - ``fallback``: dense results are fused in (reciprocal rank fusion) only when BM25's
     best leaf covers less than ``coverage_threshold`` of the question's IDF-weighted
     terms
   - ``hybrid``: BM25 and dense are always fused

   Fusion is reciprocal rank fusion weighted by BM25's term coverage ``c`` of the
   question: BM25 counts ``c``, dense ``1 - c``. With equal weights, BM25's
   near-random order on paraphrases pushed the dense hits out of the top 10 (trial
   corpus: paraphrase leaf recall 0.54 fused vs 0.90 dense alone).
3. The top-10 leaves are the silver leaves. Walking up the hierarchy to Layer 1 gives
   the **Silver Positive Set**.
4. Grades: rank 1 -> 1.0 down to rank 10 -> 0.4, plus 0.2 when both retrievers have the
   leaf in their top 10 (capped at 1.0). An ancestor takes the best grade of its
   silver leaves.
5. Hard negatives: the next 20 fused leaves outside the positive set.

``label_quality`` scores a Silver Path against known answers (the generated corpus,
or the dev sets' gold snippets). That is how we know how noisy the training labels are.

    python -m gnn_pruner.silver --corpus data/corpus_v1 --modes bm25 fallback hybrid --write hybrid
    python -m gnn_pruner.silver --dev eval/calibration_v1.jsonl eval/dev_pruning_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .index import BM25, tokenize
from .layers import LayeredGraph
from .parser import normalize_ws

MODES = ("bm25", "fallback", "hybrid")
TOP_K = 10
DEPTH = 50
NEGATIVES = 20
RRF_K = 60
COVERAGE_THRESHOLD = 0.5


class LeafIndex:
    """BM25 + dense search over one document's Layer-4 leaves."""

    def __init__(self, layered: LayeredGraph, vectors: Dict[str, np.ndarray]):
        self.layered = layered
        self.ids = [n.id for n in layered.layer(4)]
        self.bm25 = BM25([tokenize(layered.nodes[i].text) for i in self.ids])
        self.matrix = np.vstack([vectors[i] for i in self.ids]).astype(np.float32) if self.ids \
            else np.zeros((0, 1), dtype=np.float32)

    @classmethod
    def from_embeddings(cls, layered: LayeredGraph, ids: Sequence[str], matrix: np.ndarray) -> "LeafIndex":
        return cls(layered, {nid: matrix[i] for i, nid in enumerate(ids)})

    def bm25_ranked(self, question: str, depth: int = DEPTH) -> List[str]:
        scores = self.bm25.scores(tokenize(question))
        order = np.argsort(-scores, kind="stable")[:depth]
        return [self.ids[i] for i in order if scores[i] > 0]

    def dense_ranked(self, query_vector: np.ndarray, depth: int = DEPTH) -> List[str]:
        sims = self.matrix @ query_vector
        return [self.ids[i] for i in np.argsort(-sims, kind="stable")[:depth]]

    def coverage(self, question: str) -> float:
        """The best leaf's share of the question's IDF-weighted terms (a term no leaf
        contains weighs the most, as in the Phase 7 confidence signal)."""
        tokens = list(dict.fromkeys(tokenize(question)))
        if not tokens or not self.ids:
            return 0.0
        bm25 = self.bm25
        weights = np.array([bm25.idf[bm25.vocab[t]] if t in bm25.vocab else bm25.max_idf for t in tokens])
        known = [(bm25.vocab[t], w) for t, w in zip(tokens, weights) if t in bm25.vocab]
        if not known:
            return 0.0
        present = (bm25.weights[:, [c for c, _ in known]] > 0).astype(np.float64)
        covered = np.asarray(present @ np.array([w for _, w in known])).ravel()
        return float(covered.max() / weights.sum())


@dataclass
class SilverPath:
    question_id: str
    doc_id: str
    question: str
    mode: str
    used_dense: bool
    coverage: float
    leaves: List[Dict]                    # node_id, grade, rank, bm25_rank, dense_rank
    positives: Dict[str, float]           # Silver Positive Set: leaves + ancestors -> grade
    negatives: List[str]                  # hard negatives (leaves)
    fused: List[str] = field(default_factory=list, repr=False)  # full fused ranking (for MRR)

    def to_dict(self) -> Dict:
        out = asdict(self)
        out.pop("fused")
        return out


def _rrf(rankings: Sequence[List[str]], weights: Sequence[float]) -> List[str]:
    scores: Dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, nid in enumerate(ranking):
            scores[nid] = scores.get(nid, 0.0) + weight / (RRF_K + rank + 1)
    return sorted(scores, key=lambda n: -scores[n])


def silver_path(index: LeafIndex, question_id: str, question: str, query_vector: Optional[np.ndarray],
                mode: str = "hybrid", k: int = TOP_K,
                coverage_threshold: float = COVERAGE_THRESHOLD) -> SilverPath:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    bm25 = index.bm25_ranked(question)
    coverage = index.coverage(question)
    use_dense = query_vector is not None and (
        mode == "hybrid" or (mode == "fallback" and coverage < coverage_threshold))
    dense = index.dense_ranked(query_vector) if query_vector is not None else []
    fused = _rrf([bm25, dense], [coverage, 1.0 - coverage]) if use_dense else bm25
    bm25_rank = {nid: r for r, nid in enumerate(bm25)}
    dense_rank = {nid: r for r, nid in enumerate(dense)}
    leaves, positives = [], {}
    for rank, nid in enumerate(fused[:k]):
        grade = 1.0 - 0.6 * rank / max(k - 1, 1)
        if bm25_rank.get(nid, k) < k and dense_rank.get(nid, k) < k:
            grade = min(1.0, grade + 0.2)
        grade = round(grade, 3)
        leaves.append({"node_id": nid, "grade": grade, "rank": rank,
                       "bm25_rank": bm25_rank.get(nid), "dense_rank": dense_rank.get(nid)})
        for node_id in index.layered.path(nid):
            positives[node_id] = max(positives.get(node_id, 0.0), grade)
    negatives = [nid for nid in fused[k:] if nid not in positives][:NEGATIVES]
    return SilverPath(question_id, index.layered.doc_id, question, mode, use_dense, round(coverage, 4),
                      leaves, positives, negatives, fused)


# ------------------------------------------------------------- label quality

def _norm(text: str) -> str:
    return normalize_ws(text).lower()


def gold_nodes(layered: LayeredGraph, snippets: Sequence[str]) -> List[Tuple[set, set]]:
    """Per snippet: (gold leaves, gold L3 paragraphs). A snippet inside one leaf is that
    leaf. A snippet spanning several leaves makes every leaf of the paragraph gold."""
    leaves = layered.layer(4)
    paragraphs = layered.layer(3)
    out = []
    for snippet in snippets:
        s = _norm(snippet)
        holders = {n.id for n in leaves if s in _norm(n.text)}
        paras = {n.id for n in paragraphs if s in _norm(n.text)}
        if not holders:
            holders = {leaf.id for p in paras for leaf in layered.leaves_of(p)}
        if not paras:
            paras = {layered.nodes[h].parent_id for h in holders}
        out.append((holders, paras))
    return out


def label_quality(path: SilverPath, layered: LayeredGraph, snippets: Sequence[str]) -> Dict[str, float]:
    gold = gold_nodes(layered, snippets)
    found = [g for g in gold if g[0]]
    if not found:
        return {}
    silver_leaves = [leaf["node_id"] for leaf in path.leaves]
    gold_leaves = set().union(*(g[0] for g in found))
    gold_paras = set().union(*(g[1] for g in found))
    in_gold_para = [nid for nid in silver_leaves if layered.nodes[nid].parent_id in gold_paras]
    first = next((r for r, nid in enumerate(path.fused, 1) if nid in gold_leaves), None)
    return {
        "leaf_recall": sum(bool(g[0] & set(silver_leaves)) for g in found) / len(found),
        "paragraph_recall": sum(bool(g[1] & set(path.positives)) for g in found) / len(found),
        "path_recall": sum(all(a in path.positives for p in g[1] for a in layered.path(p))
                           for g in found) / len(found),
        "precision": len(in_gold_para) / len(silver_leaves) if silver_leaves else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "hit_at_1": float(bool(silver_leaves) and silver_leaves[0] in gold_leaves),
    }


def summarize(rows: List[Dict]) -> Dict[str, float]:
    keys = ["leaf_recall", "paragraph_recall", "path_recall", "precision", "mrr", "hit_at_1"]
    rows = [r for r in rows if r]
    out = {k: round(sum(r[k] for r in rows) / len(rows), 3) if rows else 0.0 for k in keys}
    out["n"] = len(rows)
    return out


# ----------------------------------------------------------------- running

def _groups(tags: Dict) -> List[str]:
    groups = ["all"]
    if tags.get("cross_chapter"):
        groups.append("cross_chapter")
    elif tags.get("kind") == "definition":
        groups.append("definition")
    else:
        groups.append("paraphrase" if tags.get("paraphrase") else "lexical")
    return groups


def run_corpus(corpus: str, modes: Sequence[str], write: Optional[str] = None,
               coverage_threshold: float = COVERAGE_THRESHOLD, splits=("train", "val")) -> Dict:
    from .embeddings import load_embedder
    from .parquet_store import read_document, read_embeddings

    root = Path(corpus)
    manifest = json.loads((root / "manifest.json").read_text())
    embedder = load_embedder(manifest["embedder"]) if manifest.get("embedder") else None
    report: Dict = {}
    started = time.perf_counter()
    for split in splits:
        questions = [json.loads(l) for l in (root / f"questions_{split}.jsonl").read_text().splitlines() if l]
        by_doc: Dict[str, List[Dict]] = {}
        for q in questions:
            by_doc.setdefault(Path(q["document"]).stem, []).append(q)
        rows = {m: {} for m in modes}
        written = []
        for doc_id, qs in by_doc.items():
            shard = root / "parquet" / doc_id
            layered = read_document(str(shard))
            model, ids, matrix = read_embeddings(str(shard))
            index = LeafIndex.from_embeddings(layered, ids, matrix)
            vectors = [embedder.encode_query(q["question"]) if embedder else None for q in qs]
            for q, vec in zip(qs, vectors):
                for mode in modes:
                    sp = silver_path(index, q["id"], q["question"], vec, mode,
                                     coverage_threshold=coverage_threshold)
                    quality = label_quality(sp, layered, q["gold_evidence"])
                    for group in _groups(q.get("tags", {})):
                        rows[mode].setdefault(group, []).append(quality)
                    if mode == write:
                        written.append({**sp.to_dict(), "gold_evidence": q["gold_evidence"],
                                        "tags": q.get("tags", {}), "quality": quality})
        report[split] = {m: {g: summarize(r) for g, r in groups.items()} for m, groups in rows.items()}
        if write:
            with open(root / f"silver_{split}.jsonl", "w", encoding="utf-8") as f:
                for item in written:
                    f.write(json.dumps(item) + "\n")
    report["seconds"] = round(time.perf_counter() - started, 1)
    return report


def run_dev(paths: Sequence[str], modes: Sequence[str], embedder_spec: str,
            coverage_threshold: float = COVERAGE_THRESHOLD) -> Dict:
    """Label quality on the dev sets (their documents are built on the fly; nothing is
    written, so no silver label from a test document can reach training)."""
    from .corpus import embed_layered
    from .document_graph import DocumentGraph
    from .embeddings import load_embedder
    from .evaluation import load_cases
    from .layers import four_layer

    embedder = load_embedder(embedder_spec)
    indexes: Dict[str, Tuple[LeafIndex, LayeredGraph]] = {}
    report = {}
    for path in paths:
        rows = {m: [] for m in modes}
        for case in load_cases(path):
            if not case.answerable:
                continue
            if case.document not in indexes:
                layered = four_layer(DocumentGraph.from_file(case.document), Path(case.document).stem)
                _, ids, matrix = embed_layered(layered, embedder)
                indexes[case.document] = (LeafIndex.from_embeddings(layered, ids, matrix), layered)
            index, layered = indexes[case.document]
            vec = embedder.encode_query(case.question)
            for mode in modes:
                sp = silver_path(index, case.id, case.question, vec, mode,
                                 coverage_threshold=coverage_threshold)
                rows[mode].append(label_quality(sp, layered, case.gold_evidence))
        report[Path(path).name] = {m: summarize(r) for m, r in rows.items()}
    return report


def _print(report: Dict) -> None:
    keys = ["leaf_recall", "paragraph_recall", "path_recall", "precision", "mrr", "hit_at_1", "n"]
    for name, by_mode in report.items():
        if not isinstance(by_mode, dict):
            continue
        print(f"== {name}")
        print(f"{'mode':<10}{'group':<15}" + "".join(f"{k:>18}" for k in keys))
        for mode, value in by_mode.items():
            groups = value if "n" not in value else {"all": value}
            for group, summary in groups.items():
                print(f"{mode:<10}{group:<15}" + "".join(f"{summary[k]:>18}" for k in keys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", help="corpus directory from gnn_pruner.corpus")
    parser.add_argument("--dev", nargs="+", help="dev sets to measure label quality on")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--write", choices=MODES, help="write silver_{split}.jsonl for this mode")
    parser.add_argument("--coverage-threshold", type=float, default=COVERAGE_THRESHOLD)
    parser.add_argument("--embedder", default="st:all-MiniLM-L6-v2", help="for --dev")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = {}
    if args.corpus:
        report.update(run_corpus(args.corpus, args.modes, args.write, args.coverage_threshold))
    if args.dev:
        report.update(run_dev(args.dev, args.modes, args.embedder, args.coverage_threshold))
    print(json.dumps(report, indent=2)) if args.json else _print(report)


if __name__ == "__main__":
    main()
