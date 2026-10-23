"""G2 readings: paragraph-level ranking of every L3 node of a document.

Methods:
- ``bm25``: BM25 over the coarse-node texts (L3 rows)
- ``dense``: cosine of paragraph and question embeddings
- ``linear``: the Phase 5-8 rule-based pipeline (plan → retrieve → expand →
  LinearScorer). Its scored content nodes are mapped to their paragraphs in score
  order; paragraphs it never reached rank after all the others.
- one entry per trained model directory (GNN over the whole coarse graph)

Metrics: Recall@10, MRR, nDCG@10 and hit@1 over the answerable questions. For models
with a policy, the deterministic top-down keep (p > 0.5) also gives the share of
paragraphs kept and the share of answers whose paragraph is kept.

    python -m gnn_pruner.gnn.evaluate --models models/coarse_gnn_v1/supervised models/coarse_gnn_v1/ppo \\
        --sets corpus-val eval/calibration_v1.jsonl eval/contracts_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .features import DocTensors, doc_tensors, node_features
from .train import (Example, evaluate_policy, load, load_split, mean_metrics, paragraph_metrics,
                    rank_l3, scores_for)


def _eval_set(path: str, embedder) -> Tuple[Dict[str, DocTensors], List[Example], Dict[str, str]]:
    """Dev/test set → coarse tensors per document and examples with gold paragraphs.
    Nothing is written; these documents never reach training."""
    from ..corpus import embed_layered
    from ..document_graph import DocumentGraph
    from ..evaluation import load_cases
    from ..layers import four_layer
    from ..silver import gold_nodes

    docs: Dict[str, DocTensors] = {}
    sources: Dict[str, str] = {}
    examples = []
    for case in load_cases(path):
        if not case.answerable:
            continue
        doc_id = case.document
        if doc_id not in docs:
            layered = four_layer(DocumentGraph.from_file(case.document), doc_id)
            _, ids, matrix = embed_layered(layered, embedder)
            docs[doc_id] = doc_tensors(layered, ids, matrix)
            sources[doc_id] = case.document
        doc = docs[doc_id]
        gold = [paras & set(doc.row) for _, paras in gold_nodes(doc.layered, case.gold_evidence)]
        examples.append(Example(case.id, doc_id, case.question,
                                np.asarray(embedder.encode_query(case.question), dtype=np.float32),
                                {}, [g for g in gold if g], case.tags))
    return docs, examples, sources


def _baseline_rankings(doc: DocTensors, e: Example) -> Dict[str, List[str]]:
    from ..index import tokenize

    bm25 = doc.bm25.scores(tokenize(e.question))
    return {"bm25": rank_l3(doc, bm25), "dense": rank_l3(doc, doc.emb @ e.query_vector)}


def _linear_ranking(qa, doc: DocTensors, question: str) -> List[str]:
    l3 = {doc.ids[i] for i in doc.l3}
    out, seen = [], set()
    for s in qa.ask(question).scored:
        nid = s.node.id
        if nid not in l3:
            parent = doc.layered.nodes.get(nid)
            nid = parent.parent_id if parent is not None else None
        if nid in l3 and nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


def run(models: Sequence[str], sets: Sequence[str], corpus: str = "data/corpus_v1",
        linear: bool = True) -> Dict:
    from ..embeddings import load_embedder
    from ..pipeline import DocumentQA

    loaded = {m: load(m) for m in models}
    embedder_name = next(iter(loaded.values()))[1]["embedder"] if loaded else "st:all-MiniLM-L6-v2"
    embedder = load_embedder(embedder_name)
    report: Dict = {}
    for name in sets:
        if name == "corpus-val":
            docs, examples = load_split(corpus, "val", embedder)
            sources = {d: str(Path(corpus) / "docs" / "val" / f"{d}.md") for d in docs}
        else:
            docs, examples, sources = _eval_set(name, embedder)
        rows: Dict[str, List[Dict]] = {}
        latency: Dict[str, List[float]] = {}
        qas = {}
        for e in examples:
            doc = docs[e.doc_id]
            for method, ranking in _baseline_rankings(doc, e).items():
                rows.setdefault(method, []).append(paragraph_metrics(ranking, e.gold))
            if linear:
                if e.doc_id not in qas:
                    qas[e.doc_id] = DocumentQA.from_file(sources[e.doc_id], embedder=embedder)
                rows.setdefault("linear", []).append(
                    paragraph_metrics(_linear_ranking(qas[e.doc_id], doc, e.question), e.gold))
            for path, (model, _) in loaded.items():
                started = time.perf_counter()
                scores = scores_for(model, doc, e.question, e.query_vector)
                latency.setdefault(path, []).append((time.perf_counter() - started) * 1000)
                rows.setdefault(path, []).append(paragraph_metrics(rank_l3(doc, scores), e.gold))
        summary = {method: mean_metrics(r) for method, r in rows.items()}
        for path, (model, meta) in loaded.items():
            summary[path]["ms_per_question"] = round(float(np.mean(latency[path])), 2)
            summary[path].update({f"policy_{k}": v for k, v in evaluate_policy(model, docs, examples).items()})
        summary["_docs"] = {"documents": len(docs), "questions": len(examples),
                            "mean_coarse_nodes": round(float(np.mean([len(d.ids) for d in docs.values()])), 1)}
        report[Path(name).stem if name != "corpus-val" else name] = summary
    return report


def print_report(report: Dict) -> None:
    keys = ["recall@10", "mrr", "ndcg@10", "hit@1", "n", "ms_per_question", "policy_kept_l3_share",
            "policy_gold_recall_kept"]
    for name, summary in report.items():
        print(f"== {name}  {summary.get('_docs')}")
        print(f"{'method':<44}" + "".join(f"{k.replace('policy_', 'p_')[:14]:>15}" for k in keys))
        for method, m in summary.items():
            if method.startswith("_"):
                continue
            print(f"{method[-44:]:<44}" + "".join(f"{m.get(k, ''):>15}" for k in keys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--sets", nargs="+", default=["corpus-val"])
    parser.add_argument("--corpus", default="data/corpus_v1")
    parser.add_argument("--no-linear", action="store_true")
    parser.add_argument("--json", metavar="FILE", help="also write the report as JSON")
    args = parser.parse_args()
    report = run(args.models, args.sets, args.corpus, linear=not args.no_linear)
    print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
