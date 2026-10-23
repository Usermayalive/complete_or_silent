"""G3 scale benchmark: two-stage (coarsened) vs single-pass FineGNN on long documents.

Documents of each size are generated with known-answer questions (``corpus.py``), so
quality is measured alongside cost. Per size:
- build time: parse, four layers, embed, tables (offline indexing)
- nodes in the document and nodes each stage processes
- latency per question: coarse stage, fine stage
- answer recall: answer paragraph kept in stage 1, answer sentence in the merged set,
  share of sentences merged

The single-pass baseline is skipped above ``--flat-max-nodes`` (memory: it builds input
features for every node at once).

    python -m gnn_pruner.gnn.bench --pages 100 1000 10000 --two-stage models/two_stage_v1/joint --flat models/g3_flat
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..corpus import embed_layered, generate_contract
from ..document_graph import DocumentGraph
from ..layers import four_layer
from ..silver import gold_nodes
from .cross_attention import load as load_two_stage
from .two_stage import StageConfig, fine_table, run_stages


def bench(pages: int, two_stage, flat: str, questions: int = 8, seed: int = 21,
          flat_max_nodes: int = 120_000, max_keep: Optional[int] = None) -> Dict:
    """``two_stage``: one model directory or a list (all run on the same document)."""
    from ..embeddings import load_embedder

    paths = [two_stage] if isinstance(two_stage, str) else list(two_stage)
    models = {p: load_two_stage(p) for p in paths}
    coarse, fine, meta = models[paths[0]]
    _, flat_fine, flat_meta = load_two_stage(flat) if flat else (None, None, None)
    embedder = load_embedder(meta["embedder"])
    cfg = StageConfig(**{**meta["stage_config"], "max_keep": max_keep})
    out: Dict = {"pages": pages, "max_keep": max_keep}
    t = time.perf_counter()
    doc = generate_contract(f"bench{pages}", pages, seed=seed, questions_per_doc=questions)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "doc.md"
        path.write_text(doc.text)
        graph = DocumentGraph.from_file(str(path))
    layered = four_layer(graph, "bench")
    out["parse_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    _, ids, matrix = embed_layered(layered, embedder)
    out["embed_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    table = fine_table(layered, ids, matrix, meta.get("pooling", "mean"))
    out["table_s"] = round(time.perf_counter() - t, 1)
    out.update({"words": doc.words, "nodes": len(table.ids), "paragraphs": len(table.doc.l3),
                "sentences": len(table.sentences), "coarse_nodes": len(table.doc.ids)})
    names = {p: ("two_stage" if i == 0 else p) for i, p in enumerate(paths)}
    rows: Dict[str, List[Dict]] = {**{n: [] for n in names.values()}, "flat": []}
    for q in doc.questions:
        qvec = np.asarray(embedder.encode_query(q.question), dtype=np.float32)
        found = [(h, p) for h, p in gold_nodes(layered, q.gold_evidence) if h]
        leaves = [h for h, _ in found]
        paras = [p for _, p in found]
        systems = [(names[p], c, f) for p, (c, f, _) in models.items()]
        if flat_fine is not None and len(table.ids) <= flat_max_nodes:
            systems.append(("flat", None, flat_fine))
        for name, c, f in systems:
            result = run_stages(c, f, table, q.question, qvec, cfg)
            sentence_rows = [i for i, r in enumerate(result.rows) if table.layer[r] == 4]
            probs = result.fine_probs[sentence_rows]
            sent_ids = [table.ids[result.rows[i]] for i in sentence_rows]
            merged = {sid for sid, p in zip(sent_ids, probs) if p > cfg.merge_ratio * probs.mean()}
            kept_ids = {table.doc.ids[k] for k in result.kept_l3}
            top10 = {sent_ids[i] for i in np.argsort(-probs)[:10]}
            rows[name].append({
                "coarse_ms": result.coarse_ms, "fine_ms": result.fine_ms,
                "total_ms": result.coarse_ms + result.fine_ms,
                "fine_nodes": len(result.rows),
                "stage1_recall": np.mean([bool(p & kept_ids) for p in paras]) if c is not None else 1.0,
                "recall@10": np.mean([bool(h & top10) for h in leaves]),
                "merged_recall": np.mean([bool(h & merged) for h in leaves]),
                "merged_share": len(merged) / len(table.sentences)})
    for name, r in rows.items():
        out[name] = {k: round(float(np.mean([x[k] for x in r])), 4) for k in r[0]} if r else "skipped (memory)"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", nargs="+", type=int, default=[100, 1000])
    parser.add_argument("--two-stage", nargs="+", default=["models/two_stage_v1/joint"])
    parser.add_argument("--flat", default="models/g3_flat")
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--flat-max-nodes", type=int, default=120_000)
    parser.add_argument("--max-keep", type=int, help="absolute cap on paragraphs kept by stage 1")
    parser.add_argument("--json")
    args = parser.parse_args()
    results = []
    for pages in args.pages:
        r = bench(pages, args.two_stage, args.flat, args.questions, flat_max_nodes=args.flat_max_nodes,
                  max_keep=args.max_keep)
        print(json.dumps(r), flush=True)
        results.append(r)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
