"""Latency benchmark: how retrieval scales with document length.

Builds the SaaS example contract padded with synthetic sections to N paragraphs,
then times indexing and each query stage.

    python -m gnn_pruner.benchmark --sizes 300 1000 3000 10000 --embedder tfidf
    python -m gnn_pruner.benchmark --sizes 300 3000 --embedder st --vector-index faiss
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from .document_graph import DocumentGraph
from .embeddings import load_embedder
from .pipeline import DocumentQA
from .synthetic import distractors

BASE = Path(__file__).resolve().parent.parent.parent / "examples" / "saas_agreement.md"
QUESTIONS = [
    "Can the Provider increase fees on renewal?",
    "What is the definition of Customer Data?",
    "What happens if availability falls below the Service Levels?",
    "Compare the renewal and termination provisions",
    "Summarize the service levels and service credits",
    "Is there a conflict in the critical incident response time?",
    "Where do they keep our files?",
    "Is there a 72 hour breach notification requirement?",
]
PARAGRAPHS_PER_SECTION = 7.5  # average of the generator's 6-9


def build_text(paragraphs: int) -> str:
    base = BASE.read_text(encoding="utf-8").rstrip() + "\n\n"
    sections = max(0, round((paragraphs - 25) / PARAGRAPHS_PER_SECTION))
    return base + distractors("Provider", "Customer", 11, seed=7, n_sections=sections)


def run(size: int, embedder_spec: str, vector_index: str, repeats: int) -> dict:
    text = build_text(size)
    start = time.perf_counter()
    graph = DocumentGraph.from_text(text, title="benchmark")
    parse_s = time.perf_counter() - start
    start = time.perf_counter()
    qa = DocumentQA(graph, embedder=load_embedder(embedder_spec), vector_index=vector_index)
    index_s = time.perf_counter() - start

    stages: dict = {}
    totals = []
    for _ in range(repeats):
        for question in QUESTIONS:
            start = time.perf_counter()
            result = qa.ask(question)
            totals.append(time.perf_counter() - start)
            for stage, seconds in result.trace.timings.items():
                stages.setdefault(stage, []).append(seconds)
    return {
        "paragraphs": len(graph.content_nodes(("paragraph",))),
        "parse_s": parse_s,
        "index_s": index_s,
        "vector_index": qa.index.vectors.kind,
        "query_ms": {k: 1000 * statistics.median(v) for k, v in stages.items()},
        "total_ms": 1000 * statistics.median(totals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", type=int, default=[300, 1000, 3000])
    parser.add_argument("--embedder", default="tfidf")
    parser.add_argument("--vector-index", default="auto", choices=["auto", "numpy", "faiss"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    stages = ["plan", "candidates", "expansion", "scoring", "pruning"]
    print(f"embedder={args.embedder}  (median ms per question over {len(QUESTIONS)} questions)")
    print(f"{'paras':>7}{'parse s':>9}{'index s':>9}{'vectors':>9}"
          + "".join(f"{s:>12}" for s in stages) + f"{'total':>10}")
    for size in args.sizes:
        r = run(size, args.embedder, args.vector_index, args.repeats)
        print(f"{r['paragraphs']:>7}{r['parse_s']:>9.2f}{r['index_s']:>9.2f}{r['vector_index']:>9}"
              + "".join(f"{r['query_ms'].get(s, 0):>12.1f}" for s in stages)
              + f"{r['total_ms']:>10.1f}")


if __name__ == "__main__":
    main()
