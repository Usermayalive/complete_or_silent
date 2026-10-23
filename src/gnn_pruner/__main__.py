"""CLI: python -m gnn_pruner DOCUMENT "question" [--explain] [--prompt] [--json]
         [--embedder tfidf|st:<model>] [--store graphs.db] [--weights w.json] [--abstain]"""

from __future__ import annotations

import argparse
import json

from .embeddings import CachedEmbedder, load_embedder
from .pipeline import DocumentQA
from .scoring import load_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", help=".md, .txt, .pdf, .docx or .html file")
    parser.add_argument("question")
    parser.add_argument("--plan", action="store_true", help="show the retrieval plan")
    parser.add_argument("--explain", action="store_true", help="show scored subgraph")
    parser.add_argument("--prompt", action="store_true", help="show the LLM prompt")
    parser.add_argument("--json", action="store_true", help="print the evidence pack as JSON")
    parser.add_argument("--embedder", default="auto",
                        help="'auto' (dense model if installed, else TF-IDF), 'tfidf' or 'st:<model>'")
    parser.add_argument("--store", metavar="DB",
                        help="SQLite store: reuse parsed graphs and cached embeddings")
    parser.add_argument("--weights", metavar="JSON", help="scoring weight overrides (JSON object)")
    parser.add_argument("--llm", default=None,
                        help="answer with: 'claude', 'claude:<model>', 'openai:<model>' (needs API key) or 'extractive'")
    parser.add_argument("--scorer", default="linear", help="'linear', 'gnn' or 'gnn:<model dir>'")
    parser.add_argument("--abstain", action="store_true",
                        help="return no evidence when the calibrated evidence confidence is insufficient")
    args = parser.parse_args()

    embedder = load_embedder(args.embedder)
    print(f"embedder: {embedder.name}")
    store = None
    if args.store:
        from .store import GraphStore

        store = GraphStore(args.store)
        if embedder.cacheable:
            embedder = CachedEmbedder(embedder, store)
    scorer = None
    if args.scorer != "linear":
        from .gnn.scorer import load_scorer

        scorer = load_scorer(args.scorer)
    qa = DocumentQA.from_file(args.document, store=store, embedder=embedder, abstain=args.abstain,
                              scorer=scorer, llm=args.llm,
                              weights=load_weights(args.weights) if args.weights else None)
    for warning in qa.graph.warnings:
        print(f"warning: {warning}")
    result = qa.ask(args.question)
    if args.json:
        print(json.dumps(result.evidence.to_dict(), indent=2))
        return

    if args.plan:
        print(result.plan.explain(qa.graph) + "\n")
    else:
        print(f"Query type: {result.plan.query_type}  strategy: {result.plan.strategy}\n")
    print(result.answer)
    print(f"\n{result.confidence.explain()}")
    if args.explain:
        print(f"pruning: {result.pruned.explain()}")
        for ev in result.pruned.evidence:
            if ev.required_by:
                print(f"  {ev.role:<11} {ev.node.id:<8} needed by {ev.required_by}")
        print("\nScored subgraph (top 15): score = sum of contributions (weight x feature)")
        for s in result.scored[:15]:
            print(f"  {s.score:6.3f}  {s.node.id:<10} {s.node.node_type:<9} {s.explain(top=6)}")
            feats = " ".join(f"{k}={v:.2f}" for k, v in s.features.items() if v)
            print(f"          features {feats}")
            print(f"          via {', '.join(s.reasons)}")
            path = result.trace.subgraph.paths.get(s.node.id, [])
            if len(path) > 1:
                print(f"          path {' > '.join(path)}")
    if args.prompt:
        print("\n" + "-" * 60 + "\n" + result.evidence.render_prompt())


if __name__ == "__main__":
    main()
