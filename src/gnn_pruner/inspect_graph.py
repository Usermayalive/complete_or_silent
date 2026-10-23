"""Inspect a document graph: stats, warnings, integrity check, outline.

    python -m gnn_pruner.inspect_graph contract.pdf --outline
    python -m gnn_pruner.inspect_graph contract.pdf --save graph.json
    python -m gnn_pruner.inspect_graph contract.pdf --store graphs.db
    python -m gnn_pruner.inspect_graph graph.json            # a saved graph
"""

from __future__ import annotations

import argparse
import sys

from .document_graph import DocumentGraph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="document (.md/.txt/.pdf/.docx/.html) or saved graph .json")
    parser.add_argument("--outline", action="store_true", help="print the section tree")
    parser.add_argument("--save", metavar="JSON", help="write the graph as JSON")
    parser.add_argument("--store", metavar="DB", help="put the graph in a SQLite store")
    args = parser.parse_args()

    if args.path.endswith(".json"):
        graph = DocumentGraph.load(args.path)
    elif args.store:
        from .store import GraphStore

        graph = GraphStore(args.store).load_or_ingest(args.path)
    else:
        graph = DocumentGraph.from_file(args.path)

    stats = graph.stats()
    print(f"title: {graph.title}")
    print("nodes: " + ", ".join(f"{k}={v}" for k, v in sorted(stats["nodes"].items())))
    print("edges: " + ", ".join(f"{k}={v}" for k, v in sorted(stats["edges"].items())))
    print(f"definitions: {', '.join(sorted(graph.definitions)) or '-'}")
    for warning in graph.warnings:
        print(f"warning: {warning}")
    problems = graph.validate()
    print("integrity: ok" if not problems else f"integrity: {len(problems)} problem(s)")
    for problem in problems:
        print(f"  {problem}")
    if args.outline:
        print()
        print(graph.outline())
    if args.save:
        graph.save(args.save)
        print(f"saved {args.save}")
    if args.store:
        print(f"stored in {args.store}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
