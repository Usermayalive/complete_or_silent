"""Upload (WORKFLOW A): document -> enriched tree, cached by file hash.

    tree = upload("book.pdf", llm="gemini:gemini-3.1-flash-lite")

Steps:
1. **Read and build the tree** (``tree.py``).
2. **Root metadata** (``metadata.py``).
3. **Key facts and summaries** (``enrich.py``).

The result is saved to ``data/trees/<name>-<hash>.json`` after every model call:
- Running again reuses it, and continues an enrichment stopped by a quota.
- A different ``--llm`` re-enriches.
- ``--rebuild`` starts over.

    python -m gnn_pruner.upload data/books/two_cities.pdf --llm gemini:gemini-3.1-flash-lite
    python -m gnn_pruner.upload examples/saas_agreement.md --show facts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from .tree import TREE_VERSION, DocTree, tree_from_file

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "trees"


def cache_path(path: str, cache_dir: Path = DEFAULT_CACHE) -> Path:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    return Path(cache_dir) / f"{Path(path).stem}-{digest}.json"


def upload(path: str, llm: Optional[str] = None, cache_dir: Path = DEFAULT_CACHE, rebuild: bool = False,
           enrich: bool = True, parser: str = "native", provider=None, refresh_metadata: bool = False,
           re_enrich: bool = False, **enrich_options) -> DocTree:
    """``provider`` overrides ``llm`` (a Provider object, e.g. a fake in tests).
    ``refresh_metadata`` re-runs metadata extraction on a cached tree."""
    from .answer import load_provider
    from .enrich import Enricher
    from .metadata import extract_metadata

    target = cache_path(path, cache_dir)
    tree = None
    if target.exists() and not rebuild:
        tree = DocTree.load(str(target))
        if tree.version != TREE_VERSION:
            tree = None
    if tree is None:
        started = time.perf_counter()
        tree = tree_from_file(path, parser)
        tree.root.metadata["build_seconds"] = round(time.perf_counter() - started, 1)
        tree.save(str(target))
    provider = provider if provider is not None else load_provider(llm)
    if refresh_metadata:
        tree.root.metadata.pop("fields", None)
        tree.root.metadata.get("usage", {}).pop("metadata", None)
    if "fields" not in tree.root.metadata or (provider is not None
                                              and "metadata" not in tree.root.metadata.get("usage", {})):
        extract_metadata(tree, provider)
        tree.save(str(target))
    if enrich:
        if re_enrich:
            from .enrich import reset

            reset(tree)
        Enricher(provider, **enrich_options).run(tree, save=lambda t: t.save(str(target)))
    tree.root.metadata["cache"] = str(target)
    return tree


def _show(tree: DocTree, what: str) -> str:
    if what == "outline":
        return tree.outline()
    if what == "stats":
        meta = tree.root.metadata
        return json.dumps({"title": tree.title, "kind": tree.kind, "tree": tree.stats(),
                           "fields": {k: v["value"] for k, v in meta.get("fields", {}).items()},
                           "enrichment": meta.get("enrichment"), "metadata_usage": meta.get("usage"),
                           "lists": {k: len(v) for k, v in meta.get("lists", {}).items()}}, indent=2)
    if what == "facts":
        lines = [f"{tree.title}: {tree.root.summary or ''}"]
        for ch in tree.chapters():
            lines.append(f"\n## {ch.title}\n   {ch.summary or ''}")
            for sub in tree.of_kind("subsection"):
                if sub.id.startswith(ch.id + "."):
                    for f in sub.key_facts:
                        lines.append(f"   - [{f['type']}] {f['text']}  {f['sources']}")
        return "\n".join(lines)
    if what == "lists":
        return json.dumps(tree.root.metadata.get("lists", {}), indent=2, ensure_ascii=False)
    if what == "json":
        return json.dumps(tree.to_dict(), indent=2, ensure_ascii=False)
    raise ValueError(what)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("document")
    parser.add_argument("--llm", help="extractive (default) | gemini:<model> | claude[:<model>] | openai:<model>")
    parser.add_argument("--no-enrich", action="store_true", help="tree and metadata only")
    parser.add_argument("--rebuild", action="store_true", help="ignore the cached tree")
    parser.add_argument("--refresh-metadata", action="store_true", help="re-extract root metadata")
    parser.add_argument("--re-enrich", action="store_true", help="discard key facts and summaries, redo them")
    parser.add_argument("--parser", default="native", choices=["native", "docling"])
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--batch-tokens", type=int, default=6000)
    parser.add_argument("--show", default="stats", choices=["stats", "outline", "facts", "lists", "json"])
    args = parser.parse_args()
    tree = upload(args.document, args.llm, Path(args.cache_dir), args.rebuild, not args.no_enrich, args.parser,
                  refresh_metadata=args.refresh_metadata, re_enrich=args.re_enrich, batch_tokens=args.batch_tokens)
    print(_show(tree, args.show))
    print(f"\nsaved: {tree.root.metadata['cache']}")


if __name__ == "__main__":
    main()
