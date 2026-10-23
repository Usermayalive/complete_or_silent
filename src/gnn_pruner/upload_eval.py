"""Readings for build step 1 (upload): structure, metadata, key facts, cost.

    python -m gnn_pruner.upload_eval data/books/sherlock.pdf data/books/two_cities.pdf \\
        --key data/books/key.json --sheet docs/fact_check_sheet.md

Readings:
- **Structure (PDFs with an outline):** the outline's entries are the reference.
  - found: share of outline entries that became headings
  - spurious: headings that are not outline entries (drop caps, captions, split lines)

  Measured with and without the outline (font-based headings only).
- **Metadata:** fields against a hand-made key. A ``null`` in the key means the field is not
  stated in the text, so the right output is no value.
- **Key facts:** proposed / kept / dropped (bad sentence ids, numbers not in the source),
  plus a sheet of sampled facts next to their source sentences for a human spot check.
- **Cost:** model calls, tokens and seconds, also per 100 pages.

Reads the trees cached by ``upload``; makes no model calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from .textutil import match_key


def structure(path: str) -> Optional[Dict]:
    if not path.lower().endswith(".pdf"):
        return None
    from .ingest import _finish
    from .ingest.pdf_loader import extract_pdf

    out: Dict = {}
    for use_toc in (False, True):
        info: Dict = {}
        blocks, title, pages, warnings = extract_pdf(Path(path), info, use_toc=use_toc)
        parsed = _finish(Path(path), blocks, title, warnings, pages, info=info)
        heads = [n.title for n in parsed.nodes if n.is_section and n.title]
        toc_keys = [match_key(t) for _, t, _ in info["toc"]]
        head_keys = {match_key(h) for h in heads}
        # A document title promoted from the first heading counts when it is an outline entry
        # (the PDF-metadata title, used when there is none, is not a heading).
        if match_key(parsed.nodes[0].title or "") in set(toc_keys):
            head_keys.add(match_key(parsed.nodes[0].title))
        found = sum(1 for k in toc_keys if k in head_keys)
        spurious = sum(1 for h in heads if match_key(h) not in set(toc_keys))
        out["outline_entries"] = len(toc_keys)
        out["with_outline" if use_toc else "fonts_only"] = {
            "headings": len(heads), "found": found,
            "found_share": round(found / len(toc_keys), 3) if toc_keys else None, "spurious": spurious}
    return out


def metadata_accuracy(fields: Dict, expected: Dict) -> Dict:
    rows = {}
    for name, want in expected.items():
        got = fields.get(name, {}).get("value")
        if isinstance(got, list):
            got = "; ".join(got)
        ok = (got is None) if want is None else (got is not None and match_key(got) == match_key(want))
        rows[name] = {"expected": want, "got": got, "correct": ok,
                      "source": fields.get(name, {}).get("source")}
    return {"fields": rows, "correct": sum(r["correct"] for r in rows.values()), "total": len(rows)}


def fact_sample(tree, n: int) -> List[Dict]:
    facts = [(sub, f) for sub in tree.of_kind("subsection") for f in sub.key_facts]
    if not facts:
        return []
    step = max(1, len(facts) // n)
    return [{"node": sub.id, "where": " / ".join(tree.path(sub.id)), "type": f["type"], "fact": f["text"],
             "source": tree.quote(f["sources"]), "pages": sub.pages} for sub, f in facts[::step][:n]]


def evaluate(paths: List[str], key: Optional[Dict] = None, sample: int = 0) -> Dict:
    from .tree import DocTree
    from .upload import cache_path

    report: Dict = {}
    for path in paths:
        cached = cache_path(path)
        if not cached.exists():
            report[Path(path).name] = {"error": f"no cached tree: run python -m gnn_pruner.upload {path}"}
            continue
        tree = DocTree.load(str(cached))
        meta = tree.root.metadata
        pages = (tree.root.pages or [0, 0])[1] or None
        enrichment = meta.get("enrichment", {})
        stats = dict(enrichment.get("stats", {}))
        usage = meta.get("usage", {}).get("metadata", {})
        calls = stats.get("calls", 0) + usage.get("calls", 0)
        tokens = (stats.get("input_tokens", 0) + stats.get("output_tokens", 0)
                  + usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
        proposed = stats.get("facts_proposed", 0)
        row = {
            "title": tree.title, "kind": tree.kind, "pages": pages, "tree": tree.stats(),
            "build_seconds": meta.get("build_seconds"),
            "enrichment": {"provider": enrichment.get("provider"), "complete": enrichment.get("complete"),
                           "stopped": stats.get("stopped")},
            "facts": {"proposed": proposed, "kept": stats.get("facts_kept", 0),
                      "kept_share": round(stats.get("facts_kept", 0) / proposed, 3) if proposed else None,
                      "dropped_bad_ids": stats.get("facts_dropped_source", 0),
                      "dropped_numbers": stats.get("facts_dropped_numbers", 0),
                      "fallbacks": stats.get("items_fallback", 0)},
            "cost": {"calls": calls, "tokens": tokens, "seconds": stats.get("seconds"),
                     "usd": stats.get("cost_usd"),
                     "calls_per_100_pages": round(100 * calls / pages, 1) if pages else None,
                     "tokens_per_100_pages": round(100 * tokens / pages) if pages else None},
            "lists": {k: len(v) for k, v in meta.get("lists", {}).items()},
            "structure": structure(path),
        }
        if key and Path(path).name in key:
            row["metadata"] = metadata_accuracy(meta.get("fields", {}), key[Path(path).name])
        if sample:
            row["fact_sample"] = fact_sample(tree, sample)
        report[Path(path).name] = row
    return report


def write_sheet(report: Dict, out: str) -> None:
    lines = ["# Key-fact spot check", "",
             "Each row is a key fact the model wrote, next to the sentences it cited (taken from the "
             "document by id). Mark **Correct?** as yes / partly / no: does the source say what the fact "
             "says?", ""]
    for name, row in report.items():
        if not row.get("fact_sample"):
            continue
        lines += [f"## {row['title']} ({name})", "", "| # | Where | Type | Fact | Source sentences | Correct? |",
                  "|---|---|---|---|---|---|"]
        for k, f in enumerate(row["fact_sample"], 1):
            cell = lambda t: t.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {k} | {cell(f['where'])} | {f['type']} | {cell(f['fact'])} | {cell(f['source'])} | |")
        lines.append("")
    Path(out).write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("documents", nargs="+")
    parser.add_argument("--key", help="JSON answer key for metadata")
    parser.add_argument("--sample", type=int, default=15, help="facts per document for the spot-check sheet")
    parser.add_argument("--sheet", help="write the spot-check sheet (markdown) here")
    parser.add_argument("--json")
    args = parser.parse_args()
    key = json.loads(Path(args.key).read_text()) if args.key else None
    report = evaluate(args.documents, key, args.sample if args.sheet else 0)
    for name, row in report.items():
        printable = {k: v for k, v in row.items() if k != "fact_sample"}
        print(name, json.dumps(printable, indent=1, ensure_ascii=False))
    if args.sheet:
        write_sheet(report, args.sheet)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
