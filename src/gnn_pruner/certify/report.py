"""P3 report: certificate honesty (RQ3) and cost (RQ4) from benchmark run rows.

    python -m gnn_pruner.certify.report eval/runs/long_traps2_S_*.jsonl [--family traps2 --tier S]

For every rows file (one system x one model) it reports:

    certified       share of questions with a closure or absence certificate (not "uncertified")
    false cert.     certificates that are wrong: a closure certificate on an answer that is
                    incorrect or misses a required provision; an absence certificate when the
                    answer was in the document. With a Clopper-Pearson 95% upper bound.
    acc (all / certified / uncertified)   answer correctness overall and split by outcome:
                    a useful certificate separates right from wrong answers
    calls, tokens, seconds per question  (RQ4)

A row needs ``certificate`` (from ``--system workers_certify``) for the certificate columns;
plain systems get only accuracy and cost. Rows of other kinds than the benchmark's are
reported per kind.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .canaries import clopper_pearson_upper


def load_rows(path: str) -> List[Dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_cases(family: str, tier: str) -> Dict[str, Dict]:
    path = Path(__file__).resolve().parents[3] / "eval" / "long" / f"{family}_{tier}.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in path.read_text().splitlines() if l.strip()}


def false_certificate(row: Dict, case: Optional[Dict]) -> Optional[bool]:
    """True / False for a certified row, None when the row is uncertified or has none."""
    kind = row.get("certificate")
    if kind not in ("closure", "absence"):
        return None
    answerable = case["answerable"] if case else row.get("answerable")
    if kind == "absence":
        return bool(answerable)                     # "not there" while it was there
    complete = row.get("qualifier_recall", 1.0) >= 1.0
    return not (bool(row.get("correct")) and complete)


def _rate(k: int, n: int) -> str:
    if not n:
        return "—"
    return f"{100 * k / n:.0f}% ({k}/{n}; ≤{100 * clopper_pearson_upper(k, n):.0f}%)"


def summarize(rows: List[Dict], cases: Dict[str, Dict]) -> Dict:
    n = len(rows)
    out: Dict = {"n": n}
    acc = [bool(r.get("correct")) for r in rows if "correct" in r]
    out["accuracy"] = (sum(acc), len(acc))
    if any("certificate" in r for r in rows):
        certified = [r for r in rows if r.get("certificate") in ("closure", "absence")]
        uncertified = [r for r in rows if r.get("certificate") not in ("closure", "absence")]
        false = [false_certificate(r, cases.get(r["id"])) for r in certified]
        out["certified"] = (len(certified), n)
        out["false_certificates"] = (sum(1 for f in false if f), len(certified))
        out["acc_certified"] = (sum(1 for r in certified if r.get("correct")), len(certified))
        out["acc_uncertified"] = (sum(1 for r in uncertified if r.get("correct")), len(uncertified))
        reasons = defaultdict(int)
        for r in uncertified:
            text = r.get("certificate_text") or ""
            for reason in re.split(r";\s*", text.replace("UNCERTIFIED: ", "").split(" (read")[0]):
                if reason:
                    reasons[re.sub(r"\d+", "N", reason)] += 1
        out["uncertified_reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    for key in ("calls", "input_tokens", "output_tokens", "llm_ms"):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        out[key] = round(sum(vals) / len(vals), 1) if vals else None
    return out


def by_kind(rows: List[Dict], cases: Dict[str, Dict]) -> Dict[str, Dict]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        groups[r.get("kind", "?")].append(r)
    return {k: summarize(v, cases) for k, v in sorted(groups.items())}


def table(results: Dict[str, Dict]) -> str:
    """Markdown: one line per rows file."""
    head = ("| Run | n | Accuracy | Certified | False certificates (≤ 95% bound) | Acc. certified | "
            "Acc. uncertified | Calls / q | Input tok / q |")
    lines = [head, "|" + "---|" * 9]
    for name, s in results.items():
        cert = s.get("certified")
        lines.append(" | ".join([
            f"| {name}", str(s["n"]), _rate(*s["accuracy"]),
            _rate(*cert) if cert else "—",
            _rate(*s["false_certificates"]) if cert else "—",
            _rate(*s["acc_certified"]) if cert else "—",
            _rate(*s["acc_uncertified"]) if cert else "—",
            str(s.get("calls") or "—"), str(s.get("input_tokens") or "—")]) + " |")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", nargs="+")
    parser.add_argument("--family", default="traps2")
    parser.add_argument("--tier", default="S")
    parser.add_argument("--by-kind", action="store_true")
    args = parser.parse_args(argv)
    cases = load_cases(args.family, args.tier)
    results = {}
    for path in args.rows:
        name = Path(path).stem.replace(f"long_{args.family}_{args.tier}_", "")
        rows = load_rows(path)
        results[name] = summarize(rows, cases)
        if args.by_kind:
            for kind, s in by_kind(rows, cases).items():
                results[f"{name} · {kind}"] = s
    print(table(results))
    for name, s in results.items():
        if s.get("uncertified_reasons") and "·" not in name:
            print(f"\n{name}: uncertified because " + "; ".join(f"{k} ({v})" for k, v in s["uncertified_reasons"].items()))


if __name__ == "__main__":
    main()
