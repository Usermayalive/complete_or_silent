"""Phase 10 readings: answer correctness, citation precision / recall, abstention, cost.

Per question:
- **correctness**:
  - ``answer_f1``: token F1 between the answer and the reference answer (numbers
    normalised, "thirty" = "30")
  - ``number_match``: every number in the reference appears in the answer
  - optional ``--judge`` (an LLM grades correct / partially_correct / incorrect against
    the reference; judge cost reported separately)
- **citations**:
  - precision: share of cited items that contain a gold snippet
  - recall: share of gold snippets inside some cited item
- **abstention**: correct on unanswerable questions; ``false_abstention`` on answerable
  ones
- **tokens / cost / latency** per question

``--dry-run`` builds every prompt and reports the expected input tokens (and cost for
known Claude models) without calling any model.

    python -m gnn_pruner.answer_eval --sets eval/contracts_v1.jsonl --llm extractive
    python -m gnn_pruner.answer_eval --sets eval/contracts_v1.jsonl --llm claude --dry-run
    python -m gnn_pruner.answer_eval --sets eval/contracts_v1.jsonl --llm claude --judge claude
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .answer import CLAUDE_PRICES, answer_question, estimate_tokens, load_provider
from .evaluation import _norm, load_cases
from .textutil import normalize_numbers

OUTPUT_TOKENS_ESTIMATE = 150
_STOP = set("a an the of to in on for and or is are be by with that this it its any all".split())

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
                   "reason": {"type": "string"}},
    "required": ["verdict", "reason"], "additionalProperties": False,
}
JUDGE_SYSTEM = ("You grade an answer to a question about a document against a reference answer. "
                "correct: it states everything the reference states (wording may differ) and nothing that "
                "contradicts it. partially_correct: it gets part right or omits a key condition. incorrect: "
                "wrong, contradicting, or an unnecessary refusal. Ignore citation markers like [E1].")


def _tokens(text: str) -> List[str]:
    text = normalize_numbers(re.sub(r"\[E\d+[^\]]*\]", " ", text)).lower()
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text) if t not in _STOP]


def answer_f1(answer: str, reference: str) -> float:
    a, r = _tokens(answer), _tokens(reference)
    if not a or not r:
        return 0.0
    common = sum(min(a.count(t), r.count(t)) for t in set(r))
    if not common:
        return 0.0
    p, q = common / len(a), common / len(r)
    return 2 * p * q / (p + q)


def number_match(answer: str, reference: str) -> Optional[float]:
    nums = lambda s: set(re.findall(r"\d+(?:\.\d+)?", normalize_numbers(s)))
    ref = nums(re.sub(r"\(Section[^)]*\)|\(\d[^)]*\)|Sections? [\d.]+(?:\(\w\))?", " ", reference))
    if not ref:
        return None
    return float(ref <= nums(answer))


def judge(provider, question: str, answer: str, reference: str) -> Dict:
    user = f"Question: {question}\n\nReference answer: {reference}\n\nAnswer to grade: {answer}"
    c = provider.complete(JUDGE_SYSTEM, user, JUDGE_SCHEMA)
    try:
        data = json.loads(c.text)
    except json.JSONDecodeError:
        data = {"verdict": "incorrect", "reason": "unparsed judge reply"}
    data["cost_usd"] = c.cost_usd
    return data


def _systems(system: str, provider_spec: Optional[str], embedder):
    """Factory: document path -> callable(question) -> (pack, Answer)."""
    from .pipeline import DocumentQA

    cache = {}

    def run(case):
        if case.document not in cache:
            if system == "rule":
                cache[case.document] = DocumentQA.from_file(case.document, embedder=embedder)
            else:
                from .gnn.spec_qa import SpecQA

                cache[case.document] = SpecQA.from_file(case.document, str(Path(__file__).resolve().parents[2]
                                                                           / "models/g4/cea_s0/joint"),
                                                        embedder=embedder)
        qa = cache[case.document]
        if system == "rule":
            result = qa.ask(case.question)
            return result.evidence
        return qa.ask(case.question).evidence

    return run


def evaluate(paths: List[str], system: str = "rule", llm: Optional[str] = None, judge_spec: Optional[str] = None,
             limit: Optional[int] = None, dry_run: bool = False, strict: bool = False, abstain: bool = True,
             resume: Optional[str] = None) -> Dict:
    """``resume``: a JSONL file; every finished row is appended to it and rows already in it
    are reused, so a run stopped by a daily quota continues where it left off."""
    from .answer import QuotaExhausted
    from .embeddings import load_embedder

    embedder = load_embedder("st:all-MiniLM-L6-v2")
    provider = None if dry_run else load_provider(llm)
    judge_provider = load_provider(judge_spec) if judge_spec and not dry_run else None
    get_pack = _systems(system, llm, embedder)
    done = {}
    if resume and Path(resume).exists():
        for line in Path(resume).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["set"], row["id"])] = row
    report, stopped = {}, None
    for path in paths:
        cases = load_cases(path)[:limit] if limit else load_cases(path)
        rows = []
        for case in cases:
            key = (Path(path).stem, case.id)
            if key in done and not dry_run:
                rows.append(done[key])
                continue
            if stopped:
                continue
            pack = get_pack(case)
            if dry_run:
                rows.append({"input_tokens": estimate_tokens(pack) if pack.items else 0,
                             "answerable": case.answerable})
                continue
            try:
                ans = answer_question(pack, provider, abstain=abstain, strict=strict)
            except QuotaExhausted as exc:
                stopped = str(exc)
                continue
            gold = [_norm(g) for g in case.gold_evidence]
            cited_texts = [_norm(e["text"]) for e in ans.evidence]
            row = {"id": case.id, "answerable": case.answerable, "abstained": ans.abstained,
                   "flags": ans.flags, "input_tokens": ans.input_tokens, "output_tokens": ans.output_tokens,
                   "cost_usd": ans.cost_usd, "latency_ms": ans.latency_ms, "route": ans.route,
                   "answer": ans.answer, "cited": ans.cited, "served_by": ans.served_by}
            if case.answerable:
                row["false_abstention"] = float(ans.abstained)
                if case.reference_answer and not ans.abstained:
                    row["answer_f1"] = answer_f1(ans.answer, case.reference_answer)
                    nm = number_match(ans.answer, case.reference_answer)
                    if nm is not None:
                        row["number_match"] = nm
                if gold and not ans.abstained:
                    row["citation_precision"] = (sum(any(g in t for g in gold) for t in cited_texts) / len(cited_texts)
                                                 if cited_texts else 0.0)
                    row["citation_recall"] = sum(any(g in t for t in cited_texts) for g in gold) / len(gold)
                if judge_provider and case.reference_answer:
                    verdict = judge(judge_provider, case.question, ans.answer, case.reference_answer)
                    row["judge"] = verdict["verdict"]
                    row["judge_score"] = {"correct": 1.0, "partially_correct": 0.5}.get(verdict["verdict"], 0.0)
                    row["judge_cost_usd"] = verdict.get("cost_usd")
            else:
                row["abstention_correct"] = float(ans.abstained)
            rows.append(row)
            if resume:
                with open(resume, "a") as fh:
                    fh.write(json.dumps({"set": key[0], **row}, default=str) + "\n")
        report[Path(path).stem] = summarize(rows, llm, dry_run)
        report[Path(path).stem]["_rows"] = rows
        if stopped:
            report[Path(path).stem]["incomplete"] = f"{len(rows)}/{len(cases)} ({stopped})"
    return report


def summarize(rows: List[Dict], llm: Optional[str], dry_run: bool) -> Dict:
    def mean(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(v)), 4) if v else None

    out = {"questions": len(rows)}
    if dry_run:
        tokens = [r["input_tokens"] for r in rows]
        out.update(input_tokens_total=int(sum(tokens)), input_tokens_mean=round(float(np.mean(tokens)), 1),
                   output_tokens_estimate=OUTPUT_TOKENS_ESTIMATE * len(rows))
        model = (llm or "").split(":", 1)[1] if llm and ":" in llm else ("claude-opus-5-5" if llm == "claude" else None)
        if model in CLAUDE_PRICES:
            pin, pout = CLAUDE_PRICES[model]
            out["estimated_cost_usd"] = round((sum(tokens) * pin + OUTPUT_TOKENS_ESTIMATE * len(rows) * pout) / 1e6, 4)
        return out
    for key in ("answer_f1", "number_match", "judge_score", "citation_precision", "citation_recall",
                "false_abstention", "abstention_correct", "input_tokens", "output_tokens", "latency_ms"):
        out[key] = mean(key)
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    out["cost_usd_total"] = round(sum(costs), 4) if costs else None
    out["cost_usd_per_question"] = round(sum(costs) / len(costs), 5) if costs else None
    judge_costs = [r["judge_cost_usd"] for r in rows if r.get("judge_cost_usd") is not None]
    out["judge_cost_usd_total"] = round(sum(judge_costs), 4) if judge_costs else None
    out["flagged"] = sum(1 for r in rows if r.get("flags") and not r.get("abstained"))
    out["served_by_fallback"] = sum(1 for r in rows if r.get("served_by"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=["eval/contracts_v1.jsonl"])
    parser.add_argument("--system", choices=["rule", "v3.0"], default="rule")
    parser.add_argument("--llm", default="extractive",
                        help="extractive | claude[:<model>] | openai:<model> | ollama:<model> | gemini:<model>")
    parser.add_argument("--judge", help="provider spec for an LLM judge of correctness (costs extra)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true", help="estimate tokens and cost; no model calls")
    parser.add_argument("--strict", action="store_true", help="abstain on flagged citations")
    parser.add_argument("--no-abstain", action="store_true",
                        help="answer even when retrieval confidence is insufficient (Phase 7 signal)")
    parser.add_argument("--json")
    parser.add_argument("--resume", help="JSONL of finished rows: appended to, and reused on the next run")
    args = parser.parse_args()
    report = evaluate(args.sets, args.system, args.llm, args.judge, args.limit, args.dry_run, args.strict,
                      abstain=not args.no_abstain, resume=args.resume)
    for name, summary in report.items():
        print(name, json.dumps({k: v for k, v in summary.items() if k != "_rows"}))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
