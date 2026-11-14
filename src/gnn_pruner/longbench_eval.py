"""Readings on the long-document benchmark (build step 2).

Systems (one per process, so a 10,000-page document fits in memory):
- ``rule``: our rule-based pipeline (planner -> hybrid retrieval -> expansion -> scoring ->
  pruning)
- ``v3``: the GNN pipeline (CoarseGNN / FineGNN / cross-edge attention)
- ``rag``: plain vector search, the common baseline
  - ~300-token chunks, each prefixed with its chapter / section titles
  - MiniLM embeddings, top-5 cosine (~1,300 tokens)
- ``rag1``: the same at top-1 (~300 tokens, the pipeline's budget)
- ``full``: the whole document sent to the model (long context). Only with ``--llm``, and
  only where the document fits the model.
- ``workers``: chapter workers + central board behind the relevance gate (steps 3-4).
  Only with ``--llm``; the document is enriched first (cached).
- ``workers_nogate``: the same without the gate (every chapter unless the question names
  one): the "workers on every chapter" system of the comparison.

Readings without a model (free):
- **evidence hit:** the gold text is in what the system would send. Measured as word-3-gram
  coverage of the best gold span, hit at >= 0.5. A two-hop question needs both parts.
- **answer overlap** (NarrativeQA, which has no spans): share of a reference answer's
  content words present in the pack (a weak signal)
- **abstention** (rule only): its confidence signal says "insufficient"
- **costs:** pack tokens, retrieval latency, build time

With ``--llm`` (and optionally ``--judge``):
- answer the question from the pack
- planted facts: the answer contains the planted value
- unanswerable questions: the model abstains
- NarrativeQA: token F1 against both human answers
- judge verdicts where a judge is given

Rows are appended to ``eval/runs/long_<family>_<tier>_<system>[_<llm>].jsonl`` as they
finish; re-running skips finished questions.

    python -m gnn_pruner.longbench_eval --family contracts --tier S --system rule
    python -m gnn_pruner.longbench_eval --family books --tier M --system rag --llm gemini:gemini-3.5-flash-lite --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval" / "long"
RUNS = ROOT / "eval" / "runs"
GNN_MODEL = ROOT / "models" / "g4" / "cea_s0" / "joint"
RAG_CHUNK_TOKENS = 300
RAG_K = 5
HIT = 0.5
_STOP = set("a an the of to in on for and or is are was were be by with that this it its as at from his her he "
            "she they them their what who whom which how why when where did does do".split())


# ------------------------------------------------------------------ scoring

def _toks(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def coverage(gold: str, pack_text: str, n: int = 3) -> float:
    """Share of the gold text's word n-grams found in the pack (1.0 = all of it)."""
    g = _toks(gold)
    if len(g) < n:
        return float(" ".join(g) in " ".join(_toks(pack_text))) if g else 0.0
    grams = {tuple(g[i:i + n]) for i in range(len(g) - n + 1)}
    p = _toks(pack_text)
    have = {tuple(p[i:i + n]) for i in range(len(p) - n + 1)}
    return len(grams & have) / len(grams)


def answer_overlap(answers: List[str], pack_text: str) -> float:
    words = set(_toks(pack_text))
    best = 0.0
    for a in answers:
        content = [t for t in _toks(a) if t not in _STOP and len(t) > 1]
        if content:
            best = max(best, sum(t in words for t in content) / len(content))
    return best


def contains_answer(answer: str, reference: str) -> bool:
    """Every content word of the reference is in the answer ("with a drawing of a fox" is
    answered by "signed with a small drawing of a fox"): strict on the planted values,
    tolerant of wording."""
    from .textutil import normalize_numbers

    have = set(_toks(normalize_numbers(answer)))
    need = [t for t in _toks(normalize_numbers(reference)) if t not in _STOP and len(t) > 1]
    return bool(need) and all(t in have for t in need)


# ------------------------------------------------------------------ systems

class RagBaseline:
    """Plain vector search: title-prefixed ~300-token chunks, MiniLM, top-k cosine."""

    def __init__(self, path: str, embedder, chunk_tokens: int = RAG_CHUNK_TOKENS, k: int = RAG_K):
        from .prompt import count_tokens
        from .tree import tree_from_file

        tree = tree_from_file(path)
        self.k, self.embedder = k, embedder
        self.chunks: List[Tuple[str, str, str]] = []   # (id, path, text)
        for ch in tree.chapters():
            buf, used, first = [], 0, None
            for p in tree.passages(ch.id):
                t = count_tokens(p.text)
                if buf and used + t > chunk_tokens:
                    self.chunks.append((first, " / ".join(tree.path(first)[:2]), " ".join(buf)))
                    buf, used = [], 0
                if not buf:
                    first = p.id
                buf.append(p.text)
                used += t
            if buf:
                self.chunks.append((first, " / ".join(tree.path(first)[:2]), " ".join(buf)))
        texts = [f"{path}: {text}" for _, path, text in self.chunks]
        embedder.fit(texts)
        # Chunk vectors are cached on disk (k=5 and k=1 runs share them).
        import hashlib

        key = hashlib.sha256(("\n".join(texts) + getattr(embedder, "model_name", type(embedder).__name__))
                             .encode()).hexdigest()[:16]
        cache = ROOT / "data" / "longbench" / "cache" / f"rag-{key}.npy"
        if cache.exists():
            vecs = np.load(cache)
        else:
            vecs = np.asarray(embedder.encode(texts), dtype=np.float32)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, vecs)
        self.vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)

    def ask(self, question: str):
        from .evidence import EvidenceItem, EvidencePack

        q = np.asarray(self.embedder.encode_query(question), dtype=np.float32).reshape(-1)
        q = q / max(np.linalg.norm(q), 1e-9)
        top = np.argsort(-(self.vecs @ q))[: self.k]
        items = [EvidenceItem(f"E{i + 1}", self.chunks[j][0], "retrieved", self.chunks[j][1], self.chunks[j][2],
                              float(self.vecs[j] @ q), 0, 0) for i, j in enumerate(top)]
        return EvidencePack(question, "unknown", items=items), None


def make_system(name: str, doc: str, embedder) -> Callable[[str], Tuple[object, Optional[str]]]:
    """question -> (EvidencePack, confidence level or None)."""
    if name == "rule":
        from .pipeline import DocumentQA

        qa = DocumentQA.from_file(doc, embedder=embedder)

        def ask(q):
            r = qa.ask(q)
            return r.evidence, getattr(r.trace.confidence, "level", None)
        return ask
    if name == "v3":
        from .gnn.spec_qa import SpecQA

        qa = SpecQA.from_file(doc, str(GNN_MODEL), embedder=embedder)
        return lambda q: (qa.ask(q).evidence, None)
    if name == "rag":
        return RagBaseline(doc, embedder).ask
    if name == "rag1":   # same evidence budget as our pipeline (~300 tokens)
        return RagBaseline(doc, embedder, k=1).ask
    raise ValueError(name)


FULL_SYSTEM = ("You answer questions about the document below using only its text. Answer briefly, quoting "
               "the relevant words. If the document does not answer the question, set insufficient to true and "
               "answer \"The document does not say.\"")
FULL_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}, "insufficient": {"type": "boolean"}},
               "required": ["answer", "insufficient"], "additionalProperties": False}


# ---------------------------------------------------------------------- run

def run(family: str, tier: str, system: str, llm: Optional[str] = None, judge_spec: Optional[str] = None,
        limit: Optional[int] = None, kinds: Optional[List[str]] = None, tag: str = "") -> Dict:
    from .answer import answer_question, load_provider
    from .answer_eval import answer_f1, judge
    from .embeddings import load_embedder
    from .evaluation import load_cases

    cases = load_cases(str(EVAL / f"{family}_{tier}.jsonl"))
    if kinds:
        cases = [c for c in cases if c.tags.get("kind") in kinds]
    if limit:
        cases = _spread(cases, limit)
    suffix = (f"_{llm.replace(':', '-')}" if llm else "") + (f"_{tag}" if tag else "")
    out = RUNS / f"long_{family}_{tier}_{system}{suffix}.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [c for c in cases if c.id not in done]
    provider = load_provider(llm) if llm else None
    judge_provider = load_provider(judge_spec) if judge_spec else None
    build_seconds = None
    if todo:
        doc = todo[0].document
        started = time.perf_counter()
        if system in ("full", "coa"):
            if provider is None:
                raise SystemExit(f"--system {system} needs --llm")
            full_text = Path(doc).read_text()
            ask = None
        elif system in ("workers", "workers_nogate", "workers_certify"):
            if provider is None:
                raise SystemExit("--system workers needs --llm")
            from .upload import upload
            from .workers.worker import SentenceIndex

            tree = upload(doc, llm=llm)          # enriched once, cached by file hash
            if not tree.root.metadata.get("enrichment", {}).get("complete"):
                print("warning: enrichment incomplete (quota?); workers see partial summaries")
            index = SentenceIndex(tree)
            ask = None
            certify_kw: Dict = {}
            if system == "workers_certify":      # Complete or Silent: calibration + qualifier graph, built once
                from .certify import ReverseIndex, build_qualifier_graph
                from .certify.canaries import Calibration, calibrate

                cal_path = Path(doc).with_suffix(".calibration.v2.json")   # v2: with type canaries
                if cal_path.exists():
                    calibration = Calibration.load(str(cal_path))
                else:
                    library = None
                    if family == "contracts" or family.startswith("traps"):   # type canaries: CUAD clauses from contracts not in this room
                        from .certify.canaries import cuad_type_library

                        library = cuad_type_library(exclude_titles=[c.title for c in tree.chapters()])
                    calibration = calibrate(tree, index=index, n=300, seed=0, type_library=library)
                    calibration.save(str(cal_path))
                graph = build_qualifier_graph(tree)
                certify_kw = {"certify": True, "calibration": calibration, "qualifier_graph": graph,
                              "reverse_index": ReverseIndex(tree, graph)}
        else:
            embedder = load_embedder("st:all-MiniLM-L6-v2")
            ask = make_system(system, doc, embedder)
        build_seconds = round(time.perf_counter() - started, 1)
    from .answer import QuotaExhausted
    from .prompt import count_tokens

    for case in todo:
        row: Dict = {"id": case.id, "kind": case.tags.get("kind"), "answerable": case.answerable,
                     "depth": case.tags.get("depth"), "build_seconds": build_seconds}
        try:
            if system == "full":
                row.update(_full_answer(provider, full_text, case))
            elif system == "coa":
                from .baselines.chain_of_agents import ask as coa_ask

                t0 = time.perf_counter()
                res = coa_ask(full_text, case.question, provider, max_calls=None)
                row.update(answer=res.answer, abstained=res.abstained, calls=res.calls, chunks=res.chunks,
                           chunks_read=res.chunks_read, truncated=res.chunks_read < res.chunks,
                           input_tokens=res.input_tokens, output_tokens=res.output_tokens,
                           llm_ms=round((time.perf_counter() - t0) * 1000), pack_tokens=count_tokens(res.notes))
                if case.gold_evidence:
                    covs = [coverage(g, res.notes) for g in case.gold_evidence]
                    row["coverage"] = round(min(covs) if row["kind"] == "needle_2hop" else max(covs), 3)
                    row["hit"] = row["coverage"] >= HIT
                _score_traps(row, case, res.notes)
            elif system in ("workers", "workers_nogate", "workers_certify"):
                from .workers import ask as workers_ask

                t0 = time.perf_counter()
                res = workers_ask(tree, case.question, provider, index=index, use_gate=system != "workers_nogate",
                                  **certify_kw)
                pack = res.pack
                text = " ".join(i.text for i in pack.items)
                row.update(pack_tokens=count_tokens(text), chapters_searched=len(res.chapters_searched),
                           routed=bool(res.route.chapters), calls=res.calls, input_tokens=res.input_tokens,
                           output_tokens=res.output_tokens, llm_ms=round((time.perf_counter() - t0) * 1000),
                           answer=res.answer.answer, abstained=res.answer.abstained, flags=res.answer.flags,
                           rejected_ids=len(res.board.rejected), scope=res.scope, rounds=res.rounds_run,
                           opened_by=(res.plan.opened_by if res.plan else None))
                if case.gold_evidence:
                    covs = [coverage(g, text) for g in case.gold_evidence]
                    row["coverage"] = round(min(covs) if row["kind"] == "needle_2hop" else max(covs), 3)
                    row["hit"] = row["coverage"] >= HIT
                if case.tags.get("answers"):
                    row["answer_overlap"] = round(answer_overlap(case.tags["answers"], text), 3)
                if res.certificate is not None:
                    row["certificate"] = res.certificate.kind
                    row["certificate_text"] = res.certificate.text()
                    row["closure"] = res.closure.summary() if res.closure else None
                    row["superseded"] = len(res.ordering.superseded()) if res.ordering else None
                    row["verified"] = res.verification.summary() if res.verification else None
                    row["certified"] = res.certified
                _score_traps(row, case, text)
            else:
                t0 = time.perf_counter()
                pack, level = ask(case.question)
                row["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                text = " ".join(i.text for i in pack.items)
                row["pack_tokens"] = count_tokens(text)
                if case.gold_evidence:
                    covs = [coverage(g, text) for g in case.gold_evidence]
                    row["coverage"] = round(min(covs) if row["kind"] == "needle_2hop" else max(covs), 3)
                    row["hit"] = row["coverage"] >= HIT
                if case.tags.get("answers"):
                    row["answer_overlap"] = round(answer_overlap(case.tags["answers"], text), 3)
                if level is not None:
                    row["would_abstain"] = level == "insufficient"
                if provider is not None:
                    ans = answer_question(pack, provider, abstain=False)
                    row.update(answer=ans.answer, abstained=ans.abstained, flags=ans.flags,
                               input_tokens=ans.input_tokens, output_tokens=ans.output_tokens,
                               llm_ms=round(ans.latency_ms or 0), cost_usd=ans.cost_usd)
            if provider is not None:
                _score_answer(row, case, judge_provider, judge, answer_f1)
        except QuotaExhausted as exc:
            print(f"stopped: {exc}")
            break
        with out.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()] if out.exists() else []
    keep = {c.id for c in cases}
    rows = [r for r in rows if r["id"] in keep]
    summary = summarize(rows)
    summary["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
    summary["rows_file"] = str(out.relative_to(ROOT))
    return summary


def _spread(cases, limit: int):
    """At most ``limit`` cases, balanced over kinds (deterministic)."""
    by: Dict[str, List] = {}
    for c in cases:
        by.setdefault(c.tags.get("kind"), []).append(c)
    out, i = [], 0
    while len(out) < limit and any(i < len(v) for v in by.values()):
        for v in by.values():
            if i < len(v) and len(out) < limit:
                out.append(v[i])
        i += 1
    return out


def _full_answer(provider, full_text: str, case) -> Dict:
    user = f"Question: {case.question}\n\n<document>\n{full_text}\n</document>"
    t0 = time.perf_counter()
    c = provider.complete(FULL_SYSTEM, user, FULL_SCHEMA)
    try:
        data = json.loads(c.text)
    except json.JSONDecodeError:
        data = {"answer": c.text, "insufficient": False}
    return {"answer": data.get("answer", ""), "abstained": bool(data.get("insufficient")),
            "input_tokens": c.input_tokens, "output_tokens": c.output_tokens, "cost_usd": c.cost_usd,
            "llm_ms": round((time.perf_counter() - t0) * 1000)}


_AMOUNT = re.compile(r"\d[\d,]*(?:\.\d+)?\s*USD")
_MONEY_IN_ANSWER = re.compile(r"(?:US\$|USD|\$)\s*(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s*(?:USD|US dollars|dollars)",
                              re.I)


def _amounts(text: str) -> set:
    """Whole amounts in ``text`` as integers ("150,000 USD", "$150,000", "USD 150000")."""
    out = set()
    for m in _MONEY_IN_ANSWER.finditer(text):
        raw = (m.group(1) or m.group(2)).replace(",", "")
        try:
            out.add(int(float(raw)))
        except ValueError:
            continue
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


_NEGATED = re.compile(r"\b(does not|doesn't|do not|did not|no|not|none|neither|nor|without)\b[^.]{0,80}?"
                      r"\b(contain|include|have|has|provide|any)\b|\bcontains no\b|\bno\b[^.]{0,40}\bclause\b", re.I)


def _says_yes(answer: str) -> bool:
    """Leading yes / no word first; otherwise a negation anywhere near "contain" means no."""
    head = re.sub(r"^[\s*_>#\-\"']+", "", answer).lower()
    if re.match(r"^(yes)\b", head):
        return True
    if re.match(r"^(no)\b", head) or "does not say" in head or "not found" in head:
        return False
    return not _NEGATED.search(answer) and bool(re.search(r"\b(contain|include|has|provide)", head))


def _company_of(title: str) -> str:
    """The distinctive part of a CUAD room title: "License Agreement of WebmdHealthCorp
    (2005-10-01)" -> "webmdhealthcorp"."""
    m = re.search(r"\bof\s+([^()#]+)", title)
    name = (m.group(1) if m else title).strip().lower()
    return re.sub(r"[,.]?\s*(inc|corp|ltd|llc|co)\.?$", "", name).strip(" ,.")


def trap_correct(answer: str, case) -> bool:
    """A trap answer is correct when it states every amount of the reference (the cap and
    each exception; for an override chain, the final amount) and, for an unlinked
    carve-out, names fraud. Existence: yes / no as expected (an abstention counts as no)."""
    text = answer.lower()
    if case.tags.get("kind") == "room_existence":
        expected = case.tags.get("expected_units") or []
        if not expected:      # nothing in the room: a refusal or an explicit "none"
            return not re.search(r"\b(contract|agreement)s?\b.*\b(contain|include|ha[sv]e?)\b", text) or \
                bool(re.search(r"\b(none|no contract|does not say|not found)\b", text))
        return all(_company_of(t) in text for t in expected)
    if case.tags.get("kind") == "existence":
        said_yes = _says_yes(answer)
        return said_yes if case.tags.get("expected") == "yes" else not said_yes
    # every amount of the reference, as a whole number ("50,000" is not inside "150,000")
    ok = _amounts(case.reference_answer) <= _amounts(answer)
    if case.tags.get("kind") == "unlinked":
        ok = ok and "fraud" in text
    return ok


def _score_traps(row: Dict, case, pack_text: str) -> None:
    """Trap metrics (family ``traps``): ``qualifier_recall`` = share of the required
    sentences present in the evidence; ``certificate_correct`` = the certificate kind the
    system gave equals the one the case expects (closure for a found trap, absence for a
    planted "no")."""
    tags = case.tags
    if not str(tags.get("family", "")).startswith("traps"):
        return
    required = tags.get("required") or []
    if required:   # exact (whitespace / case normalised): look-alikes differ only in an amount or name
        have = _norm(pack_text)
        row["qualifier_recall"] = round(sum(_norm(q) in have for q in required) / len(required), 3)
    row["pack_text"] = pack_text   # kept so the evidence can be re-scored without new model calls
    if "certificate" in row:
        row["certificate_correct"] = row["certificate"] == tags.get("certificate")


def _score_answer(row: Dict, case, judge_provider, judge, answer_f1) -> None:
    """``correct`` only where it can be decided: planted facts (the values), unanswerable
    questions (abstained), and judged answers. ``false_abstention`` for answerable ones."""
    answer = row.get("answer", "")
    for key in ("correct", "judge", "judge_score", "answer_f1", "false_abstention"):
        row.pop(key, None)
    is_trap = str(case.tags.get("family", "")).startswith("traps")
    if is_trap and row["kind"] in ("existence", "room_existence"):
        row["correct"] = trap_correct(answer, case) if not row.get("abstained") else not case.answerable
        if case.answerable:
            row["false_abstention"] = bool(row.get("abstained"))
        return
    if not case.answerable:
        row["correct"] = bool(row.get("abstained"))
        return
    row["false_abstention"] = bool(row.get("abstained"))
    if row.get("abstained"):
        if row["kind"] in ("needle", "needle_2hop") or judge_provider is not None or is_trap:
            row["correct"] = False
        if judge_provider is not None and row["kind"] in ("cuad", "narrativeqa"):
            row["judge_score"] = 0.0
        return
    if str(case.tags.get("family", "")).startswith("traps"):
        row["correct"] = trap_correct(answer, case)
    elif row["kind"] in ("needle", "needle_2hop"):
        row["correct"] = contains_answer(answer, case.reference_answer)
    if case.tags.get("answers"):
        row["answer_f1"] = round(max(answer_f1(answer, a) for a in case.tags["answers"]), 3)
    if judge_provider is not None and row["kind"] in ("cuad", "narrativeqa"):
        ref = " / ".join(case.tags.get("answers") or case.gold_evidence[:2])
        verdict = judge(judge_provider, case.question, answer, ref)
        row["judge"] = verdict["verdict"]
        row["correct"] = verdict["verdict"] == "correct"
        row["judge_score"] = {"correct": 1.0, "partially_correct": 0.5}.get(verdict["verdict"], 0.0)


def summarize(rows: List[Dict]) -> Dict:
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    out: Dict = {"questions": len(rows), "build_seconds": next((r["build_seconds"] for r in rows
                                                                 if r.get("build_seconds") is not None), None)}
    by = Counter(r["kind"] for r in rows)
    out["by_kind"] = {}
    for kind in sorted(by):
        rs = [r for r in rows if r["kind"] == kind]
        k = {"n": len(rs)}
        if any("hit" in r for r in rs):
            k["evidence_hit"] = mean([float(r["hit"]) for r in rs if "hit" in r])
            k["coverage"] = mean([r.get("coverage") for r in rs])
        if any("answer_overlap" in r for r in rs):
            k["answer_overlap"] = mean([r.get("answer_overlap") for r in rs])
        if any("would_abstain" in r for r in rs):
            k["would_abstain"] = mean([float(r["would_abstain"]) for r in rs if "would_abstain" in r])
        if any("correct" in r for r in rs):
            k["correct"] = mean([float(r["correct"]) for r in rs if "correct" in r])
        if any("answer_f1" in r for r in rs):
            k["answer_f1"] = mean([r.get("answer_f1") for r in rs])
        if any("judge_score" in r for r in rs):
            k["judge_score"] = mean([r.get("judge_score") for r in rs])
        if any("false_abstention" in r for r in rs):
            k["false_abstention"] = mean([float(r["false_abstention"]) for r in rs if "false_abstention" in r])
        if any("qualifier_recall" in r for r in rs):
            k["qualifier_recall"] = mean([r.get("qualifier_recall") for r in rs])
        if any("certificate" in r for r in rs):
            k["certificates"] = dict(Counter(r["certificate"] for r in rs if r.get("certificate")))
            k["certificate_correct"] = mean([float(r["certificate_correct"]) for r in rs if "certificate_correct" in r])
        out["by_kind"][kind] = k
    out["pack_tokens"] = mean([r.get("pack_tokens") for r in rows])
    out["retrieval_ms"] = mean([r.get("retrieval_ms") for r in rows])
    if any("input_tokens" in r for r in rows):
        out["llm_input_tokens"] = mean([r.get("input_tokens") for r in rows])
        out["llm_ms"] = mean([r.get("llm_ms") for r in rows])
    if any("calls" in r for r in rows):
        out["calls_per_question"] = mean([r.get("calls") for r in rows])
        out["chapters_searched"] = mean([r.get("chapters_searched") for r in rows])
        out["routed_share"] = mean([float(r.get("routed", False)) for r in rows])
    return out


def rescore(family: str, tier: str, system: str, llm: str, judge_spec: Optional[str] = None, tag: str = "") -> Dict:
    """Grade saved answers again with the current rules (and optionally an LLM judge for
    clause / book answers); no new answer calls. Judge verdicts already saved are kept."""
    from .answer import QuotaExhausted, load_provider
    from .answer_eval import answer_f1, judge
    from .evaluation import load_cases

    out = RUNS / f"long_{family}_{tier}_{system}_{llm.replace(':', '-')}{'_' + tag if tag else ''}.jsonl"
    cases = {c.id: c for c in load_cases(str(EVAL / f"{family}_{tier}.jsonl"))}
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    judge_provider = load_provider(judge_spec) if judge_spec else None
    for row in rows:
        old = {k: row.get(k) for k in ("judge", "judge_score", "correct")}
        try:
            already = old["judge"] is not None
            _score_answer(row, cases[row["id"]], None if already else judge_provider, judge, answer_f1)
            if row.get("pack_text") is not None:
                _score_traps(row, cases[row["id"]], row["pack_text"])
            if already:
                row.update({k: v for k, v in old.items() if v is not None})
        except QuotaExhausted as exc:
            row.update({k: v for k, v in old.items() if v is not None})
            print(f"stopped: {exc}")
            judge_provider = None
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    return summarize(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", required=True, choices=["contracts", "books", "traps", "traps2"])
    parser.add_argument("--tier", required=True, choices=["S", "M", "L"])
    parser.add_argument("--system", choices=["rule", "v3", "rag", "rag1", "full", "workers", "workers_nogate",
                                             "workers_certify", "coa"])
    parser.add_argument("--llm")
    parser.add_argument("--judge")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--kinds", help="comma-separated question kinds")
    parser.add_argument("--rescore", action="store_true", help="grade saved answers again (no answer calls)")
    parser.add_argument("--tag", default="", help="name a variant run (its own rows file)")
    parser.add_argument("--gate-readings", action="store_true", help="gate recall per question (no model)")
    parser.add_argument("--dense", action="store_true", help="gate readings: add the dense signal")
    args = parser.parse_args()
    if args.gate_readings:
        print(json.dumps(gate_readings(args.family, args.tier, args.dense), indent=1))
        return
    if args.rescore:
        print(json.dumps(rescore(args.family, args.tier, args.system, args.llm, args.judge, args.tag), indent=1))
        return
    summary = run(args.family, args.tier, args.system, args.llm, args.judge, args.limit,
                  args.kinds.split(",") if args.kinds else None, args.tag)
    print(json.dumps(summary, indent=1))



# ------------------------------------------------------------ gate readings

def gate_readings(family: str, tier: str, dense: bool = False, k: Optional[int] = None) -> Dict:
    """No model calls. For each question with gold evidence: the chapter that holds the
    answer (the chapter of the best-matching sentence for the gold text), whether routing
    names it, and whether the gate opens it in round 1 / by round 2 (TOC screen off: it
    needs a model).

    Measured separately when the question names its unit and when it does not."""
    from .evaluation import load_cases
    from .tree import tree_from_file
    from .workers.gate import Gate
    from .workers.orchestrator import NAMED_AS_FILTER, SCOPES
    from .workers.routing import route, scope, words
    from .workers.worker import SentenceIndex

    cases = [c for c in load_cases(str(EVAL / f"{family}_{tier}.jsonl")) if c.gold_evidence]
    started = time.perf_counter()
    tree = tree_from_file(cases[0].document)
    index = SentenceIndex(tree)
    vectors, embedder = None, None
    if dense:
        from .embeddings import load_embedder

        embedder = load_embedder("st:all-MiniLM-L6-v2")
        passages = tree.passages()
        embedder.fit([p.text for p in passages])
        mat = np.asarray(embedder.encode([p.text for p in passages]), dtype=np.float32)
        mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
        vectors = {p.id: mat[i] for i, p in enumerate(passages)}
    gate = Gate(tree, index, vectors, embedder)
    build = round(time.perf_counter() - started, 1)
    rows = []
    import re as _re

    def unit_chapters(title):
        base = lambda t: _re.sub(r"\s*\(\d+/\d+\)$", "", _re.sub(r"^Book:\s*", "", t))
        ids = {ch.id for ch in tree.chapters() if base(ch.title) == title and ch.role == "body"}
        for part in tree.parts():                     # a book: all its chapters
            if base(part.title) == title:
                ids |= set(part.children)
        return ids

    for c in cases:
        # The answer's chapter, looked up inside the unit the question is about (a clause
        # like "governed by the laws of New York" occurs in many contracts).
        units = [c.tags.get("unit"), c.tags.get("second_unit")]
        allowed = set().union(*[unit_chapters(u) for u in units if u]) or None
        exclude = set(index.chapter_names) - allowed if allowed else set()
        gold_chapters = {index.best_chapter(g, exclude=exclude) for g in c.gold_evidence} - {None}
        if not gold_chapters:        # the unit (e.g. a one-book library's title page) holds no gold: whole document
            gold_chapters = {index.best_chapter(g) for g in c.gold_evidence} - {None}
        if not gold_chapters:
            continue                 # gold text not in the document: never count it as found
        r = route(tree, c.question)
        cfg = SCOPES[scope(c.question)]
        named, within = r.chapters, None
        if len(named) > NAMED_AS_FILTER:
            within, named = named, []
        plan = gate.plan(c.question, r.content or words(c.question), named, k=k or cfg["k"], within=within)
        r1 = set(plan.rounds[0]) if plan.rounds else set()
        r12 = r1 | (set(plan.rounds[1]) if len(plan.rounds) > 1 else set())
        rows.append({"id": c.id, "kind": c.tags.get("kind"), "named": bool(r.chapters),
                     "route_ok": bool(gold_chapters & set(r.chapters)) if r.chapters else None,
                     "round1": gold_chapters <= r1, "round12": gold_chapters <= r12,
                     "opened1": len(r1), "opened12": len(r12), "gold_chapters": len(gold_chapters)})

    def agg(rs):
        if not rs:
            return None
        m = lambda key: round(float(np.mean([float(x[key]) for x in rs if x[key] is not None])), 3) \
            if any(x[key] is not None for x in rs) else None
        return {"n": len(rs), "route_ok": m("route_ok"), "round1_recall": m("round1"),
                "round12_recall": m("round12"), "opened_round1": m("opened1"), "opened_by_round2": m("opened12")}

    return {"chapters": len(tree.chapters()), "build_seconds": build, "dense": dense,
            "named": agg([x for x in rows if x["named"]]), "unnamed": agg([x for x in rows if not x["named"]]),
            "by_kind": {k: agg([x for x in rows if x["kind"] == k]) for k in sorted({x["kind"] for x in rows})}}


if __name__ == "__main__":
    main()
