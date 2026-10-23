"""Question -> relevance check -> chapter workers -> central board -> evidence pack -> answer
(WORKFLOW B).

1. **Route** (``routing``): the chapters the question names, its content words, and its
   scope (single / multi / global).
2. **Choose chapters** (``gate``):
   - named chapters are opened, plus every relevance signal's top picks (any signal opens)
   - a named unit with many chapters (a book) limits the gate to its chapters
   - ``use_gate=False`` searches every chapter (the "no gate" system of the benchmark)
3. **Run one worker per chapter**, in parallel (threads: the work is waiting on the model).
4. **Follow references** the workers asked for, at most ``follow_hops`` per round.
5. **Widen before "not found"** (R1): if nothing was accepted,
   - round 2 opens the next chapters
   - round 3 rewrites the question in the document's wording and searches again
6. **Add definitions** of terms used in the accepted passages.
7. **Whole-document questions** (scope ``global``, R2): more chapters are opened (8 per
   signal), each worker also writes a short partial answer for its chapter, and the final
   answer receives those as orientation notes. Only quotes are evidence and can be cited.
8. **Answer** from an evidence pack sized by scope (R9):

   | Scope | Evidence |
   |---|---|
   | single | ~600 tokens |
   | multi | ~1,000 tokens |
   | global | ~2,000 tokens |

   - nothing accepted: "The document does not say", without an answer call
   - otherwise: the Phase 10 answer step (citations checked; the model may still say the
     evidence is insufficient)

Every model call is counted (gate, workers, rewrite, answer); workers stop at the budget.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..answer import ABSTAIN_TEXT, Answer, answer_question
from .board import Board
from .gate import Gate, GatePlan
from .routing import Route, route, scope, words
from .worker import ChapterWorker, SentenceIndex, WorkerReport

# Per scope: chapters per gate signal, evidence items, evidence tokens.
SCOPES: Dict[str, Dict[str, int]] = {
    "single": {"k": 3, "items": 8, "tokens": 600},
    "multi": {"k": 4, "items": 12, "tokens": 1000},
    "global": {"k": 8, "items": 24, "tokens": 2000},
}
NAMED_AS_FILTER = 6   # a named unit with more chapters than this narrows the gate instead


@dataclass
class WorkerResult:
    question: str
    answer: Answer
    board: Board
    route: Route
    scope: str = "single"
    reports: List[WorkerReport] = field(default_factory=list)
    chapters_searched: List[str] = field(default_factory=list)
    rounds_run: int = 0
    plan: Optional[GatePlan] = None
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    @property
    def pack(self):
        cfg = SCOPES[self.scope]
        return self.board.to_pack(cfg["items"], cfg["tokens"])


def ask(tree, question: str, provider, answer_provider=None, chapters: Optional[List[str]] = None,
        index: Optional[SentenceIndex] = None, gate: Optional[Gate] = None, use_gate: bool = True,
        use_toc: bool = True, widen: bool = True, max_calls: int = 40, max_steps: int = 4, parallel: int = 4,
        follow_hops: int = 2) -> WorkerResult:
    started = time.perf_counter()
    index = index or SentenceIndex(tree)
    r = route(tree, question)
    sc = scope(question)
    cfg = SCOPES[sc]
    terms = r.content or words(question)
    board = Board(tree, question, max_calls)
    board.index = index
    result = WorkerResult(question, Answer(question, ABSTAIN_TEXT), board, r, sc)

    # ---- which chapters, in which rounds
    within: Optional[List[str]] = None
    if chapters:
        rounds = [list(chapters)]
    elif not use_gate:
        rounds = [r.chapters or [c.id for c in tree.chapters() if c.role != "contents"]]
    else:
        gate = gate or Gate(tree, index)
        named = r.chapters
        if len(named) > NAMED_AS_FILTER:       # e.g. a whole book: search within it
            within, named = named, []
        plan = gate.plan(question, terms, named, k=cfg["k"], provider=provider, use_toc=use_toc,
                         board=board, within=within)
        result.plan = plan
        rounds = plan.rounds if widen else plan.rounds[:1]

    search_terms = {"now": terms}

    def work(chapter: str) -> WorkerReport:
        board.workers[chapter] = "searching"
        try:
            report = ChapterWorker(tree, chapter, provider, board, index, max_steps).run(question,
                                                                                          search_terms["now"])
        except Exception as exc:  # a failed worker must not sink the question
            from ..answer import QuotaExhausted

            if isinstance(exc, QuotaExhausted):
                raise
            report = WorkerReport(chapter, error=f"{type(exc).__name__}: {exc}")
        board.workers[chapter] = "done"
        return report

    done: List[str] = []

    def run_round(batch: List[str]) -> None:
        for _ in range(follow_hops + 1):
            batch = [c for c in dict.fromkeys(batch) if c not in done]
            if not batch or not board.has_budget():
                return
            with ThreadPoolExecutor(max_workers=max(1, min(parallel, len(batch)))) as pool:
                result.reports.extend(pool.map(work, batch))
            done.extend(batch)
            batch = [c for c in board.follow if c not in done]

    for k, batch in enumerate(rounds):
        if k and board.status == "found":
            break
        run_round(batch)
        result.rounds_run += 1
    if (use_gate and widen and gate is not None and not chapters and board.status != "found"
            and board.has_budget() and provider is not None):
        extra, rewritten = gate.widen_by_rewrite(question, provider, done, cfg["k"], board, within)
        if extra:
            for c in extra:
                result.plan.opened_by.setdefault(c, []).append("rewrite")
            search_terms["now"] = rewritten + [t for t in terms if t not in rewritten]
            run_round(extra)
            result.rounds_run += 1
    result.chapters_searched = done
    board.add_definitions()

    if board.status == "found":
        pack = board.to_pack(cfg["items"], cfg["tokens"])
        notes = board.notes() if sc == "global" else ""
        answer = answer_question(pack, answer_provider or provider, abstain=False, notes=notes)
        board.calls += 1
        board.input_tokens += answer.input_tokens or 0
        board.output_tokens += answer.output_tokens or 0
        result.answer = answer
    else:
        result.answer.abstained = True
        result.answer.flags = ["no_evidence"]
    result.calls, result.input_tokens, result.output_tokens = board.calls, board.input_tokens, board.output_tokens
    result.seconds = round(time.perf_counter() - started, 2)
    return result
