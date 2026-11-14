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

   Widening is kept cheap: a widening round opens only chapters whose best sentence scores
   at least ``WIDEN_MIN_SHARE`` of the best chapter's (chapters with no real word match are
   skipped without a call), and its workers get ``WIDEN_STEPS`` steps instead of
   ``max_steps``.
6. **Bridge the second hop** (board, no model): sentences elsewhere that continue a rare
   phrase of an accepted sentence ("rate card RC-8985") join the evidence.
7. **Add definitions** of terms used in the accepted passages.
8. **Whole-document questions** (scope ``global``, R2): more chapters are opened (8 per
   signal), each worker also writes a short partial answer for its chapter, and the final
   answer receives those as orientation notes. Only quotes are evidence and can be cited.
9. **Answer** from an evidence pack sized by scope (R9):

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
from ..certify.plan import DEFAULT_ALPHA, certified_alpha, reading_plan
from .board import Board

CLOSURE_NOTE = ("Evidence roles: items marked overrides, amends or qualifies change the main clause. State the main "
                "clause first, then what qualifies or overrides it, and say which text was replaced (superseded).")
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
WIDEN_MIN_SHARE = 0.5  # widening opens a chapter only if its best sentence scores this share of the best
WIDEN_STEPS = 2        # steps per worker in widening rounds


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
    reading: object = None            # certify.plan.ReadingPlan (when a calibration was given)
    certified: Optional[float] = None  # highest alpha whose calibrated chapters were all opened
    closure: object = None            # certify.closure.Closure (certify=True, answer found)
    ordering: object = None           # certify.precedence.Ordering
    verification: object = None       # certify.verify.Verification
    certificate: object = None        # certify.certificate.Certificate
    opened: List[str] = field(default_factory=list)   # chapters a worker actually read (>= 1 parsed step, no error)
    answer_pack: object = None        # the evidence pack the answer (and verifier) actually saw
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    @property
    def pack(self):
        if self.answer_pack is not None:
            return self.answer_pack
        cfg = SCOPES[self.scope]
        return self.board.to_pack(cfg["items"], cfg["tokens"])


def ask(tree, question: str, provider, answer_provider=None, chapters: Optional[List[str]] = None,
        index: Optional[SentenceIndex] = None, gate: Optional[Gate] = None, use_gate: bool = True,
        use_toc: bool = True, widen: bool = True, max_calls: int = 40, max_steps: int = 4, parallel: int = 4,
        follow_hops: int = 2, calibration=None, alpha: float = DEFAULT_ALPHA, certify: bool = False,
        qualifier_graph=None, reverse_index=None) -> WorkerResult:
    """``calibration``: this document's canary calibration (certify.canaries). When given,
    the gate opens the calibrated number of chapters for ``alpha`` instead of the scope's
    fixed k, and the result records the level actually reached (``certified``).
    ``certify``: Complete or Silent. An answer found is expanded to its closure (forward
    links, reverse index, sweep judged by the qualifier worker), ordered by precedence,
    verified sentence by sentence, and given a closure certificate; "not found" is given an
    absence certificate from the calibration. ``qualifier_graph`` / ``reverse_index`` may be
    passed precomputed (``certify.build_qualifier_graph``, ``certify.ReverseIndex``)."""
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
        reading = reading_plan(calibration, question, named, alpha, cfg["k"])
        result.reading = reading
        k = reading.k if reading.basis in ("unit", "room") else cfg["k"]
        if len(named) > NAMED_AS_FILTER:       # e.g. a whole book: search within it
            within, named = named, []
        plan = gate.plan(question, terms, named, k=k, provider=provider, use_toc=use_toc,
                         board=board, within=within)
        if reading.basis == "threshold":     # "which contracts contain X?": every chapter above the threshold
            from ..certify.plan import above_threshold

            plan.rounds = [above_threshold(index.best_per_chapter(terms), reading.threshold, gate.chapters)]
            for c in plan.rounds[0]:
                plan.opened_by.setdefault(c, []).append("threshold")
        result.plan = plan
        rounds = plan.rounds if widen else plan.rounds[:1]

    search_terms = {"now": terms}

    steps = {"now": max_steps}

    def work(chapter: str) -> WorkerReport:
        board.workers[chapter] = "searching"
        try:
            report = ChapterWorker(tree, chapter, provider, board, index, steps["now"]).run(question,
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
                reports = list(pool.map(work, batch))
            result.reports.extend(reports)
            done.extend(batch)
            # "opened" (what a certificate may rest on): the worker made at least one parsed
            # step and did not stop on the budget or an error
            result.opened.extend(r.chapter for r in reports if r.steps and not r.error)
            batch = [c for c in board.follow if c not in done]

    def worth_opening(batch: List[str], query: List[str]) -> List[str]:
        best = index.best_per_chapter(query)
        top = max(best.values(), default=0.0)
        named = set(r.chapters)
        return [c for c in batch if c in named or (top > 0 and best.get(c, 0.0) >= WIDEN_MIN_SHARE * top)]

    for k, batch in enumerate(rounds):
        if k and board.status == "found":
            break
        if k:
            batch = worth_opening(batch, terms)
            steps["now"] = min(max_steps, WIDEN_STEPS)
            if not batch:
                continue
        run_round(batch)
        result.rounds_run += 1
    if (use_gate and widen and gate is not None and not chapters and board.status != "found"
            and board.has_budget() and provider is not None):
        extra, rewritten = gate.widen_by_rewrite(question, provider, done, cfg["k"], board, within)
        extra = worth_opening(extra, rewritten + list(terms)) if extra else extra
        steps["now"] = min(max_steps, WIDEN_STEPS)
        if extra:
            for c in extra:
                result.plan.opened_by.setdefault(c, []).append("rewrite")
            search_terms["now"] = rewritten + [t for t in terms if t not in rewritten]
            run_round(extra)
            result.rounds_run += 1
    result.chapters_searched = done
    if result.reading is not None and gate is not None:
        allowed = [c for c in gate.chapters if within is None or c in within]
        ranking = [c for c in gate.lexical(terms) if c in allowed]
        ranking += [c for c in allowed if c not in set(ranking)]   # unmatched chapters last, as for the canaries
        result.certified = certified_alpha(calibration, result.reading, result.opened, ranking,
                                           scores=index.best_per_chapter(terms))
    if board.status == "found":
        board.add_bridges()
    board.add_definitions()

    if board.status == "found":
        notes = board.notes() if sc == "global" else ""
        if certify:
            from ..certify import ReverseIndex, build_qualifier_graph
            from ..certify.closure import apply_to_board, expand
            from ..certify.precedence import order

            qualifier_graph = qualifier_graph or build_qualifier_graph(tree)
            reverse_index = reverse_index or ReverseIndex(tree, qualifier_graph)
            seeds = [(f.passage, list(f.sentence_ids)) for f in board.accepted if f.role == "answer"]
            result.closure = expand(tree, qualifier_graph, reverse_index, seeds, provider=provider, board=board,
                                    index=index)
            apply_to_board(board, result.closure)
            result.ordering = order(tree, result.closure)
            if result.closure.items:
                notes = (notes + "\n" if notes else "") + CLOSURE_NOTE
                # a certified answer must carry its whole closure: size the pack for it
                needed = sum(1 for f in board.accepted if f.role not in ("bridge", "definition"))
                cfg = {"items": max(SCOPES[sc]["items"], needed + 4),
                       "tokens": max(SCOPES[sc]["tokens"], SCOPES["global"]["tokens"])}
        pack = board.to_pack(cfg["items"], cfg["tokens"])
        result.answer_pack = pack
        answer = answer_question(pack, answer_provider or provider, abstain=False, notes=notes)
        board.calls += 1
        board.input_tokens += answer.input_tokens or 0
        board.output_tokens += answer.output_tokens or 0
        result.answer = answer
        if certify:
            from ..certify.certificate import closure_certificate
            from ..certify.verify import verify

            if not answer.abstained and answer.answer:
                result.verification = verify(answer.answer, pack, answer_provider or provider, board)
            in_pack = {i.node_id for i in pack.items}
            missing = [i for i in result.closure.items
                       if i.relation not in ("definition", "reference") and i.passage not in in_pack]
            result.certificate = closure_certificate(
                tree, result.closure, result.ordering, result.opened, result.verification,
                abstained=answer.abstained, abstain_flags=answer.flags, missing_from_pack=len(missing))
    else:
        result.answer.abstained = True
        result.answer.flags = ["no_evidence"]
        if certify:
            from ..certify.certificate import absence_certificate

            failed = [r.chapter for r in result.reports if r.error]
            result.certificate = absence_certificate(tree, calibration, result.reading, result.certified,
                                                     result.opened, budget_reached=board.status == "budget_reached",
                                                     failed=failed)
    result.calls, result.input_tokens, result.output_tokens = board.calls, board.input_tokens, board.output_tokens
    result.seconds = round(time.perf_counter() - started, 2)
    return result
