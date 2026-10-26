"""Service layer for the demo app (and later a web API): one place the UI calls.

    svc = DocumentService()
    path = svc.save_upload("lease.pdf", data)
    tree = svc.process(path, llm="gemini:gemini-3.5-flash-lite", progress=print)
    session = svc.open(tree)
    reply = svc.ask(session, "Which law governs it?", history=[...], llm=...)
    png = svc.page_image(reply.citations[0])      # the PDF page with the quote highlighted

- **Library:** documents already processed (the trees cached by ``upload``).
- **Processing:** tree → metadata → key facts and summaries, reporting progress. It can
  skip summaries for speed; workers then navigate by titles and search alone.
- **Asking:**
  - a follow-up ("and the renewal term?") is first rewritten into a standalone question
    using the conversation
  - then the relevance gate → chapter workers → board → cited answer
  - **offline (no model):** the gate still picks the chapters; their best sentences are
    the evidence; the answer is the top extract
- **Citations:** each carries its page, chapter / section path and exact quote. The quote
  is highlighted on the rendered PDF page, or in the passage text for other formats.
- **Trace:** which chapters were opened and why, rounds, model calls, tokens, cost, time.
- **Feedback:** appended to ``data/feedback.jsonl`` (input for the learning loop).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .tree import DocTree

ROOT = Path(__file__).resolve().parents[2]
UPLOADS = ROOT / "data" / "uploads"
FEEDBACK = ROOT / "data" / "feedback.jsonl"
FOLLOW_UP = re.compile(r"^(and|but|also|what about|how about|same|then)\b|\b(it|its|they|them|their|this|that|"
                       r"these|those|he|she|him|her|the same)\b", re.I)
REWRITE_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}},
                  "required": ["question"], "additionalProperties": False}
REWRITE_SYSTEM = ("Rewrite the user's latest question so it can be understood without the conversation: replace "
                  "pronouns and references (it, that clause, the same agreement, what about ...) with what they "
                  "refer to. Keep it a question; do not answer it. If it is already standalone, return it unchanged.")


@dataclass
class Citation:
    label: str
    section: str
    quote: str
    passage_id: str
    page: Optional[int] = None
    source: Optional[str] = None
    role: str = "answer"


@dataclass
class Reply:
    asked: str
    question: str                         # standalone form actually answered
    answer: str
    abstained: bool
    citations: List[Citation] = field(default_factory=list)
    trace: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Session:
    tree: DocTree
    index: object
    gate: object
    cache: Optional[str] = None


class DocumentService:
    def __init__(self, cache_dir: Optional[Path] = None, uploads: Path = UPLOADS, feedback_file: Path = FEEDBACK):
        from .upload import DEFAULT_CACHE

        self.cache_dir = Path(cache_dir or DEFAULT_CACHE)
        self.uploads, self.feedback_file = Path(uploads), Path(feedback_file)
        self._providers: Dict[str, object] = {}

    # ------------------------------------------------------------- library
    def library(self) -> List[Dict]:
        out = []
        for path in sorted(self.cache_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                tree = DocTree.load(str(path))
            except Exception:
                continue
            meta = tree.root.metadata
            out.append({"title": tree.title, "kind": tree.kind, "source": tree.source, "cache": str(path),
                        "pages": (tree.root.pages or [None, None])[1], "tokens": tree.root.tokens,
                        "chapters": len(tree.chapters()),
                        "enriched": bool(meta.get("enrichment", {}).get("complete")),
                        "fields": {k: v["value"] for k, v in meta.get("fields", {}).items()}})
        return out

    def save_upload(self, filename: str, data: bytes) -> Path:
        self.uploads.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^\w.\- ]", "_", Path(filename).name)
        path = self.uploads / safe
        path.write_bytes(data)
        return path

    def provider(self, llm: Optional[str]):
        """Provider for ``llm`` ("offline" / None = no model). Reused across questions."""
        if not llm or llm == "offline":
            return None
        if llm not in self._providers:
            from .answer import load_provider

            self._providers[llm] = load_provider(llm)
        return self._providers[llm]

    # ---------------------------------------------------------- processing
    def process(self, path: Path, llm: Optional[str] = None, summaries: bool = True,
                progress: Callable[[str, float], None] = lambda msg, frac: None) -> DocTree:
        from .enrich import Enricher
        from .metadata import extract_metadata
        from .upload import cache_path

        progress("Reading the document and building its tree", 0.05)
        target = cache_path(str(path), self.cache_dir)
        if target.exists():
            tree = DocTree.load(str(target))
        else:
            from .tree import tree_from_file

            tree = tree_from_file(str(path))
            tree.save(str(target))
        provider = self.provider(llm)
        if "fields" not in tree.root.metadata:
            progress("Finding title, author / parties, dates", 0.15)
            extract_metadata(tree, provider)
            tree.save(str(target))
        if summaries:
            subs = [n for n in tree.walk() if n.kind == "subsection"]

            def save(t: DocTree) -> None:
                t.save(str(target))
                done = sum(1 for n in subs if n.metadata.get("facts_done"))
                progress(f"Key facts and summaries: {done}/{len(subs)} parts", 0.2 + 0.75 * done / max(1, len(subs)))

            Enricher(provider).run(tree, save=save)
        tree.root.metadata["cache"] = str(target)
        tree.save(str(target))
        progress("Ready", 1.0)
        return tree

    def open(self, tree_or_cache) -> Session:
        from .workers.gate import Gate
        from .workers.worker import SentenceIndex

        tree = DocTree.load(tree_or_cache) if isinstance(tree_or_cache, (str, Path)) else tree_or_cache
        index = SentenceIndex(tree)
        return Session(tree, index, Gate(tree, index), tree.root.metadata.get("cache"))

    # ---------------------------------------------------------------- asking
    def standalone(self, question: str, history: List[Dict], provider) -> str:
        """Rewrite a follow-up using the conversation (one model call); else unchanged."""
        if not history or provider is None or not FOLLOW_UP.search(question.strip()):
            return question
        turns = "\n".join(f"{h['role']}: {h['content'][:400]}" for h in history[-4:])
        c = provider.complete(REWRITE_SYSTEM, f"Conversation:\n{turns}\n\nLatest question: {question}", REWRITE_SCHEMA)
        try:
            return json.loads(c.text).get("question") or question
        except (json.JSONDecodeError, AttributeError):
            return question

    def ask(self, session: Session, question: str, history: Optional[List[Dict]] = None,
            llm: Optional[str] = None) -> Reply:
        started = time.perf_counter()
        provider = self.provider(llm)
        standalone = self.standalone(question, history or [], provider)
        if provider is None:
            return self._offline(session, question, standalone, started)
        from .workers import ask as workers_ask

        res = workers_ask(session.tree, standalone, provider, index=session.index, gate=session.gate)
        pack = res.pack
        by_label = {i.label: i for i in pack.items}
        cited = res.answer.cited or [i.label for i in pack.items[:1]]
        citations = [self._citation(session, by_label[l]) for l in cited if l in by_label]
        opened = []
        for c in res.chapters_searched:
            reasons = (res.plan.opened_by.get(c, []) if res.plan else []) or ["followed a reference"]
            opened.append({"chapter": session.tree.node(c).title, "why": reasons})
        trace = {"scope": res.scope, "chapters_total": len(session.tree.chapters()), "chapters_opened": opened,
                 "rounds": res.rounds_run, "model_calls": res.calls, "tokens": res.input_tokens + res.output_tokens,
                 "cost_usd": res.answer.cost_usd, "seconds": round(time.perf_counter() - started, 1),
                 "evidence_items": len(pack.items), "rejected_ids": len(res.board.rejected), "model": llm,
                 "rewritten": standalone != question}
        return Reply(question, standalone, res.answer.answer, res.answer.abstained, citations, trace)

    def _offline(self, session: Session, asked: str, question: str, started: float) -> Reply:
        """No model: gate picks chapters; their best sentences are cited extracts."""
        from .answer import ABSTAIN_TEXT
        from .workers.board import Board
        from .workers.routing import route, words

        r = route(session.tree, question)
        terms = r.content or words(question)
        named, within = r.chapters, None
        if len(named) > 6:
            within, named = named, []
        plan = session.gate.plan(question, terms, named, k=3, within=within)
        board = Board(session.tree, question)
        for chapter in (plan.rounds[0] if plan.rounds else []):
            rows = session.index.search(terms, chapter, k=2)
            if rows:
                board.submit(chapter, [{"sentence_ids": [session.index.ids[i] for i in rows], "why": "best match"}])
        pack = board.to_pack(max_items=4, max_tokens=600)
        citations = [self._citation(session, i) for i in pack.items]
        answer = citations[0].quote if citations else ABSTAIN_TEXT
        opened = [{"chapter": session.tree.node(c).title, "why": plan.opened_by.get(c, [])}
                  for c in (plan.rounds[0] if plan.rounds else [])]
        trace = {"scope": "offline", "chapters_total": len(session.tree.chapters()), "chapters_opened": opened,
                 "rounds": 1, "model_calls": 0, "tokens": 0, "cost_usd": 0.0,
                 "seconds": round(time.perf_counter() - started, 1), "evidence_items": len(pack.items),
                 "model": "offline (no AI: best-matching extracts)", "rewritten": False}
        return Reply(asked, question, answer, not citations, citations, trace)

    def _citation(self, session: Session, item) -> Citation:
        return Citation(item.label, item.section_path, item.text, item.node_id, item.page, session.tree.source,
                        item.role)

    # -------------------------------------------------------------- sources
    def page_image(self, citation: Citation, zoom: float = 1.6) -> Optional[bytes]:
        """The cited PDF page as PNG, the quote highlighted (None for non-PDF sources)."""
        if not citation.source or not citation.source.lower().endswith(".pdf") or not citation.page:
            return None
        import pymupdf as fitz

        doc = fitz.open(citation.source)
        page = doc[citation.page - 1]
        for sentence in re.split(r"(?<=[.!?])\s+", citation.quote):
            words_ = sentence.split()
            for size in (12, 7, 4):          # long quotes wrap lines: search for a shorter start
                if len(words_) < size and size != 4:
                    continue
                hits = page.search_for(" ".join(words_[:size]))
                if hits:
                    for rect in hits:
                        page.add_highlight_annot(rect)
                    break
        png = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")
        doc.close()
        return png

    def passage_html(self, session: Session, citation: Citation) -> str:
        """The cited passage with the quote marked (any format)."""
        import html

        node = session.tree.nodes.get(citation.passage_id)
        text = node.text if node else citation.quote
        safe, quote = html.escape(text), html.escape(citation.quote)
        return safe.replace(quote, f"<mark>{quote}</mark>") if quote in safe else f"<mark>{safe}</mark>"

    # ------------------------------------------------------------- feedback
    def feedback(self, reply: Reply, rating: str, note: str = "", document: Optional[str] = None) -> None:
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "document": document, "rating": rating, "note": note,
               **reply.to_dict()}
        with self.feedback_file.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
