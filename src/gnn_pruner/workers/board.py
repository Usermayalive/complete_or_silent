"""The central board (WORKFLOW B2, B5): one shared record per question.

Workers post findings as **sentence ids** (or passage ids), never as retyped text (R3).
For every post the board:
1. **Checks:** each id must exist in the document; unknown ids are rejected and logged.
2. **Quotes:** the text is taken from the document by id, so it is exact.
3. **Merges:** a finding whose sentences are already accepted adds nothing; findings from
   the same passage are combined.
4. **Adds required context:** the definitions of defined terms used in an accepted
   passage (tree links ``uses_definition``).
5. **Tracks references to follow** (``follow``): chapters a worker asks the board to
   search next.
6. **Builds the evidence pack** for the final answer: accepted findings first, then the
   definitions, each with its chapter / section path and pages.

**Budget:** every model call is counted. Workers stop when the board's budget is spent.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..evidence import EvidenceItem, EvidencePack


@dataclass
class Finding:
    passage: str
    sentence_ids: List[str]
    chapter: str
    worker: str
    why: str = ""
    role: str = "answer"            # answer | definition


@dataclass
class Board:
    tree: object
    question: str
    max_calls: int = 40
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    accepted: List[Finding] = field(default_factory=list)
    rejected: List[Dict] = field(default_factory=list)
    follow: List[str] = field(default_factory=list)       # chapter ids requested by workers
    workers: Dict[str, str] = field(default_factory=dict)  # chapter id -> state
    index: object = None                                   # SentenceIndex: resolves free-text follows
    partials: Dict[str, str] = field(default_factory=dict)  # chapter -> worker's partial answer (global)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---------------------------------------------------------------- budget
    def spend(self, completion) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += completion.input_tokens or 0
            self.output_tokens += completion.output_tokens or 0

    def has_budget(self) -> bool:
        return self.calls < self.max_calls

    # ----------------------------------------------------------------- posts
    def _chapter_of(self, node_id: str) -> Optional[str]:
        node = self.tree.node(node_id)
        while node.parent is not None and node.kind != "chapter":
            node = self.tree.node(node.parent)
        return node.id if node.kind == "chapter" else None

    def submit(self, worker: str, findings: List[Dict]) -> Dict[str, int]:
        """Check and record a worker's findings: ``[{"sentence_ids": [...], "why": ...}]``."""
        accepted = rejected = 0
        with self._lock:
            have = {s for f in self.accepted for s in f.sentence_ids}
            for f in findings:
                by_passage: Dict[str, List[str]] = {}
                bad = []
                for raw in f.get("sentence_ids", []):
                    sid = str(raw).strip("[] ")
                    if self.tree.has_sentence(sid):
                        by_passage.setdefault(self.tree.sentence(sid)[1], []).append(sid)
                    elif sid in self.tree.nodes and self.tree.node(sid).kind == "passage":
                        by_passage.setdefault(sid, []).extend(s for s, _ in self.tree.node(sid).sentences)
                    else:
                        bad.append(sid)
                if bad:
                    self.rejected.append({"worker": worker, "ids": bad, "why": f.get("why", ""),
                                          "reason": "unknown sentence id"})
                    rejected += 1
                for passage, sids in by_passage.items():
                    new = [s for s in dict.fromkeys(sids) if s not in have]
                    if not new:
                        continue
                    existing = next((a for a in self.accepted if a.passage == passage and a.role == "answer"), None)
                    if existing:
                        existing.sentence_ids.extend(new)
                    else:
                        self.accepted.append(Finding(passage, new, self._chapter_of(passage) or "", worker,
                                                     f.get("why", "")))
                    have.update(new)
                    accepted += 1
        return {"accepted": accepted, "rejected": rejected}

    def request_follow(self, worker: str, targets: List[str]) -> None:
        """A worker asks for something else to be searched: a node id, a chapter title or
        number, or free text ("rate card RC-8985"), which is looked up across the whole
        document and resolves to the chapter of the best-matching sentence outside the
        worker's own chapter."""
        with self._lock:
            for t in targets:
                chapter = None
                if t in self.tree.nodes:
                    chapter = self._chapter_of(t)
                else:
                    key = str(t).lower().strip()
                    chapter = next((c.id for c in self.tree.chapters()
                                    if key and (c.title.lower().startswith(key) or (c.number or "") == key)), None)
                    if chapter is None and self.index is not None:
                        chapter = self.index.best_chapter(str(t), exclude={worker})
                if chapter and chapter != worker and chapter not in self.follow:
                    self.follow.append(chapter)

    # ------------------------------------------------------------ evidence
    def add_definitions(self) -> int:
        """Definitions used by accepted passages join the pack (dependency rule, Phase 6)."""
        added = 0
        have = {f.passage for f in self.accepted}
        for f in list(self.accepted):
            for target in self.tree.node(f.passage).links.get("uses_definition", []):
                node = self.tree.nodes.get(target)
                if node is None or node.kind != "passage" or target in have:
                    continue
                self.accepted.append(Finding(target, [s for s, _ in node.sentences], self._chapter_of(target) or "",
                                             "board", "definition of a term used", role="definition"))
                have.add(target)
                added += 1
        return added

    def quote(self, finding: Finding) -> str:
        order = [s for s, _ in self.tree.node(finding.passage).sentences]
        ids = sorted(finding.sentence_ids, key=lambda s: order.index(s) if s in order else 0)
        return self.tree.quote(ids)

    def to_pack(self, max_items: int = 12, max_tokens: Optional[int] = None) -> EvidencePack:
        """Answers first, then definitions; at most ``max_items`` items and (when given)
        ``max_tokens`` of quoted text. The first item is always kept."""
        from ..prompt import count_tokens

        items, used = [], 0
        answers = [f for f in self.accepted if f.role == "answer"]
        defs = [f for f in self.accepted if f.role == "definition"]
        for k, f in enumerate((answers + defs)[:max_items], 1):
            t = count_tokens(self.quote(f))
            if max_tokens is not None and items and used + t > max_tokens:
                break
            used += t
            node = self.tree.node(f.passage)
            span = node.span or [0, 0]
            items.append(EvidenceItem(f"E{k}", f.passage, f.role, " / ".join(self.tree.path(f.passage)),
                                      self.quote(f), 1.0 if f.role == "answer" else 0.5, span[0], span[1],
                                      page=(node.pages or [None])[0], required_by="board" if f.role != "answer" else None))
        return EvidencePack(self.question, "unknown", items=items)

    def notes(self) -> str:
        """Workers' partial answers, in document order (whole-document questions)."""
        order = [c.id for c in self.tree.chapters()]
        return "\n".join(f"- {' / '.join(self.tree.path(c))}: {self.partials[c]}"
                         for c in sorted(self.partials, key=lambda c: order.index(c) if c in order else 0))

    @property
    def status(self) -> str:
        if any(f.role == "answer" for f in self.accepted):
            return "found"
        return "budget_reached" if not self.has_budget() else "not_found"
