"""Phase 9: topology-routed prompts with a token budget (spec §5 phase 7).

**Topology analysis** of the kept evidence graph (items + their ancestors, with the
clause citations between items):
- ``depth``: the longest chapter → ... → item path (section levels + the item's own
  level; a sentence under a subsection under a chapter = 4)
- ``siblings``: the largest number of items sharing one parent section
- ``cross_refs``: clause citations (``cross_reference`` edges) between kept items, in
  either direction. Definitions are not counted: nearly every clause uses a defined
  term, so counting them would send almost every question to SYNTHESIS.

**Routes** (spec thresholds; precedence is decision D12, because the spec's rule list,
which checks depth first, contradicts its worked example, which sends depth 4 with 2
cross-references to SYNTHESIS):

    SYNTHESIS  cross_refs > 2   evidence that cites other clauses: combine, state exceptions
    SIBLING    siblings > 3     several provisions of one section: enumerate, don't drop any
    PATH       depth > 3        deeply nested evidence: answer at the governing level
    QA         otherwise        direct answer

**Token budget.** Items are kept in priority order:
1. primaries by score, each followed by the dependencies it needs (definitions,
   cited clauses, qualifiers, continuations, counterparts)
2. a dependency is dropped when the item that needs it is dropped
3. an item that does not fit is cut at a sentence boundary when at least 60% of it
   fits; otherwise it is dropped

Tokens are counted with ``count_tokens`` (pass a real tokenizer, or use the default
estimate: word and punctuation pieces, within ~15% of LLM tokenizers on contract text).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .models import CROSS_REFERENCE

SYNTHESIS, SIBLING, PATH, QA = "SYNTHESIS", "SIBLING", "PATH", "QA"
ROUTES = (SYNTHESIS, SIBLING, PATH, QA)
THRESHOLDS = {"cross_refs": 2, "siblings": 3, "depth": 3}

ROLE_TEXT = {
    "definition": "defines a term used in {by}",
    "reference": "cited by {by}",
    "qualifier": "qualifies {by}",
    "context": "continues {by}",
    "counterpart": "may conflict with {by}",
}

COMMON_RULES = """\
- Answer only from the evidence above. Do not use outside knowledge or infer beyond it.
- Cite every claim with its evidence label and section, e.g. [E2: 4.2 Exclusions from the Cap].
- If the evidence does not answer the question, say "The document does not say" and stop.
- If evidence items conflict, report both and cite each."""

TEMPLATES = {
    QA: """\
Task: answer the question directly in one or two sentences, then cite.""",
    PATH: """\
Task: the evidence is nested several levels deep (chapter > section > clause > sentence).
Answer at the most specific level that governs the question, and say which section it is.
If a broader section sets a rule and a narrower one changes it, the narrower one wins; say so.""",
    SIBLING: """\
Task: several provisions from the same section bear on the question. Go through each relevant
one; do not drop any. Give the answer as a short list, one cited line per provision.""",
    SYNTHESIS: """\
Task: the evidence items cite each other (shown as -> E#). Follow every citation, combine what the
clauses say, and state exceptions, carve-outs and conditions explicitly (e.g. "X applies, except
where Y [E3]"). Answer with the combined rule, then cite each clause used.""",
}

RESPONSE_FORMAT = """\
Respond in this format:
Answer: <answer>
Evidence: <labels with sections>
Confidence: <high | medium | low>, and why"""

# Per-route token budgets for the evidence block, chosen on the dev sets (docs/READINGS.md,
# Phase 9): the smallest budget that lost no dev answers, plus a margin (QA lost 1% at 200
# with v3.0; SYNTHESIS lost none at 200; SIBLING lost answers below 250; PATH never occurred).
DEFAULT_BUDGETS = {QA: 250, PATH: 300, SIBLING: 300, SYNTHESIS: 250}
TRUNCATE_MIN_SHARE = 0.6


def count_tokens(text: str) -> int:
    """Default estimate: word pieces and punctuation marks."""
    return len(re.findall(r"\w+|[^\w\s]", text))


@dataclass
class Topology:
    depth: int
    siblings: int
    cross_refs: int
    route: str

    def explain(self) -> str:
        return (f"route {self.route}: depth {self.depth}, max siblings {self.siblings}, "
                f"cross-references between items {self.cross_refs}")


def analyse(graph, node_ids: Sequence[str]) -> Topology:
    """Topology of the kept evidence graph and its route."""
    kept = set(node_ids)
    depth = 0
    by_parent: Dict[str, int] = {}
    for nid in kept:
        ancestors = [a for a in graph.ancestors(nid) if a.is_section]
        node = graph.nodes[nid]
        own = 1 if not node.is_content else (2 if node.node_type == "sentence" or
                                             (node.parent_id and graph.nodes[node.parent_id].is_content) else 1)
        depth = max(depth, len(ancestors) + own)
        section = next((a for a in graph.ancestors(nid) if a.is_section), None)
        key = section.id if section is not None else "doc"
        by_parent[key] = by_parent.get(key, 0) + 1
    siblings = max(by_parent.values(), default=0)

    def owner(nid: str) -> Optional[str]:
        """The kept item a node belongs to (itself, its paragraph, or a kept clause it sits in)."""
        if nid in kept:
            return nid
        for a in graph.ancestors(nid):
            if a.id in kept:
                return a.id
        return None

    pairs = set()
    for nid in kept:
        sources = [nid] + [c.id for c in graph.children(nid) if c.is_content]
        for src in sources:
            for nbr, _, _ in graph.neighbors(src, (CROSS_REFERENCE,), "out"):
                targets = [nbr.id] + ([c.id for c in graph.children(nbr.id) if c.is_content] if nbr.is_section else [])
                for t in targets:
                    o = owner(t)
                    if o is not None and o != nid:
                        pairs.add(tuple(sorted((nid, o))))
    cross_refs = len(pairs)
    if cross_refs > THRESHOLDS["cross_refs"]:
        route = SYNTHESIS
    elif siblings > THRESHOLDS["siblings"]:
        route = SIBLING
    elif depth > THRESHOLDS["depth"]:
        route = PATH
    else:
        route = QA
    return Topology(depth, siblings, cross_refs, route)


@dataclass
class BudgetReport:
    budget: Optional[int]
    used: int
    kept: List[str] = field(default_factory=list)       # node ids
    dropped: List[str] = field(default_factory=list)
    truncated: List[str] = field(default_factory=list)


def _cut_at_sentence(text: str, budget: int, counter: Callable[[str], int]) -> Optional[str]:
    sentences = re.split(r"(?<=[.;:])\s+", text)
    out = []
    for s in sentences:
        candidate = " ".join(out + [s])
        if counter(candidate) > budget:
            break
        out.append(s)
    return (" ".join(out) + " …") if out else None


def apply_budget(items: List, budget: Optional[int], counter: Callable[[str], int] = count_tokens) -> BudgetReport:
    """Trim ``items`` (EvidenceItems, in pack order) in place to ``budget`` tokens of
    evidence text. Priority: primaries by score, each followed by its own dependencies."""
    report = BudgetReport(budget, 0)
    if budget is None:
        report.used = sum(counter(i.text) for i in items)
        report.kept = [i.node_id for i in items]
        return report
    primaries = sorted((i for i in items if i.role == "primary"), key=lambda i: -i.score)
    deps: Dict[str, List] = {}
    orphans = []
    ids = {i.node_id for i in items}
    for i in items:
        if i.role != "primary":
            (deps.setdefault(i.required_by, []) if i.required_by in ids else orphans).append(i)
    order = []
    for p in primaries:
        order.append(p)
        stack = list(deps.get(p.node_id, []))
        while stack:  # dependencies of dependencies (qualifier -> cited clause -> ...)
            d = stack.pop(0)
            order.append(d)
            stack.extend(deps.get(d.node_id, []))
    order += [i for i in orphans if i not in order]
    order += [i for i in items if i not in order]
    kept_ids, used = set(), 0
    for i in order:
        if i.role != "primary" and i.required_by in ids and i.required_by not in kept_ids:
            report.dropped.append(i.node_id)  # what needed it was dropped
            continue
        cost = counter(i.text)
        if used + cost <= budget:
            kept_ids.add(i.node_id)
            used += cost
            continue
        room = budget - used
        if i.role == "primary" and room >= TRUNCATE_MIN_SHARE * cost:
            cut = _cut_at_sentence(i.text, room, counter)
            if cut:
                i.text = cut
                i.truncated = True
                kept_ids.add(i.node_id)
                used += counter(cut)
                report.truncated.append(i.node_id)
                continue
        report.dropped.append(i.node_id)
    items[:] = [i for i in items if i.node_id in kept_ids]
    report.used, report.kept = used, [i.node_id for i in items]
    return report


JSON_FORMAT = """\
Return JSON: "answer" (the answer, with inline citations like [E2]), "evidence" (the labels you
relied on, e.g. ["E1", "E3"]), "confidence" ("high", "medium" or "low"), and "insufficient"
(true only if the evidence does not answer the question; then "answer" says so and "evidence" is [])."""


def _evidence_block(pack) -> List[str]:
    labels = {i.node_id: i.label for i in pack.items}
    lines = [f"Question: {pack.question}", ""]
    if pack.confidence:
        lines.append(f"Retrieval confidence that the document answers this: {pack.confidence}.")
    lines += [f"Evidence ({len(pack.items)} items, route {pack.route}):", ""]
    for item in pack.items:
        note = ""
        if item.role != "primary" and item.required_by in labels:
            note = f" - {ROLE_TEXT.get(item.role, item.role).format(by=labels[item.required_by])}"
        elif item.role != "primary":
            note = f" - {item.role}"
        lines.append(f"[{item.label}] {item.citation}{note}")
        lines.append(item.text)
        refs = [f"-> {labels[t]}" for t in item.cites if t in labels]
        excerpts = [e for t, e in item.cite_excerpts.items() if t not in labels]
        if refs:
            lines.append("Cites: " + ", ".join(refs))
        for e in excerpts:
            lines.append(f"Cites (not in evidence): {e}")
        lines.append("")
    if pack.budget and pack.budget.dropped:
        lines.append(f"({len(pack.budget.dropped)} lower-ranked items omitted for length.)")
        lines.append("")
    return lines


def render_parts(pack, structured: bool = False):
    """(system, user) for chat APIs: the rules, route task and response format are the
    system prompt; the question and evidence are the user turn. ``structured``: ask for
    JSON (the API also enforces the schema)."""
    system = "\n".join(["You answer questions about a document using only the evidence you are given.", "",
                        "Rules:", COMMON_RULES, "", TEMPLATES[pack.route], "",
                        JSON_FORMAT if structured else RESPONSE_FORMAT])
    user = "\n".join(_evidence_block(pack)).rstrip()
    return system, user


def render(pack, graph=None) -> str:
    """The full prompt for ``pack`` (an EvidencePack) as one string, using its route's template."""
    lines = _evidence_block(pack)
    lines += ["Rules:", COMMON_RULES, "", TEMPLATES[pack.route], "", RESPONSE_FORMAT]
    return "\n".join(lines)
