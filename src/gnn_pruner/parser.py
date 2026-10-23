"""Phase 1: turn documents into hierarchical nodes with source offsets.

Every loader (text, PDF, DOCX, HTML) emits a flat list of ``Block`` objects
(heading / paragraph / table / figure) with offsets into a canonical text.
``build_nodes`` then turns blocks into a parent-linked node hierarchy, so
structure handling is identical regardless of input format.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .models import (
    CHAPTER,
    CLAUSE,
    DOCUMENT,
    FIGURE,
    PARAGRAPH,
    SECTION,
    SENTENCE,
    SUBSECTION,
    TABLE,
    Node,
)

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUM_HEADING = re.compile(r"^((?:\d+\.)*\d+)\.?\s+([A-Z][^.!?:;]{0,80})$")
_LEADING_NUMBER = re.compile(r"^((?:\d+\.)*\d+)\.?\s+(.+)$")
# Split after sentence punctuation followed by whitespace and an uppercase/quote
# start, so "Section 2.1" or "e.g. this" are not split.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[“])")
_WS = re.compile(r"\s+")

MAX_NUMBERED_HEADING_WORDS = 10

HEADING = "heading"


@dataclass
class Block:
    kind: str  # "heading" | "paragraph" | "table" | "figure"
    text: str
    level: int = 0
    number: Optional[str] = None
    page: Optional[int] = None
    start: int = 0
    end: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def heading_number(title: str) -> Optional[str]:
    match = _LEADING_NUMBER.match(title)
    return match.group(1) if match else None


def match_heading(line: str) -> Optional[Tuple[int, str, Optional[str]]]:
    """Return (level, title, section_number) if ``line`` is a heading."""
    md = _MD_HEADING.match(line)
    if md:
        title = md.group(2).strip()
        return len(md.group(1)), title, heading_number(title)
    num = _NUM_HEADING.match(line)
    if num and len(num.group(2).split()) <= MAX_NUMBERED_HEADING_WORDS:
        number = num.group(1)
        return number.count(".") + 1, line.strip(), number
    return None


def split_sentences(text: str, offset: int = 0) -> Iterator[Tuple[str, int, int]]:
    """Yield (sentence_text, start, end) with offsets relative to the source."""
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        chunk = text[start:match.start()]
        if chunk.strip():
            yield normalize_ws(chunk), offset + start, offset + match.start()
        start = match.end()
    chunk = text[start:].rstrip()
    if chunk.strip():
        yield normalize_ws(chunk), offset + start, offset + start + len(chunk)


# ------------------------------------------------------------ text -> blocks

def text_blocks(text: str) -> List[Block]:
    """Markdown / plain text -> blocks with offsets into ``text`` itself."""
    blocks: List[Block] = []
    block_start: Optional[int] = None
    block_end = 0

    def flush() -> None:
        nonlocal block_start
        if block_start is None:
            return
        raw = text[block_start:block_end]
        is_table = all(l.strip().startswith("|") for l in raw.splitlines() if l.strip())
        blocks.append(Block(TABLE if is_table else PARAGRAPH, raw,
                            start=block_start, end=block_end))
        block_start = None

    pos = 0
    for line in text.splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        heading = match_heading(stripped)
        if heading:
            flush()
            level, title, number = heading
            blocks.append(Block(HEADING, title, level=level, number=number,
                                start=line_start, end=line_start + len(line.rstrip())))
            continue
        if block_start is None:
            block_start = line_start + (len(line) - len(line.lstrip()))
        block_end = line_start + len(line.rstrip())
    flush()
    return blocks


def assemble(blocks: Sequence[Block]) -> Tuple[str, List[Block]]:
    """Render loader blocks into a canonical markdown-like text and set offsets.

    Used by formats without a natural plain-text form (PDF, DOCX, HTML), so
    every node still has a character span to cite.
    """
    parts: List[str] = []
    pos = 0
    for block in blocks:
        if parts:
            parts.append("\n\n")
            pos += 2
        if block.kind == HEADING:
            prefix = "#" * max(1, min(block.level, 6)) + " "
            parts.append(prefix)
            pos += len(prefix)
        block.start = pos
        parts.append(block.text)
        pos += len(block.text)
        block.end = pos
    return "".join(parts), list(blocks)


# ------------------------------------------------------------ blocks -> nodes

_CHAPTER_WORDS = re.compile(r"^(chapter|part|article|schedule|annex|appendix|exhibit)\b", re.I)
# A paragraph that opens with its own label: "4.1.2 The Supplier shall ...", "(a) ...".
_CLAUSE_LABEL = re.compile(r"^((?:\d+\.)+\d+\.?|\d+\.|\([a-z]{1,3}\)|\(\d{1,2}\)|[a-z]\))\s+(?=\S)")
# Inline enumerations inside a paragraph: "...: (a) x; (b) y; and (c) z."
_INLINE_CLAUSE = re.compile(r"(?:(?<=^)|(?<=[\s:;]))\(([a-z]{1,3}|\d{1,2})\)\s+(?=\S)")
MIN_INLINE_CLAUSES = 2


def _promote_title(blocks: Sequence[Block]) -> Tuple[Optional[str], List[Block]]:
    """A lone top heading followed directly by the next heading is the document title."""
    blocks = list(blocks)
    if not blocks or blocks[0].kind != HEADING:
        return None, blocks
    first = blocks[0]
    if first.metadata.get("document_title"):
        return first.text, blocks[1:]
    if first.number or len(blocks) < 2 or blocks[1].kind != HEADING:
        return None, blocks
    if all(b.level > first.level for b in blocks[1:] if b.kind == HEADING):
        return first.text, blocks[1:]
    return None, blocks


def clause_label(text: str) -> Optional[str]:
    match = _CLAUSE_LABEL.match(text)
    if not match:
        return None
    return match.group(1).rstrip(".").strip("()").rstrip(")")


def inline_clauses(text: str) -> List[Tuple[str, int, int]]:
    """(label, start, end) of enumerated items within ``text``, if it has a list."""
    matches = list(_INLINE_CLAUSE.finditer(text))
    if len(matches) < MIN_INLINE_CLAUSES or matches[0].group(1) not in ("a", "i", "1"):
        return []
    spans = []
    for k, m in enumerate(matches):
        end = matches[k + 1].start() if k + 1 < len(matches) else len(text.rstrip())
        spans.append((m.group(1), m.start(), end))
    return spans


def _assign_section_kinds(nodes: List[Node]) -> None:
    by_id = {n.id: n for n in nodes}
    has_child_section = {n.parent_id for n in nodes if n.node_type == SECTION}
    for node in nodes:  # parents precede children
        if node.node_type != SECTION:
            continue
        if _CHAPTER_WORDS.match(node.title or "") and node.id in has_child_section:
            node.node_type = CHAPTER
            continue
        parent = by_id.get(node.parent_id)
        if parent is not None and parent.node_type in (SECTION, SUBSECTION):
            node.node_type = SUBSECTION


def build_nodes(blocks: Sequence[Block], text_length: int, title: str = "Document",
                source: Optional[str] = None) -> List[Node]:
    counter = itertools.count(1)
    promoted, blocks = _promote_title(blocks)
    title = promoted or title
    root = Node(id="doc", node_type=DOCUMENT, text=title, title=title, level=0,
                start=0, end=text_length, metadata={"source": source} if source else {})
    nodes: List[Node] = [root]
    stack: List[Node] = [root]

    def base_metadata(block: Block) -> Dict[str, Any]:
        meta = {k: v for k, v in block.metadata.items() if k != "document_title"}
        if source:
            meta["source"] = f"{source}#page={block.page}" if block.page else source
        return meta

    def section_number() -> Optional[str]:
        return next((s.metadata["number"] for s in reversed(stack) if s.metadata.get("number")), None)

    for block in blocks:
        if block.kind == HEADING:
            level = block.level
            # Hierarchy validation: "4.1" belongs under "4" whatever its font
            # size or markup level says.
            if block.number and "." in block.number:
                prefix = block.number.rsplit(".", 1)[0]
                for anc in reversed(stack[1:]):
                    if anc.metadata.get("number") == prefix:
                        level = anc.level + 1
                        break
            while stack[-1].level >= level:
                stack.pop().end = block.start
            meta = base_metadata(block)
            if block.number:
                meta["number"] = block.number
            section = Node(id=f"sec{next(counter)}", node_type=SECTION, text=block.text,
                           title=block.text, parent_id=stack[-1].id, level=level,
                           start=block.start, end=text_length, page=block.page,
                           metadata=meta)
            nodes.append(section)
            stack.append(section)
            continue

        parent = stack[-1]
        node_type = {"table": TABLE, "figure": FIGURE}.get(block.kind, PARAGRAPH)
        meta = base_metadata(block)
        label = clause_label(block.text) if node_type == PARAGRAPH else None
        if label:
            node_type = CLAUSE
            meta["label"] = label
            # "4.1.2" is a full number; "(a)" is qualified by the section: "5(a)".
            meta["number"] = label if label[0].isdigit() and "." in label else \
                (f"{section_number()}({label})" if section_number() else None)
        para = Node(
            id=f"p{next(counter)}",
            node_type=node_type,
            text=block.text.strip() if node_type == TABLE else normalize_ws(block.text),
            parent_id=parent.id,
            level=parent.level + 1,
            start=block.start,
            end=block.end,
            page=block.page,
            metadata=meta,
        )
        nodes.append(para)
        if node_type not in (PARAGRAPH, CLAUSE):
            continue
        items = inline_clauses(block.text)
        if items:
            number = section_number()
            for k, (item_label, s, e) in enumerate(items, 1):
                nodes.append(Node(
                    id=f"{para.id}.c{k}", node_type=CLAUSE,
                    text=normalize_ws(block.text[s:e]), parent_id=para.id,
                    level=para.level + 1, start=block.start + s, end=block.start + e,
                    page=block.page,
                    metadata={"label": item_label,
                              "number": f"{number}({item_label})" if number else None}))
            continue
        sentences = list(split_sentences(block.text, block.start))
        # A one-sentence paragraph would just duplicate its parent.
        if len(sentences) > 1:
            for k, (sent, s, e) in enumerate(sentences, 1):
                nodes.append(Node(id=f"{para.id}.s{k}", node_type=SENTENCE,
                                  text=sent, parent_id=para.id,
                                  level=para.level + 1, start=s, end=e,
                                  page=block.page))
    _assign_section_kinds(nodes)
    return nodes


def parse_text(text: str, title: str = "Document", source: Optional[str] = None) -> List[Node]:
    """Parse markdown / plain text into a flat, parent-linked list of nodes."""
    return build_nodes(text_blocks(text), len(text), title=title, source=source)
