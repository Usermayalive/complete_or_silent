"""DOCX -> blocks via python-docx.

Headings come from "Title" / "Heading N" styles, falling back to bold or
numbered lines for documents that fake headings with direct formatting.
Body order (paragraphs and tables interleaved) is preserved.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from ..parser import HEADING, Block, heading_number, match_heading, normalize_ws
from .html_loader import rows_to_markdown

_HEADING_STYLE = re.compile(r"^heading\s*(\d)$", re.I)
MAX_BOLD_HEADING_WORDS = 12


def _iter_body(document):
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield Paragraph(child, document)
        elif tag == "tbl":
            yield Table(child, document)


def _is_bold(paragraph) -> bool:
    runs = [r for r in paragraph.runs if r.text.strip()]
    return bool(runs) and all(
        r.bold or (r.style is not None and r.style.font.bold) for r in runs)


def _figure_alt(paragraph) -> Optional[str]:
    drawings = paragraph._p.xpath(".//w:drawing")
    if not drawings:
        return None
    descr = paragraph._p.xpath(".//wp:docPr/@descr")
    return normalize_ws(descr[0]) if descr else ""


def extract_docx(path: Path) -> Tuple[List[Block], Optional[str], List[str]]:
    import docx

    document = docx.Document(str(path))
    blocks: List[Block] = []
    title: Optional[str] = document.core_properties.title or None
    has_title_style = False

    for item in _iter_body(document):
        if item.__class__.__name__ == "Table":
            rows = [[normalize_ws(c.text) for c in row.cells] for row in item.rows]
            if rows:
                blocks.append(Block("table", rows_to_markdown(rows)))
            continue

        text = normalize_ws(item.text)
        alt = _figure_alt(item)
        if alt is not None:
            blocks.append(Block("figure", text or alt or "[Figure]"))
            continue
        if not text:
            continue

        style = (item.style.name if item.style is not None else "") or ""
        styled = _HEADING_STYLE.match(style)
        if style.lower() == "title":
            has_title_style = True
            title = title or text
            blocks.append(Block(HEADING, text, level=0, number=heading_number(text),
                                metadata={"document_title": not blocks}))
        elif styled:
            blocks.append(Block(HEADING, text, level=int(styled.group(1)),
                                number=heading_number(text)))
        elif _is_bold(item) and len(text.split()) <= MAX_BOLD_HEADING_WORDS \
                and not text.endswith("."):
            number = heading_number(text)
            level = number.count(".") + 1 if number else 1
            blocks.append(Block(HEADING, text, level=level, number=number))
        elif match_heading(text) and not text.startswith("#"):
            level, heading_title, number = match_heading(text)
            blocks.append(Block(HEADING, heading_title, level=level, number=number))
        else:
            if style.lower().startswith("list"):
                text = f"- {text}"
            blocks.append(Block("paragraph", text))

    # "Title" sits above "Heading 1": shift so levels start at 1.
    shift = 1 if has_title_style else 0
    for block in blocks:
        if block.kind == HEADING:
            block.level += shift
    return blocks, title, []
