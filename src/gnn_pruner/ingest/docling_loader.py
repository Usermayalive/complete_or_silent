"""G1 (decision D7): Docling as the spec's document parser.

Docling converts PDF / DOCX / HTML into a DoclingDocument (layout model for PDFs).
Its items become the same Blocks the native loaders produce, so the rest of the
pipeline (nodes, graph, four layers) is unchanged:

    TitleItem             -> heading, level 1
    SectionHeaderItem     -> heading; level from its number ("4.1" -> 3), else Docling's level + 1
    TextItem / ListItem   -> paragraph
    TableItem             -> table (markdown)
    PictureItem           -> figure (caption)

The first PDF conversion downloads Docling's layout models (needs network once).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from ..parser import Block, heading_number

HEADING = "heading"


def extract_docling(path: Path) -> Tuple[List[Block], Optional[str], Optional[int], List[str]]:
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem, TextItem, TitleItem

    result = DocumentConverter().convert(str(path))
    doc = result.document
    blocks: List[Block] = []
    title: Optional[str] = None
    warnings = [f"docling: {e}" for e in getattr(result, "errors", []) or []]
    pages = set()
    for item, _ in doc.iterate_items():
        page = item.prov[0].page_no if getattr(item, "prov", None) else None
        if page is not None:
            pages.add(page)
        if isinstance(item, TitleItem):
            title = title or item.text
            blocks.append(Block(HEADING, item.text, level=1, page=page))
        elif isinstance(item, SectionHeaderItem):
            number = heading_number(item.text)
            level = number.count(".") + 2 if number else (getattr(item, "level", 1) or 1) + 1
            blocks.append(Block(HEADING, item.text, level=level, number=number, page=page))
        elif isinstance(item, TableItem):
            blocks.append(Block("table", item.export_to_markdown(doc=doc), page=page))
        elif isinstance(item, PictureItem):
            caption = item.caption_text(doc) if hasattr(item, "caption_text") else ""
            blocks.append(Block("figure", caption or "[Figure]", page=page))
        elif isinstance(item, TextItem) and item.text.strip():
            blocks.append(Block("paragraph", item.text, page=page))
    return blocks, title, (max(pages) if pages else None), warnings
