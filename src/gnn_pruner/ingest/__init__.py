"""Phase 1 ingestion: load PDF / DOCX / HTML / markdown into blocks and nodes.

    doc = load_document("contract.pdf")
    doc.nodes, doc.text, doc.warnings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..models import Node
from ..parser import Block, assemble, build_nodes, text_blocks


@dataclass
class ParsedDocument:
    title: str
    source: str
    text: str  # canonical text; node offsets point into it
    blocks: List[Block]
    nodes: List[Node]
    page_count: Optional[int] = None
    warnings: List[str] = field(default_factory=list)
    info: Dict = field(default_factory=dict)  # loader extras: PDF metadata, outline


def _finish(path: Path, blocks: List[Block], title: Optional[str], warnings: List[str],
            page_count: Optional[int] = None, text: Optional[str] = None,
            info: Optional[Dict] = None) -> ParsedDocument:
    if text is None:
        text, blocks = assemble(blocks)
    title = title or path.stem
    nodes = build_nodes(blocks, len(text), title=title, source=str(path))
    return ParsedDocument(title, str(path), text, blocks, nodes, page_count, warnings, info or {})


def load_text(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8")
    return _finish(path, text_blocks(text), None, [], text=text)


def load_pdf(path: Path) -> ParsedDocument:
    from .pdf_loader import extract_pdf

    info: Dict = {}
    blocks, title, page_count, warnings = extract_pdf(path, info)
    return _finish(path, blocks, title, warnings, page_count, info=info)


def load_docx(path: Path) -> ParsedDocument:
    from .docx_loader import extract_docx

    blocks, title, warnings = extract_docx(path)
    return _finish(path, blocks, title, warnings)


def load_html(path: Path) -> ParsedDocument:
    from .html_loader import extract_html

    blocks, title, warnings = extract_html(path.read_text(encoding="utf-8", errors="replace"))
    return _finish(path, blocks, title, warnings)


LOADERS: Dict[str, Callable[[Path], ParsedDocument]] = {
    ".md": load_text, ".markdown": load_text, ".txt": load_text,
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html, ".htm": load_html,
}


def load_docling(path: Path) -> ParsedDocument:
    from .docling_loader import extract_docling

    blocks, title, page_count, warnings = extract_docling(path)
    return _finish(path, blocks, title, warnings, page_count)


PARSERS = ("native", "docling")
DOCLING_SUFFIXES = (".pdf", ".docx", ".html", ".htm", ".md")


def load_document(path: str, parser: str = "native") -> ParsedDocument:
    """``parser="docling"`` uses Docling (spec §6) for PDF / DOCX / HTML / markdown."""
    p = Path(path)
    if parser not in PARSERS:
        raise ValueError(f"parser must be one of {PARSERS}")
    if parser == "docling" and p.suffix.lower() in DOCLING_SUFFIXES:
        return load_docling(p)
    loader = LOADERS.get(p.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type {p.suffix!r}; supported: {sorted(LOADERS)}")
    return loader(p)
