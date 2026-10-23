"""PDF -> blocks via PyMuPDF, with light layout analysis.

Steps per document:
1. Extract text lines with font size / boldness / position (OCR for pages
   without a text layer, when Tesseract is available).
2. Drop running headers/footers and page numbers.
3. Pull out tables (``page.find_tables``) and images (figures + captions).
4. Group lines into paragraphs by vertical gap, indentation and style.
5. Classify headings by font size relative to the body text, boldness, or a
   numbered-heading pattern; assign levels by size rank and numbering depth.
6. When the PDF has a table of contents (bookmarks), it is the authority for headings:
   entries are matched to the text (also across split lines), take their levels from the
   outline, and other unnumbered "headings" (drop caps, styled captions) become text.
7. Merge paragraphs that continue across a page break.

Text is cleaned on extraction (``clean_text``: ligatures, invisible characters, curly
quotes), so quotes cited later match the canonical text exactly.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..parser import HEADING, Block, heading_number, match_heading, normalize_ws
from ..textutil import clean_text, match_key
from .html_loader import rows_to_markdown

BOLD_FLAG = 1 << 4
HEADING_SIZE_RATIO = 1.15
MAX_HEADING_WORDS = 15
MARGIN_FRACTION = 0.08          # top/bottom band searched for headers/footers
PARAGRAPH_GAP = 0.45            # gap (in font sizes) that starts a new paragraph
MIN_TEXT_CHARS_FOR_TEXT_LAYER = 20
_PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.I)
_CAPTION = re.compile(r"^(figure|fig\.|image|chart|diagram)\s*\d*", re.I)
_SENTENCE_END = re.compile(r"[.!?:;\"”)\]]$")
MIN_TOC_ENTRIES = 3
DROP_CAP_RATIO = 1.5            # a 1-2 letter line this much larger than body text
# Outline entries that group chapters, and entries that end such a group.
PART_WORDS = re.compile(r"^(book|part|volume|vol\.)\b", re.I)
END_OF_PART = re.compile(r"^(contents|table of contents|index|glossary|bibliography|references|"
                         r"endnotes|notes|acknowledg|about the author|appendix|appendices|afterword|"
                         r"the full project gutenberg|end of the project gutenberg)", re.I)


@dataclass
class Line:
    text: str
    size: float
    bold: bool
    x0: float
    y0: float
    y1: float
    page: int


@dataclass
class Group:
    kind: str
    lines: List[Line] = field(default_factory=list)
    text: str = ""
    page: int = 0
    y0: float = 0.0
    metadata: Dict = field(default_factory=dict)

    @property
    def size(self) -> float:
        return max(l.size for l in self.lines) if self.lines else 0.0

    @property
    def bold(self) -> bool:
        return bool(self.lines) and all(l.bold for l in self.lines)


def _join_lines(lines: List[str]) -> str:
    out = ""
    for line in lines:
        line = line.strip()
        if not out:
            out = line
        elif out.endswith("-") and line[:1].islower():
            out = out[:-1] + line  # de-hyphenate "indem-\nnity"
        else:
            out += " " + line
    return out


def _inside(bbox, rects) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rects)


def _page_lines(page, page_no: int, exclude, warnings: List[str]) -> Tuple[List[Line], list]:
    """Text lines and image bboxes for one page (OCR fallback if no text layer)."""
    import pymupdf as fitz

    textpage = None
    if len(page.get_text("text").strip()) < MIN_TEXT_CHARS_FOR_TEXT_LAYER:
        has_images = bool(page.get_images(full=False))
        if has_images:
            try:
                textpage = page.get_textpage_ocr(language="eng", dpi=300, full=True)
                warnings.append(f"page {page_no}: no text layer, used OCR")
            except Exception as exc:  # Tesseract missing or failed
                warnings.append(f"page {page_no}: no text layer and OCR unavailable ({exc})")
    data = page.get_text("dict", textpage=textpage,
                         flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_IMAGES)
    lines, images = [], []
    for block in data["blocks"]:
        if block.get("type") == 1:
            if textpage is None:  # a scanned page's image is the text, not a figure
                images.append(block["bbox"])
            continue
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans or _inside(line["bbox"], exclude):
                continue
            text = clean_text("".join(s["text"] for s in line["spans"])).strip()
            size = round(max(s["size"] for s in spans) * 2) / 2
            bold = all((s["flags"] & BOLD_FLAG) or "bold" in s["font"].lower() for s in spans)
            x0, y0, _, y1 = line["bbox"]
            lines.append(Line(text, size, bold, x0, y0, y1, page_no))
    return lines, images


def _furniture_keys(pages: List[Tuple[List[Line], float]]) -> set:
    """Texts repeated in the top/bottom margin across pages (headers/footers)."""
    counts: Counter = Counter()
    for lines, height in pages:
        seen = set()
        for line in lines:
            if line.y0 < height * MARGIN_FRACTION or line.y1 > height * (1 - MARGIN_FRACTION):
                seen.add(re.sub(r"\d+", "#", line.text.lower()))
        counts.update(seen)
    threshold = max(2, len(pages) // 2)
    return {key for key, c in counts.items() if c >= threshold}


def _is_furniture(line: Line, height: float, repeated: set) -> bool:
    in_margin = line.y0 < height * MARGIN_FRACTION or line.y1 > height * (1 - MARGIN_FRACTION)
    if not in_margin:
        return False
    return bool(_PAGE_NUMBER.match(line.text)) or re.sub(r"\d+", "#", line.text.lower()) in repeated


def _group_lines(lines: List[Line]) -> List[Group]:
    groups: List[Group] = []
    current: Optional[Group] = None
    for line in lines:
        if current is not None:
            prev = current.lines[-1]
            gap = line.y0 - prev.y1
            new = (
                gap > PARAGRAPH_GAP * prev.size
                or abs(line.size - prev.size) > 0.5
                or line.bold != prev.bold
                or line.x0 > prev.x0 + prev.size  # first-line indent
                # an indent of exactly one em, when this paragraph also began indented
                # (first-line-indent style; a hanging indent starts at the margin)
                or (line.x0 >= prev.x0 + 0.9 * prev.size
                    and abs(current.lines[0].x0 - line.x0) < 1.0)
            )
            if not new:
                current.lines.append(line)
                continue
        current = Group("text", [line], page=line.page, y0=line.y0)
        groups.append(current)
    for g in groups:
        g.text = _join_lines([l.text for l in g.lines])
    return groups


def _body_size(groups: List[Group]) -> float:
    weights: Counter = Counter()
    for g in groups:
        for line in g.lines:
            weights[line.size] += len(line.text)
    return weights.most_common(1)[0][0] if weights else 10.0


def _classify(groups: List[Group], body: float) -> None:
    for g in groups:
        if g.kind != "text":
            continue
        words = len(g.text.split())
        short = words <= MAX_HEADING_WORDS and len(g.lines) <= 3
        larger = g.size >= body * HEADING_SIZE_RATIO
        if short and (larger or (g.bold and not _SENTENCE_END.search(g.text))):
            g.kind = HEADING
        elif len(g.lines) == 1 and match_heading(g.text) and not g.text.startswith("#"):
            g.kind = HEADING  # plain-styled "2.1 Definitions"
        else:
            g.kind = "paragraph"

    heading_sizes = sorted({g.size for g in groups if g.kind == HEADING and g.size > body * 1.01},
                           reverse=True)
    rank = {size: i + 1 for i, size in enumerate(heading_sizes)}
    body_rank = len(heading_sizes) + 1
    for g in groups:
        if g.kind != HEADING:
            continue
        number = heading_number(g.text)
        depth = number.count(".") + 1 if number else 1
        g.metadata["number"] = number
        g.metadata["level"] = rank.get(g.size, body_rank) + depth - 1


def _merge_drop_caps(groups: List[Group], body: float) -> None:
    """A large one- or two-letter line ("I", "W") is the first letter of the next paragraph."""
    for i, g in enumerate(groups[:-1]):
        nxt = groups[i + 1]
        if (g.kind == "text" and len(g.lines) == 1 and g.text.isalpha() and len(g.text) <= 2
                and g.size >= body * DROP_CAP_RATIO and nxt.kind == "text" and nxt.page == g.page):
            joiner = "" if nxt.text[:1].islower() else " "
            nxt.text = g.text + joiner + nxt.text
            g.kind = "consumed"


def toc_levels(toc: List[Tuple[int, str, int]]) -> List[int]:
    """Heading levels from an outline, fixing flat outlines where "Book the First" and
    "CHAPTER I" share a level: chapters after a part entry move one level down, until the
    next part, a shallower entry, or back matter. Properly nested outlines are kept."""
    out: List[int] = []
    part_level: Optional[int] = None
    shift: Optional[int] = None
    for level, title, _ in toc:
        title = clean_text(title).strip()
        if part_level is not None and (level < part_level
                                       or (level == part_level and END_OF_PART.match(title))):
            part_level = shift = None
        if PART_WORDS.match(title):
            part_level, shift = level, None
            out.append(level)
            continue
        if part_level is not None and level >= part_level:
            if shift is None:
                shift = 1 if level == part_level else 0
            out.append(level + shift)
        else:
            out.append(level)
    return out


def _apply_toc(groups: List[Group], toc: List[Tuple[int, str, int]], info: Dict) -> None:
    """Make outline entries the headings (see module docstring, step 6)."""
    levels = toc_levels(toc)
    on_page: Dict[int, List[int]] = {}
    for i, g in enumerate(groups):
        on_page.setdefault(g.page, []).append(i)
    used, last, matched, unmatched = set(), -1, 0, []
    for (_, title, page), level in zip(toc, levels):
        key = match_key(title)
        if not key:
            continue
        hit = None
        for p in (page, page + 1, page - 1):
            for i in on_page.get(p, []):
                if i <= last or i in used or groups[i].kind not in ("text", HEADING, "paragraph"):
                    continue
                acc, j = "", i
                while j < len(groups) and j - i < 4 and groups[j].kind in ("text", HEADING, "paragraph"):
                    acc += match_key(groups[j].text)
                    if acc == key:
                        hit = (i, j)
                        break
                    if not acc or not key.startswith(acc):
                        break
                    j += 1
                if hit:
                    break
            if hit:
                break
        if hit is None:
            unmatched.append(title)
            continue
        i, j = hit
        g = groups[i]
        if j > i:  # a heading split over lines: "I." + "A SCANDAL IN BOHEMIA"
            g.text = " ".join(groups[k].text for k in range(i, j + 1))
            for k in range(i + 1, j + 1):
                groups[k].kind = "consumed"
                used.add(k)
        g.kind = HEADING
        g.metadata.update(level=level, number=heading_number(g.text), toc=True)
        used.add(i)
        last = j
        matched += 1
    # Other headings: numbered ones ("5.2 Fees") nest under the last outline heading;
    # unnumbered ones (drop caps, captions, styled lines) are text.
    toc_level = 0
    for g in groups:
        if g.kind != HEADING:
            continue
        if g.metadata.get("toc"):
            toc_level = g.metadata["level"]
            continue
        number = heading_number(g.text)
        if number and match_heading(g.text):
            g.metadata["level"] = toc_level + number.count(".") + 1
        else:
            g.kind = "paragraph"
    info.update(toc_used=True, toc_matched=matched, toc_unmatched=unmatched)


def _attach_captions(groups: List[Group]) -> None:
    """A text group right after a figure that looks like a caption becomes its text."""
    for i, g in enumerate(groups):
        if g.kind != "figure":
            continue
        nxt = groups[i + 1] if i + 1 < len(groups) else None
        if nxt is not None and nxt.kind != "figure" and _CAPTION.match(nxt.text) \
                and nxt.page == g.page:
            g.text = nxt.text
            nxt.kind = "consumed"


def _merge_continuations(groups: List[Group]) -> List[Group]:
    """Join a paragraph split by a page break: the earlier part has no sentence end."""
    merged: List[Group] = []
    for g in groups:
        prev = merged[-1] if merged else None
        if (prev is not None and g.kind == "paragraph" and prev.kind == "paragraph"
                and g.page == prev.page + 1 and not _SENTENCE_END.search(prev.text)):
            prev.text = _join_lines([prev.text, g.text])
            prev.metadata.setdefault("pages", [prev.page]).append(g.page)
            continue
        merged.append(g)
    return merged


def extract_pdf(path: Path, info: Optional[Dict] = None,
                use_toc: bool = True) -> Tuple[List[Block], Optional[str], int, List[str]]:
    """``info``, when given, is filled with the PDF's metadata fields, its outline and how
    the outline was used (for metadata extraction and structure readings). ``use_toc=False``
    ignores the outline (font-based headings only; for comparison)."""
    import pymupdf as fitz

    warnings: List[str] = []
    info = {} if info is None else info
    doc = fitz.open(str(path))
    toc = [(lvl, clean_text(title).strip(), page) for lvl, title, page in doc.get_toc(simple=True)]
    info.update(pdf_metadata={k: clean_text(v) for k, v in (doc.metadata or {}).items()
                              if isinstance(v, str) and v.strip()},
                toc=toc, toc_used=False, page_count=len(doc))
    page_data = []
    for page_no, page in enumerate(doc, 1):
        tables = []
        try:
            for table in page.find_tables().tables:
                rows = [[normalize_ws(c or "") for c in row] for row in table.extract()]
                rows = [r for r in rows if any(r)]
                if rows:
                    tables.append((tuple(table.bbox), rows_to_markdown(rows)))
        except Exception as exc:
            warnings.append(f"page {page_no}: table detection failed ({exc})")
        lines, images = _page_lines(page, page_no, [t[0] for t in tables], warnings)
        page_data.append((page_no, page.rect.height, lines, images, tables))

    repeated = _furniture_keys([(lines, h) for _, h, lines, _, _ in page_data])
    groups: List[Group] = []
    for page_no, height, lines, images, tables in page_data:
        lines = [l for l in lines if not _is_furniture(l, height, repeated)]
        page_groups = _group_lines(lines)
        for bbox, markdown in tables:
            page_groups.append(Group("table", text=markdown, page=page_no, y0=bbox[1]))
        for bbox in images:
            page_groups.append(Group("figure", text="", page=page_no, y0=bbox[1],
                                     metadata={"bbox": list(bbox)}))
        page_groups.sort(key=lambda g: g.y0)
        groups.extend(page_groups)

    body = _body_size(groups)
    _merge_drop_caps(groups, body)
    groups = [g for g in groups if g.kind != "consumed"]
    _classify(groups, body)
    if use_toc and len(toc) >= MIN_TOC_ENTRIES:
        _apply_toc(groups, toc, info)
        groups = [g for g in groups if g.kind != "consumed"]
    _attach_captions(groups)
    groups = _merge_continuations([g for g in groups if g.kind != "consumed"])

    title = (doc.metadata or {}).get("title") or None
    blocks: List[Block] = []
    for g in groups:
        if g.kind == HEADING:
            blocks.append(Block(HEADING, g.text, level=g.metadata["level"],
                                number=g.metadata["number"], page=g.page))
        elif g.kind == "figure":
            blocks.append(Block("figure", g.text or f"[Figure on page {g.page}]",
                                page=g.page, metadata={"bbox": g.metadata["bbox"]}))
        else:
            meta = {"pages": g.metadata["pages"]} if "pages" in g.metadata else {}
            blocks.append(Block(g.kind, g.text, page=g.page, metadata=meta))
    if not any(b.kind != HEADING for b in blocks):
        warnings.append("no body text extracted; the PDF may be scanned or image-only")
    return blocks, title, len(doc), warnings
