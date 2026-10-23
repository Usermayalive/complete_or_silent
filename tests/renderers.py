"""Render the markdown fixtures to PDF / DOCX / HTML for ingestion tests.

Deliberately independent of the package's own parser. The PDF output includes
the layout noise real files have: running headers, "Page N of M" footers, a
ruled table, and small pages so paragraphs break across pages.

    python tests/renderers.py   # writes examples/*.pdf|.docx|.html
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import List, Tuple

Item = Tuple[str, object]  # ("heading", (level, text)) | ("paragraph", text) | ("table", rows)


def read_markdown(text: str) -> List[Item]:
    items: List[Item] = []
    for chunk in re.split(r"\n\s*\n", text.strip()):
        lines = [l.strip() for l in chunk.strip().splitlines() if l.strip()]
        if not lines:
            continue
        if lines[0].startswith("#"):
            hashes, title = lines[0].split(" ", 1)
            items.append(("heading", (len(hashes), title.strip())))
        elif lines[0].startswith("|"):
            rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines
                    if not re.fullmatch(r"\|?[\s\-|:]+\|?", l)]
            items.append(("table", rows))
        else:
            items.append(("paragraph", " ".join(lines)))
    return items


# ---------------------------------------------------------------------- PDF

PAGE_W, PAGE_H, MARGIN = 420, 520, 50
HEADING_SIZE = {1: 18, 2: 14, 3: 12}
BODY_SIZE = 10


def render_pdf(text: str, out: Path, header: str = "CONFIDENTIAL") -> Path:
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = MARGIN

    def ensure(space: float):
        nonlocal page, y
        if y + space > PAGE_H - MARGIN:
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            y = MARGIN

    def write_wrapped(content: str, size: float, font: str):
        nonlocal y
        width = PAGE_W - 2 * MARGIN
        line = ""
        for word in content.split():
            trial = f"{line} {word}".strip()
            if fitz.get_text_length(trial, fontname=font, fontsize=size) > width and line:
                ensure(size * 1.3)
                page.insert_text((MARGIN, y + size), line, fontname=font, fontsize=size)
                y += size * 1.3
                line = word
            else:
                line = trial
        if line:
            ensure(size * 1.3)
            page.insert_text((MARGIN, y + size), line, fontname=font, fontsize=size)
            y += size * 1.3

    for kind, value in read_markdown(text):
        if kind == "heading":
            level, title = value
            size = HEADING_SIZE.get(level, 12)
            ensure(size * 4)
            y += size * 0.6
            write_wrapped(title, size, "hebo")
            y += size * 0.5
        elif kind == "paragraph":
            write_wrapped(value, BODY_SIZE, "helv")
            y += BODY_SIZE * 0.9
        elif kind == "table":
            rows = value
            col_w = (PAGE_W - 2 * MARGIN) / len(rows[0])
            row_h = BODY_SIZE * 2
            ensure(row_h * len(rows))
            for row in rows:
                for c, cell in enumerate(row):
                    rect = fitz.Rect(MARGIN + c * col_w, y, MARGIN + (c + 1) * col_w, y + row_h)
                    page.draw_rect(rect, color=(0, 0, 0), width=0.7)
                    page.insert_text((rect.x0 + 4, rect.y0 + BODY_SIZE * 1.3), cell,
                                     fontname="helv", fontsize=BODY_SIZE)
                y += row_h
            y += BODY_SIZE

    total = len(doc)
    for i, pg in enumerate(doc, 1):
        pg.insert_text((MARGIN, 25), header, fontname="helv", fontsize=8)
        pg.insert_text((PAGE_W / 2 - 20, PAGE_H - 20), f"Page {i} of {total}",
                       fontname="helv", fontsize=8)
    doc.save(str(out))
    return out


# --------------------------------------------------------------------- DOCX

def render_docx(text: str, out: Path) -> Path:
    import docx

    document = docx.Document()
    for kind, value in read_markdown(text):
        if kind == "heading":
            level, title = value
            # "#" is the document title; "##" -> Heading 1, "###" -> Heading 2.
            document.add_heading(title, level=level - 1)
            if level == 1:
                document.core_properties.title = title
        elif kind == "paragraph":
            document.add_paragraph(value)
        elif kind == "table":
            rows = value
            table = document.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Table Grid"
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    table.cell(r, c).text = cell
    document.save(str(out))
    return out


# --------------------------------------------------------------------- HTML

def render_html(text: str, out: Path) -> Path:
    parts = ["<html><head><title>Contract</title><style>p{margin:0}</style></head><body>",
             "<nav><a href='/'>Home</a> | <a href='/docs'>All documents</a></nav>",
             "<header>Contract portal</header><main>"]
    for kind, value in read_markdown(text):
        if kind == "heading":
            level, title = value
            parts.append(f"<h{level}>{html.escape(title)}</h{level}>")
        elif kind == "paragraph":
            parts.append(f"<p>{html.escape(value)}</p>")
        elif kind == "table":
            head, *body = value
            parts.append("<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in head) + "</tr>")
            for row in body:
                parts.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>")
            parts.append("</table>")
    parts.append("</main><footer>&copy; 2026 Contract portal</footer>"
                 "<script>console.log('tracking')</script></body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


RENDERERS = {".pdf": render_pdf, ".docx": render_docx, ".html": render_html}


if __name__ == "__main__":
    examples = Path(__file__).resolve().parent.parent / "examples"
    for md in sorted(examples.glob("*.md")):
        for suffix, render in RENDERERS.items():
            target = md.with_suffix(suffix)
            render(md.read_text(encoding="utf-8"), target)
            print(f"wrote {target}")
