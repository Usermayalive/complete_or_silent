"""HTML -> blocks using the standard library parser.

Headings come from <h1>-<h6>, paragraphs from block-level elements, tables
from <table>, figures from <img>/<figure>. Navigation and other page furniture
are skipped.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import List, Optional, Tuple

from ..parser import HEADING, Block, heading_number, normalize_ws

SKIP = {"script", "style", "nav", "header", "footer", "noscript", "svg", "head", "aside", "form"}
BLOCK = {"p", "div", "li", "blockquote", "pre", "section", "article", "main", "dd", "dt",
         "ul", "ol", "body", "figcaption"}
HEADINGS = {f"h{i}": i for i in range(1, 7)}


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: List[Block] = []
        self.title: Optional[str] = None
        self._buf: List[str] = []
        self._skip = 0
        self._in_title = False
        self._heading: Optional[int] = None
        self._table: Optional[List[List[str]]] = None
        self._cell: Optional[List[str]] = None
        self._figure: Optional[dict] = None
        self._in_caption = False

    # -- helpers
    def _flush(self) -> None:
        text = normalize_ws("".join(self._buf))
        self._buf = []
        if not text:
            return
        if self._heading:
            self.blocks.append(Block(HEADING, text, level=self._heading,
                                     number=heading_number(text)))
        else:
            self.blocks.append(Block("paragraph", text))

    # -- parser callbacks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in SKIP:
            if tag == "head":
                return  # still need <title>
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "title":
            self._in_title = True
        elif tag in HEADINGS:
            self._flush()
            self._heading = HEADINGS[tag]
        elif tag == "table":
            self._flush()
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._table.append([])
        elif tag in ("td", "th") and self._table is not None:
            self._cell = []
        elif tag == "figure":
            self._flush()
            self._figure = {"alt": "", "caption": []}
        elif tag == "figcaption" and self._figure is not None:
            self._in_caption = True
        elif tag == "img":
            alt = normalize_ws(attrs.get("alt") or "")
            if self._figure is not None:
                self._figure["alt"] = alt
            else:
                self._flush()
                self.blocks.append(Block("figure", alt or "[Figure]",
                                         metadata={"src": attrs.get("src")}))
        elif tag == "br":
            self._buf.append(" ")
        elif tag in BLOCK:
            self._flush()

    def handle_endtag(self, tag):
        if tag in SKIP and tag != "head":
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == "title":
            self._in_title = False
        elif tag in HEADINGS:
            self._flush()
            self._heading = None
        elif tag in ("td", "th") and self._cell is not None and self._table:
            self._table[-1].append(normalize_ws("".join(self._cell)))
            self._cell = None
        elif tag == "table" and self._table is not None:
            rows = [r for r in self._table if r]
            if rows:
                self.blocks.append(Block("table", rows_to_markdown(rows)))
            self._table = None
        elif tag == "figcaption":
            self._in_caption = False
        elif tag == "figure" and self._figure is not None:
            caption = normalize_ws("".join(self._figure["caption"]))
            self.blocks.append(Block("figure", caption or self._figure["alt"] or "[Figure]"))
            self._figure = None
        elif tag in BLOCK:
            self._flush()

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title = normalize_ws((self.title or "") + data)
        elif self._cell is not None:
            self._cell.append(data)
        elif self._in_caption and self._figure is not None:
            self._figure["caption"].append(data)
        elif self._table is None:
            self._buf.append(data)


def rows_to_markdown(rows: List[List[str]]) -> str:
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def extract_html(html: str) -> Tuple[List[Block], Optional[str], List[str]]:
    parser = _Extractor()
    parser.feed(html)
    parser.close()
    parser._flush()
    return parser.blocks, parser.title, []
