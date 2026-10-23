"""The document tree (WORKFLOW A3): root -> chapters -> sections -> subsections -> passages.

This is the navigation view that chapter workers search. It is built from the parsed
``DocumentGraph`` and does not replace it: the GNN keeps its own four-layer view
(``layers.py``), and node ids are shared, so a passage here is the same node there.

    root        the document: title, metadata, overall summary
    part        optional grouping of chapters ("Book the First"); not a level of its own
    chapter     one worker's unit: chapters, articles, schedules, or virtual chapters
    section     sections inside a chapter
    subsection  deeper headings (flattened), or ~1,200-token slices of long text
    passage     original text: paragraphs, clauses, tables, each with sentence ids

Rules:
- **Same depth everywhere.** A level the document lacks gets a placeholder (virtual) node,
  so every passage sits at root > chapter > section > subsection > passage.
- **Wrapper headings are unwrapped.** A top heading holding nearly all the text (a book's
  title heading) is not a chapter; its children are.
- **Schedules, annexes, exhibits and appendices are chapters** wherever they appear.
- **Long unstructured text is split** into virtual chapters (~25 pages, at most ~40) and
  long subsections into ~1,200-token slices, always at passage boundaries.
- **Cite by id:** every sentence has an id; ``quote(ids)`` returns the exact source text.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .document_graph import DocumentGraph
from .models import CLAUSE, CROSS_REFERENCE, DEFINITION_LINK, FIGURE, PARAGRAPH, SENTENCE, TABLE
from .prompt import count_tokens

ROOT, PART, CHAPTER, SECTION, SUBSECTION, PASSAGE = "root", "part", "chapter", "section", "subsection", "passage"
LAYERS = {ROOT: 0, PART: 0, CHAPTER: 1, SECTION: 2, SUBSECTION: 3, PASSAGE: 4}
TREE_VERSION = 1

PASSAGE_TYPES = (PARAGRAPH, CLAUSE, TABLE, FIGURE)
WRAPPER_SHARE = 0.85            # a top heading holding this share of the text is a wrapper
MAX_SUBSECTION_TOKENS = 2000    # longer subsections are sliced ...
TARGET_SUBSECTION_TOKENS = 1200  # ... into slices of about this size
MAX_CHAPTER_PAGES = 40          # longer chapters without sections become virtual chapters ...
TARGET_CHAPTER_PAGES = 25       # ... of about this many pages
TOKENS_PER_PAGE = 500           # for documents without page numbers

PART_TITLE = re.compile(r"^(book|part|volume|vol\.)\b", re.I)
LIFT_TO_CHAPTER = re.compile(r"^(schedule|annex|appendix|exhibit|attachment)\b", re.I)
CONTENTS_TITLE = re.compile(r"^(table of )?contents\.?$", re.I)


@dataclass
class TreeNode:
    id: str
    kind: str
    title: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    number: Optional[str] = None
    pages: Optional[List[int]] = None          # [first, last]
    span: Optional[List[int]] = None           # [start, end] offsets in the source text
    tokens: int = 0
    virtual: bool = False
    role: str = "body"                         # body | contents
    part: Optional[str] = None                 # chapters: the part they belong to
    text: Optional[str] = None                 # passages only
    sentences: List[List[str]] = field(default_factory=list)  # passages: [[id, text], ...]
    heading: Optional[str] = None              # passages: flattened deeper heading, if any
    summary: Optional[str] = None              # filled by enrichment (sections and up)
    key_facts: List[Dict] = field(default_factory=list)       # filled by enrichment (subsections)
    defined_terms: List[str] = field(default_factory=list)
    links: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    @property
    def layer(self) -> int:
        return LAYERS[self.kind]


@dataclass
class DocTree:
    nodes: Dict[str, TreeNode]
    title: str
    source: Optional[str] = None
    kind: str = "document"                     # book | contract | document
    info: Dict = field(default_factory=dict)   # loader extras (PDF metadata, outline)
    warnings: List[str] = field(default_factory=list)
    version: int = TREE_VERSION
    _sentence_index: Optional[Dict[str, Tuple[str, str]]] = field(default=None, repr=False)

    # ------------------------------------------------------------------ access
    @property
    def root(self) -> TreeNode:
        return self.nodes[ROOT]

    def node(self, node_id: str) -> TreeNode:
        return self.nodes[node_id]

    def children(self, node_id: str) -> List[TreeNode]:
        return [self.nodes[c] for c in self.nodes[node_id].children]

    def of_kind(self, kind: str) -> List[TreeNode]:
        return [n for n in self.walk() if n.kind == kind]

    def chapters(self) -> List[TreeNode]:
        return self.children(ROOT)

    def parts(self) -> List[TreeNode]:
        return [self.nodes[p] for p in self.root.metadata.get("parts", [])]

    def walk(self, node_id: str = ROOT) -> Iterable[TreeNode]:
        node = self.nodes[node_id]
        yield node
        for child in node.children:
            yield from self.walk(child)

    def passages(self, node_id: str = ROOT) -> List[TreeNode]:
        return [n for n in self.walk(node_id) if n.kind == PASSAGE]

    def path(self, node_id: str) -> List[str]:
        """Titles from the part / chapter down. Placeholder levels are left out; a slice of
        a long subsection keeps its "(k/n)" marker."""
        out, node = [], self.nodes[node_id]
        if node.kind == PASSAGE:
            if node.heading:
                out.append(node.heading)
            node = self.nodes[node.parent]
        while node.parent is not None:
            slice_mark = re.search(r"\(\d+/\d+\)$", node.title or "")
            if not node.virtual or node.kind == CHAPTER:
                out.append(node.title)
            elif slice_mark and node.kind == SUBSECTION:
                out.append(slice_mark.group(0))
            if node.kind == CHAPTER and node.part:
                out.append(self.nodes[node.part].title)
            node = self.nodes[node.parent]
        return list(reversed(out))

    def sentence(self, sentence_id: str) -> Tuple[str, str]:
        """(text, passage id) for a sentence id."""
        if self._sentence_index is None:
            self._sentence_index = {sid: (text, p.id) for p in self.passages() for sid, text in p.sentences}
        return self._sentence_index[sentence_id]

    def has_sentence(self, sentence_id: str) -> bool:
        try:
            self.sentence(sentence_id)
            return True
        except KeyError:
            return False

    def quote(self, sentence_ids: Iterable[str]) -> str:
        """Exact source text for sentence ids (cite by id, R3)."""
        return " ".join(self.sentence(s)[0] for s in sentence_ids)

    # ---------------------------------------------------------------- display
    def stats(self) -> Dict[str, int]:
        counts = {k: 0 for k in (PART, CHAPTER, SECTION, SUBSECTION, PASSAGE)}
        virtual = {k: 0 for k in (CHAPTER, SECTION, SUBSECTION)}
        for n in self.nodes.values():
            if n.kind in counts:
                counts[n.kind] += 1
            if n.virtual and n.kind in virtual:
                virtual[n.kind] += 1
        out = {f"{k}s": v for k, v in counts.items()}
        out.update({f"virtual_{k}s": v for k, v in virtual.items()})
        out["sentences"] = sum(len(p.sentences) for p in self.passages())
        out["tokens"] = self.root.tokens
        out["pages"] = (self.root.pages or [0, 0])[1]
        return out

    def outline(self, depth: int = SECTION, width: int = 70) -> str:
        lines = [f"{self.title}  [{self.kind}, {self.root.tokens:,} tokens"
                 + (f", {self.root.pages[1]} pages]" if self.root.pages else "]")]
        current_part = None

        def walk(node_id: str, indent: int) -> None:
            nonlocal current_part
            for child in self.children(node_id):
                if child.layer > depth:
                    continue
                if child.kind == CHAPTER and child.part != current_part:
                    current_part = child.part
                    if child.part:
                        lines.append(f"  {self.nodes[child.part].title}")
                pages = f"p.{child.pages[0]}-{child.pages[1]}" if child.pages else ""
                mark = "~" if child.virtual else " "
                extra = "  (contents)" if child.role == "contents" else ""
                pad = "  " * (indent + (1 if child.part else 0))
                lines.append(f"{pad}{mark}{child.title[:width]}  [{pages} {child.tokens:,} tok]{extra}")
                walk(child.id, indent + 1)

        walk(ROOT, 1)
        return "\n".join(lines)

    # ------------------------------------------------------------- save / load
    def to_dict(self) -> Dict:
        return {"version": self.version, "title": self.title, "source": self.source, "kind": self.kind,
                "info": self.info, "warnings": self.warnings,
                "nodes": [asdict(n) for n in self.walk_all()]}

    def walk_all(self) -> List[TreeNode]:
        seen = [n for n in self.walk()]
        ids = {n.id for n in seen}
        return seen + [n for n in self.nodes.values() if n.id not in ids]  # parts

    @classmethod
    def from_dict(cls, data: Dict) -> "DocTree":
        nodes = {d["id"]: TreeNode(**d) for d in data["nodes"]}
        return cls(nodes, data["title"], data.get("source"), data.get("kind", "document"),
                   data.get("info", {}), data.get("warnings", []), data.get("version", 0))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False))

    @classmethod
    def load(cls, path: str) -> "DocTree":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ===================================================================== building

def _pages_of(node) -> List[int]:
    pages = list(node.metadata.get("pages") or [])
    if node.page:
        pages.append(node.page)
    return pages


class _Builder:
    def __init__(self, graph: DocumentGraph, counter=count_tokens):
        self.g = graph
        self.count = counter
        self.nodes: Dict[str, TreeNode] = {}
        self.graph_to_tree: Dict[str, str] = {}
        self._chars: Dict[str, int] = {}
        self.n_parts = self.n_chapters = 0
        self.wrapper_titles: List[str] = []
        self.lifted: set = set()   # schedules cut out of a chapter, built as chapters of their own
        self.origin: Dict[str, str] = {}  # top item -> title of the wrapper heading it came from

    # ---- helpers
    def is_passage(self, nid: str) -> bool:
        return self.g.nodes[nid].node_type in PASSAGE_TYPES

    def content_chars(self, nid: str) -> int:
        if nid not in self._chars:
            node = self.g.nodes[nid]
            if node.node_type in PASSAGE_TYPES:
                self._chars[nid] = len(node.text)
            else:
                self._chars[nid] = sum(self.content_chars(c.id) for c in self.g.children(nid))
        return self._chars[nid]

    def add(self, node: TreeNode, parent: Optional[TreeNode]) -> TreeNode:
        self.nodes[node.id] = node
        if parent is not None:
            node.parent = parent.id
            parent.children.append(node.id)
        return node

    def passage(self, nid: str, heading: Optional[str] = None) -> TreeNode:
        gnode = self.g.nodes[nid]
        sentences = [[c.id, c.text] for c in self.g.children(nid) if c.node_type in (SENTENCE, CLAUSE)]
        if not sentences:
            sentences = [[nid, gnode.text]]
        pages = _pages_of(gnode)
        p = TreeNode(nid, PASSAGE, gnode.title or gnode.node_type, text=gnode.text, sentences=sentences,
                     number=gnode.metadata.get("number"), heading=heading,
                     pages=[min(pages), max(pages)] if pages else None, span=[gnode.start, gnode.end],
                     tokens=self.count(gnode.text), metadata={"type": gnode.node_type})
        self.graph_to_tree[nid] = nid
        for sid, _ in sentences:
            self.graph_to_tree[sid] = nid
        return p

    def flat_passages(self, nid: str, heading: Optional[str] = None,
                      heading_id: Optional[str] = None) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """(passage graph id, deeper heading, its graph id) under a graph node, in order."""
        out = []
        for child in self.g.children(nid):
            if child.id in self.lifted:
                continue
            if child.node_type in PASSAGE_TYPES:
                out.append((child.id, heading, heading_id))
            elif child.is_section:
                out.extend(self.flat_passages(child.id, child.title, child.id))
        return out

    def size(self, items: List[Tuple[str, Optional[str]]]) -> Tuple[int, int]:
        """(pages spanned, tokens) of a passage list."""
        pages = [p for item in items for p in _pages_of(self.g.nodes[item[0]])]
        tokens = sum(self.count(self.g.nodes[item[0]].text) for item in items)
        return (max(pages) - min(pages) + 1 if pages else 0), tokens

    def slices(self, items: List, target_tokens: int) -> List[List]:
        out, current, used = [], [], 0
        for item in items:
            t = self.count(self.g.nodes[item[0]].text)
            if current and used + t > target_tokens:
                out.append(current)
                current, used = [], 0
            current.append(item)
            used += t
        if current:
            out.append(current)
        return out

    # ---- levels
    def subsections(self, section: TreeNode, title: str, items: List, virtual: bool,
                    number: Optional[str] = None, graph_id: Optional[str] = None) -> None:
        _, tokens = self.size(items)
        groups = self.slices(items, TARGET_SUBSECTION_TOKENS) if tokens > MAX_SUBSECTION_TOKENS else [items]
        for k, group in enumerate(groups, 1):
            sid = f"{section.id}.u{len(section.children) + 1}"
            label = title if len(groups) == 1 else f"{title} ({k}/{len(groups)})"
            sub = self.add(TreeNode(sid, SUBSECTION, label, number=number,
                                    virtual=virtual or len(groups) > 1,
                                    metadata={"graph_id": graph_id} if graph_id else {}), section)
            if graph_id and k == 1:
                self.graph_to_tree[graph_id] = sid
            for nid, heading, heading_id in group:
                if heading_id:  # a flattened deeper heading resolves to its subsection
                    self.graph_to_tree.setdefault(heading_id, sid)
                self.add(self.passage(nid, heading), sub)

    def section(self, chapter: TreeNode, gid: Optional[str], loose: Optional[List] = None) -> None:
        """A section from graph section ``gid``, or a virtual one holding loose passages."""
        sid = f"{chapter.id}.s{len(chapter.children) + 1}"
        if gid is None:
            sec = self.add(TreeNode(sid, SECTION, chapter.title, virtual=True), chapter)
            self.subsections(sec, chapter.title, loose, virtual=True)
            return
        g = self.g.nodes[gid]
        sec = self.add(TreeNode(sid, SECTION, g.title or "", number=g.metadata.get("number"),
                                metadata={"graph_id": gid}), chapter)
        self.graph_to_tree[gid] = sid
        run: List = []
        for child in self.g.children(gid):
            if child.id in self.lifted:
                continue
            if child.node_type in PASSAGE_TYPES:
                run.append((child.id, None, None))
                continue
            if not child.is_section:
                continue
            if run:
                self.subsections(sec, sec.title, run, virtual=True)
                run = []
            self.subsections(sec, child.title or "", self.flat_passages(child.id),
                             virtual=False, number=child.metadata.get("number"), graph_id=child.id)
        if run:
            self.subsections(sec, sec.title, run, virtual=True)

    def chapter(self, root: TreeNode, title: str, gid: Optional[str] = None, part: Optional[str] = None,
                loose: Optional[List] = None, virtual: bool = False) -> List[str]:
        """Chapter(s) from graph section ``gid`` or from loose passages. Long chapters
        without sections become several virtual chapters. Returns graph ids of schedules
        found inside, to be lifted to chapters of their own."""
        lifted: List[str] = []
        if gid is not None:
            child_sections = [c for c in self.g.children(gid) if c.is_section]
            if not child_sections:
                loose = [(c.id, None, None) for c in self.g.children(gid) if c.node_type in PASSAGE_TYPES]
        if loose is not None:
            pages, tokens = self.size(loose)
            too_long = pages > MAX_CHAPTER_PAGES if pages else tokens > MAX_CHAPTER_PAGES * TOKENS_PER_PAGE
            if too_long:
                target = TARGET_CHAPTER_PAGES * TOKENS_PER_PAGE
                if pages:  # convert the page target to tokens using this text's density
                    target = max(1, int(tokens * TARGET_CHAPTER_PAGES / pages))
                groups = self.slices(loose, target)
            else:
                groups = [loose]
            for k, group in enumerate(groups, 1):
                label = title if len(groups) == 1 else f"{title} ({k}/{len(groups)})"
                ch = self._new_chapter(root, label, gid if k == 1 else None, part, virtual or len(groups) > 1)
                self.section(ch, None, group)
            return lifted
        ch = self._new_chapter(root, title, gid, part, virtual)
        run: List = []
        for child in self.g.children(gid):
            if child.id in self.lifted:
                continue
            if child.node_type in PASSAGE_TYPES:
                run.append((child.id, None, None))
                continue
            if not child.is_section:
                continue
            if LIFT_TO_CHAPTER.match(child.title or ""):
                self.lifted.add(child.id)
                lifted.append(child.id)
                continue
            if run:
                self.section(ch, None, run)
                run = []
            lifted.extend(self._lift_inside(child.id))
            self.section(ch, child.id)
        if run:
            self.section(ch, None, run)
        return lifted

    def _lift_inside(self, gid: str) -> List[str]:
        """Schedules nested deeper (by font size) are cut out and returned."""
        found = []
        for child in self.g.children(gid):
            if child.is_section:
                if LIFT_TO_CHAPTER.match(child.title or ""):
                    self.lifted.add(child.id)
                    found.append(child.id)
                else:
                    found.extend(self._lift_inside(child.id))
        return found

    def _new_chapter(self, root: TreeNode, title: str, gid: Optional[str], part: Optional[str],
                     virtual: bool) -> TreeNode:
        self.n_chapters += 1
        g = self.g.nodes[gid] if gid else None
        ch = self.add(TreeNode(f"ch{self.n_chapters}", CHAPTER, title, part=part, virtual=virtual,
                               number=g.metadata.get("number") if g else None,
                               role="contents" if CONTENTS_TITLE.match(title.strip()) else "body",
                               metadata={"graph_id": gid} if gid else {}), root)
        if gid:
            self.graph_to_tree[gid] = ch.id
        if part:
            self.nodes[part].children.append(ch.id)
        return ch

    # ---- top level
    def top_items(self) -> List[str]:
        items = [c.id for c in self.g.children(self.g.root.id)]
        total = max(1, self.content_chars(self.g.root.id))
        for _ in range(3):
            wrapper = next((i for i in items if self.g.nodes[i].is_section
                            and self.content_chars(i) >= WRAPPER_SHARE * total
                            and sum(1 for c in self.g.children(i) if c.is_section) >= 2), None)
            if wrapper is None:
                break
            self.wrapper_titles.append(self.g.nodes[wrapper].title or "")
            k = items.index(wrapper)
            children = [c.id for c in self.g.children(wrapper)]
            for c in children:
                self.origin[c] = self.g.nodes[wrapper].title or ""
            items[k:k + 1] = children
        return items

    def build(self) -> TreeNode:
        groot = self.g.root
        root = self.add(TreeNode(ROOT, ROOT, groot.title or "Document",
                                 metadata={"graph_id": groot.id, "parts": []}), None)
        pending: List[str] = []   # schedules lifted out of chapters
        loose: List = []

        def flush_loose() -> None:
            if loose:  # text outside any heading: the title page under a wrapper heading, or front matter
                title = self.origin.get(loose[0][0]) or ("Front matter" if not root.children else "Untitled text")
                self.chapter(root, title, loose=list(loose), virtual=True)
                loose.clear()

        def flush_lifted() -> None:
            while pending:
                gid = pending.pop(0)
                pending.extend(self.chapter(root, self.g.nodes[gid].title or "", gid))

        for item in self.top_items():
            gnode = self.g.nodes[item]
            if gnode.node_type in PASSAGE_TYPES:
                loose.append((item, None, None))
                continue
            if not gnode.is_section:
                continue
            flush_loose()
            title = gnode.title or ""
            sections = [c for c in self.g.children(item) if c.is_section]
            if PART_TITLE.match(title) and sections:
                self.n_parts += 1
                part = TreeNode(f"pt{self.n_parts}", PART, title, metadata={"graph_id": item})
                self.nodes[part.id] = part
                root.metadata["parts"].append(part.id)
                self.graph_to_tree[item] = part.id
                intro = [(c.id, None, None) for c in self.g.children(item) if c.node_type in PASSAGE_TYPES]
                if intro:
                    self.chapter(root, title, part=part.id, loose=intro, virtual=True)
                for sec in sections:
                    pending.extend(self.chapter(root, sec.title or "", sec.id, part=part.id))
                    flush_lifted()
                continue
            pending.extend(self.chapter(root, title, item))
            flush_lifted()
        flush_loose()
        return root


def _finish(tree: DocTree, b: _Builder) -> None:
    """Pages, spans and tokens bottom-up; defined terms and links between passages."""
    def fill(node: TreeNode) -> None:
        if node.kind == PASSAGE:
            return
        kids = [tree.nodes[c] for c in node.children]
        for k in kids:
            fill(k)
        pages = [p for k in kids if k.pages for p in k.pages]
        spans = [s for k in kids if k.span for s in k.span]
        node.pages = [min(pages), max(pages)] if pages else None
        node.span = [min(spans), max(spans)] if spans else None
        node.tokens = sum(k.tokens for k in kids)

    fill(tree.root)
    for pid in tree.root.metadata.get("parts", []):
        part = tree.nodes[pid]
        kids = [tree.nodes[c] for c in part.children]
        pages = [p for k in kids if k.pages for p in k.pages]
        part.pages = [min(pages), max(pages)] if pages else None
        part.tokens = sum(k.tokens for k in kids)

    g = b.g
    for term, def_id in g.definitions.items():
        target = b.graph_to_tree.get(def_id)
        if target in tree.nodes:
            tree.nodes[target].defined_terms.append(term)
    labels = {CROSS_REFERENCE: "refers_to", DEFINITION_LINK: "uses_definition"}
    for edge in g.edges:
        if edge.edge_type not in labels:
            continue
        src, dst = b.graph_to_tree.get(edge.source), b.graph_to_tree.get(edge.target)
        if src in tree.nodes and dst in tree.nodes and src != dst:
            links = tree.nodes[src].links.setdefault(labels[edge.edge_type], [])
            if dst not in links:
                links.append(dst)


_CONTRACT_WORDS = re.compile(r"\b(shall|hereby|herein|hereunder|indemnif\w*|warrant\w*|party|parties|"
                             r"agreement|termination|governing law)\b", re.I)


def guess_kind(tree: DocTree) -> str:
    words = sum(len(p.text.split()) for p in tree.passages()) or 1
    hits = sum(len(_CONTRACT_WORDS.findall(p.text)) for p in tree.passages())
    if hits / words > 0.008 or re.search(r"\b(agreement|contract)\b", tree.title, re.I):
        return "contract"
    if (tree.root.pages and tree.root.pages[1] >= 30) or len(tree.chapters()) >= 8:
        return "book"
    return "document"


def build_tree(graph: DocumentGraph, source: Optional[str] = None, info: Optional[Dict] = None,
               counter=count_tokens) -> DocTree:
    """The document tree for ``graph`` (see module docstring)."""
    b = _Builder(graph, counter)
    root = b.build()
    title = b.wrapper_titles[0] if b.wrapper_titles else root.title
    root.title = title
    tree = DocTree(b.nodes, title, source, info=dict(info or {}), warnings=list(graph.warnings))
    if b.wrapper_titles:
        tree.root.metadata["wrapper_titles"] = b.wrapper_titles
    # Heading texts (incl. a document-title heading) count as document text for verification.
    headings = [n.title for n in graph.nodes.values() if n.is_section and n.title]
    if graph.root.title and graph.root.title in graph.source_text:
        headings.insert(0, graph.root.title)
    tree.root.metadata["headings"] = headings
    _finish(tree, b)
    tree.kind = guess_kind(tree)
    return tree


def tree_from_file(path: str, parser: str = "native") -> DocTree:
    from .ingest import load_document

    doc = load_document(path, parser)
    graph = DocumentGraph(doc.nodes, source_text=doc.text, warnings=doc.warnings)
    info = dict(doc.info)
    info.setdefault("page_count", doc.page_count)
    return build_tree(graph, source=str(path), info=info)
