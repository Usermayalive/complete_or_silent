"""Phase 2: the document graph ("document memory").

Nodes come from the parser (document > chapter > section > subsection >
paragraph / clause / table / figure > sentence / clause). Edges:

- parent_child       hierarchy
- next_prev          reading order between siblings
- same_section       content units that share a section
- cross_reference    "see Section 4.2" / "Sections 4.1 and 4.2" / "clause 2(c)"
- definition_link    usage of a defined term -> its definition
- semantic_similarity added by the index once embeddings exist

Graphs serialise to plain dicts / JSON (``save``/``load``) and can be put in a
SQLite ``GraphStore`` or exported to networkx.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import (
    CLAUSE,
    CROSS_REFERENCE,
    DEFINITION_LINK,
    DOCUMENT,
    NEXT_PREV,
    PARAGRAPH,
    PARENT_CHILD,
    SAME_SECTION,
    SECTION_TYPES,
    SEMANTIC_SIMILARITY,
    SENTENCE,
    Edge,
    Node,
)
from .parser import _CLAUSE_LABEL, normalize_ws, parse_text, split_sentences

FORMAT_VERSION = 1
# Units that carry references and definitions (sentences are covered by their paragraph).
REFERENCE_UNITS = (PARAGRAPH, CLAUSE)
MAX_SAME_SECTION_UNITS = 25

_XREF = re.compile(
    r"\b(?:sections?|clauses?|articles?|paragraphs?|§§?)\s*"
    r"(\d+(?:\.\d+)*(?:\([a-z0-9]{1,3}\))*"
    r"(?:\s*(?:,|and|or|&)\s*\d+(?:\.\d+)*(?:\([a-z0-9]{1,3}\))*)*)",
    re.I,
)
_XREF_NUMBER = re.compile(r"\d+(?:\.\d+)*(?:\([a-z0-9]{1,3}\))*", re.I)
_DEFINITION = re.compile(
    r"^(?:the\s+terms?\s+)?[\"'“]?([A-Za-z][\w\s\-]{0,40}?)[\"'”]?\s*"
    r"(?:\([^)]*\)\s*)?"
    r"(?:means|shall mean|refers to|is defined as|has the meaning)\b",
    re.I,
)
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.I)


class DocumentGraph:
    def __init__(self, nodes: Sequence[Node], source_text: str = "",
                 warnings: Optional[List[str]] = None,
                 edges: Optional[Sequence[Edge]] = None,
                 definitions: Optional[Dict[str, str]] = None):
        """Build edges from ``nodes``, or restore them when ``edges`` is given."""
        self.warnings: List[str] = list(warnings or [])
        self.nodes: Dict[str, Node] = {n.id: n for n in nodes}
        self.order: List[str] = [n.id for n in nodes]
        self.source_text = source_text
        self.edges: List[Edge] = []
        self.definitions: Dict[str, str] = dict(definitions or {})  # term -> node id
        # node -> [(neighbour, edge_type, weight, outgoing)]
        self._adj: Dict[str, List[Tuple[str, str, float, bool]]] = defaultdict(list)
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._numbered: Dict[str, str] = {}  # "4.2" / "5(a)" -> section or clause id
        self._index_structure()
        if edges is None:
            self._build_edges()
        else:
            for e in edges:
                self.add_edge(e.source, e.target, e.edge_type, e.weight)

    @classmethod
    def from_text(cls, text: str, title: str = "Document") -> "DocumentGraph":
        return cls(parse_text(text, title=title), source_text=text)

    @classmethod
    def from_file(cls, path: str, parser: str = "native") -> "DocumentGraph":
        """Load any supported format (.md/.txt/.pdf/.docx/.html); ``parser="docling"``
        uses Docling instead of the native loaders."""
        from .ingest import load_document

        doc = load_document(path, parser)
        return cls(doc.nodes, source_text=doc.text, warnings=doc.warnings)

    # ------------------------------------------------------------------ build

    def add_edge(self, source: str, target: str, edge_type: str, weight: float = 1.0) -> None:
        """Add a directed edge; adjacency is indexed from both ends."""
        if source == target:
            return
        self.edges.append(Edge(source, target, edge_type, weight))
        self._adj[source].append((target, edge_type, weight, True))
        self._adj[target].append((source, edge_type, weight, False))

    def _index_structure(self) -> None:
        for nid in self.order:
            node = self.nodes[nid]
            if node.parent_id is not None:
                self._children[node.parent_id].append(nid)
            number = node.metadata.get("number")
            if number and (node.is_section or node.node_type == CLAUSE):
                self._numbered.setdefault(number, nid)

    def _build_edges(self) -> None:
        for nid in self.order:
            node = self.nodes[nid]
            if node.parent_id is not None:
                self.add_edge(node.parent_id, nid, PARENT_CHILD)

        for siblings in self._children.values():
            for a, b in zip(siblings, siblings[1:]):
                self.add_edge(a, b, NEXT_PREV)

        self._link_same_section()
        self._link_cross_references()
        self._link_definitions()

    def _link_same_section(self) -> None:
        for parent_id, child_ids in self._children.items():
            if not self.nodes[parent_id].is_section:
                continue
            units = [c for c in child_ids if self.nodes[c].is_content]
            if len(units) > MAX_SAME_SECTION_UNITS:
                continue  # large sections: rely on parent_child / next_prev
            for i, a in enumerate(units):
                for b in units[i + 1:]:
                    self.add_edge(a, b, SAME_SECTION)

    def resolve_reference(self, number: str) -> Optional[Node]:
        """"4.2" -> section 4.2; "5(a)" -> clause 5(a), falling back to section 5."""
        while number:
            nid = self._numbered.get(number)
            if nid:
                return self.nodes[nid]
            if "(" not in number:
                return None
            number = number[:number.rindex("(")]
        return None

    def _link_cross_references(self) -> None:
        for node in self.content_nodes(REFERENCE_UNITS):
            enclosing = {a.id for a in self.ancestors(node.id)} | {node.id}
            targets = set()
            for match in _XREF.finditer(node.text):
                for number in _XREF_NUMBER.findall(match.group(1)):
                    target = self.resolve_reference(number.lower())
                    if target is not None and target.id not in enclosing \
                            and target.parent_id != node.id:
                        targets.add(target.id)
            for target_id in sorted(targets):
                self.add_edge(node.id, target_id, CROSS_REFERENCE)

    def _link_definitions(self) -> None:
        for node in self.content_nodes(REFERENCE_UNITS):
            text = _CLAUSE_LABEL.sub("", node.text, count=1)
            for sentence, _, _ in split_sentences(text):
                match = _DEFINITION.match(sentence)
                if not match:
                    continue
                term = _LEADING_ARTICLE.sub("", match.group(1)).strip().lower()
                if term and len(term.split()) <= 5:
                    self.definitions.setdefault(term, node.id)
        for term, def_id in self.definitions.items():
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.I)
            def_node = self.nodes[def_id]
            for node in self.content_nodes(REFERENCE_UNITS):
                related = node.id in (def_id, def_node.parent_id) or node.parent_id == def_id
                if not related and pattern.search(node.text):
                    self.add_edge(node.id, def_id, DEFINITION_LINK)

    def remove_edges(self, edge_type: str) -> None:
        self.edges = [e for e in self.edges if e.edge_type != edge_type]
        for nid, adj in self._adj.items():
            self._adj[nid] = [a for a in adj if a[1] != edge_type]

    def add_semantic_edges(self, pairs: Iterable[Tuple[str, str, float]]) -> None:
        for a, b, sim in pairs:
            self.add_edge(a, b, SEMANTIC_SIMILARITY, weight=float(sim))

    # ---------------------------------------------------------------- queries

    @property
    def root(self) -> Optional[Node]:
        return self.nodes.get(self.order[0]) if self.order else None

    @property
    def title(self) -> str:
        return (self.root.title or "") if self.root else ""

    def lookup(self, title: str) -> Optional[Node]:
        """Find a section by title (exact, case-insensitive; then substring)."""
        wanted = title.strip().lower()
        sections = [n for n in self.nodes.values() if n.is_section or n.node_type == DOCUMENT]
        for node in sections:
            if (node.title or "").lower() == wanted:
                return node
        for node in sections:
            if wanted in (node.title or "").lower():
                return node
        return None

    def section_by_number(self, number: str) -> Optional[Node]:
        node = self.resolve_reference(number)
        return node if node is not None and node.is_section else None

    def sections(self) -> List[Node]:
        return [self.nodes[nid] for nid in self.order if self.nodes[nid].is_section]

    def parent(self, node_id: str) -> Optional[Node]:
        pid = self.nodes[node_id].parent_id
        return self.nodes[pid] if pid else None

    def children(self, node_id: str) -> List[Node]:
        return [self.nodes[c] for c in self._children.get(node_id, [])]

    def ancestors(self, node_id: str) -> List[Node]:
        """Ancestors nearest-first, ending at the document root."""
        out = []
        current = self.parent(node_id)
        while current is not None:
            out.append(current)
            current = self.parent(current.id)
        return out

    def section_of(self, node_id: str) -> Optional[Node]:
        for anc in self.ancestors(node_id):
            if anc.is_section:
                return anc
        return None

    def neighbors(self, node_id: str, edge_types: Optional[Iterable[str]] = None,
                  direction: str = "both") -> List[Tuple[Node, str, float]]:
        """Neighbours via ``edge_types``; ``direction`` is "out", "in" or "both"."""
        allowed = set(edge_types) if edge_types is not None else None
        return [(self.nodes[t], et, w) for t, et, w, outgoing in self._adj.get(node_id, [])
                if (allowed is None or et in allowed)
                and (direction == "both" or outgoing == (direction == "out"))]

    def content_nodes(self, types: Iterable[str]) -> List[Node]:
        wanted = set(types)
        return [self.nodes[nid] for nid in self.order if self.nodes[nid].node_type in wanted]

    def path(self, node_id: str) -> str:
        """Human-readable section path used for citations."""
        node = self.nodes[node_id]
        chain = [a for a in reversed(self.ancestors(node_id)) if a.is_section]
        if node.is_section:
            chain.append(node)
        return " > ".join(a.title or a.text for a in chain) or self.title

    # ------------------------------------------------------ inspection / checks

    def stats(self) -> Dict[str, Dict[str, int]]:
        return {
            "nodes": dict(Counter(n.node_type for n in self.nodes.values())),
            "edges": dict(Counter(e.edge_type for e in self.edges)),
        }

    def validate(self) -> List[str]:
        """Structural integrity problems; an empty list means the graph is sound."""
        problems = []
        roots = [n for n in self.nodes.values() if n.parent_id is None]
        if len(roots) != 1 or roots[0].node_type != DOCUMENT:
            problems.append(f"expected one document root, found {[r.id for r in roots]}")
        for node in self.nodes.values():
            if node.parent_id is not None and node.parent_id not in self.nodes:
                problems.append(f"{node.id}: missing parent {node.parent_id}")
                continue
            seen = {node.id}
            for anc in self.ancestors(node.id):
                if anc.id in seen:
                    problems.append(f"{node.id}: cycle in ancestry")
                    break
                seen.add(anc.id)
            if self.source_text and not 0 <= node.start <= node.end <= len(self.source_text):
                problems.append(f"{node.id}: offsets {node.start}-{node.end} out of range")
            elif self.source_text and node.node_type in (PARAGRAPH, CLAUSE, SENTENCE):
                if normalize_ws(self.source_text[node.start:node.end]) != node.text:
                    problems.append(f"{node.id}: text does not match source offsets")
            parent = self.nodes.get(node.parent_id) if node.parent_id else None
            if parent is not None and parent.is_content and not node.is_content:
                problems.append(f"{node.id}: {node.node_type} nested under {parent.node_type}")
        for e in self.edges:
            if e.source not in self.nodes or e.target not in self.nodes:
                problems.append(f"dangling {e.edge_type} edge {e.source}->{e.target}")
        for term, nid in self.definitions.items():
            if nid not in self.nodes:
                problems.append(f"definition {term!r} points to missing node {nid}")
        return problems

    def outline(self, max_text: int = 70) -> str:
        """Indented tree of the document, for humans."""
        lines = []

        def walk(node: Node, depth: int) -> None:
            label = normalize_ws(node.title if node.title else node.text)
            label = label if len(label) <= max_text else label[:max_text - 1] + "…"
            extra = []
            if node.page:
                extra.append(f"p.{node.page}")
            if node.metadata.get("number") and not node.is_section:
                extra.append(node.metadata["number"])
            suffix = f"  ({', '.join(extra)})" if extra else ""
            lines.append(f"{'  ' * depth}[{node.node_type}] {label}{suffix}")
            for child in self.children(node.id):
                if child.node_type != SENTENCE:
                    walk(child, depth + 1)

        if self.root:
            walk(self.root, 0)
        return "\n".join(lines)

    # ---------------------------------------------------------- serialisation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "source_text": self.source_text,
            "warnings": self.warnings,
            "definitions": self.definitions,
            "nodes": [asdict(self.nodes[nid]) for nid in self.order],
            "edges": [asdict(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentGraph":
        version = data.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported graph format version {version!r}")
        return cls([Node(**n) for n in data["nodes"]], source_text=data["source_text"],
                   warnings=data.get("warnings"), edges=[Edge(**e) for e in data["edges"]],
                   definitions=data.get("definitions"))

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "DocumentGraph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_networkx(self):
        """Export as a ``networkx.MultiDiGraph`` (requires networkx)."""
        import networkx as nx

        g = nx.MultiDiGraph(title=self.title)
        for nid in self.order:
            node = self.nodes[nid]
            g.add_node(nid, node_type=node.node_type, text=node.text, title=node.title,
                       page=node.page, start=node.start, end=node.end)
        for e in self.edges:
            g.add_edge(e.source, e.target, key=e.edge_type, edge_type=e.edge_type,
                       weight=e.weight)
        return g
