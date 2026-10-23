"""G1: the spec's four-layer document DAG (spec §2.1), as a view over DocumentGraph.

    L1  section headings (chapters)        section / chapter nodes under the root
    L2  sub-section headings               any deeper heading (flattened)
    L3  paragraph blocks                   paragraph / table / figure
    L4  atomic claims                      sentence / clause

Edges (spec): hierarchical (parent -> child), sequential (next in document order within
the same layer) and reference (cross-references and defined-term links).

Design decision D8: a paragraph without sentences or clauses gets one *virtual* L4 node
with the same text, so every L3 node has atomic claims (the GNN coarsens L4 -> L3).
Levels a document lacks stay empty (a paragraph directly under a section has an L1
parent). The document root is not a layer.

``validate_and_repair`` is the spec's GVRM step (decision D7): it checks the layering
invariants and repairs what it can, recording every repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .document_graph import DocumentGraph
from .models import (CHAPTER, CLAUSE, CROSS_REFERENCE, DEFINITION_LINK, DOCUMENT, FIGURE,
                     PARAGRAPH, SECTION, SENTENCE, SUBSECTION, TABLE)

HIERARCHICAL = "hierarchical"
SEQUENTIAL = "sequential"
REFERENCE = "reference"
LAYERS = (1, 2, 3, 4)
VIRTUAL_SUFFIX = "#0"


@dataclass
class LayerNode:
    id: str
    layer: int
    node_type: str
    text: str
    parent_id: Optional[str]
    position: int                 # document order
    title: Optional[str] = None
    number: Optional[str] = None
    page: Optional[int] = None
    start: int = 0
    end: int = 0
    virtual: bool = False         # synthesised L4 for a paragraph without sentences


@dataclass(frozen=True)
class LayerEdge:
    source: str
    target: str
    edge_type: str                # hierarchical | sequential | reference
    subtype: str = ""             # reference: cross_reference | definition_link


@dataclass
class LayeredGraph:
    doc_id: str
    title: str
    nodes: Dict[str, LayerNode]
    edges: List[LayerEdge]
    repairs: List[str] = field(default_factory=list)
    _children: Optional[Dict[Optional[str], List[str]]] = field(default=None, repr=False)

    def layer(self, n: int) -> List[LayerNode]:
        return sorted((x for x in self.nodes.values() if x.layer == n), key=lambda x: x.position)

    def children(self, node_id: str) -> List[LayerNode]:
        if self._children is None:
            index: Dict[Optional[str], List[str]] = {}
            for x in sorted(self.nodes.values(), key=lambda x: (x.position, x.layer)):
                index.setdefault(x.parent_id, []).append(x.id)
            self._children = index
        return [self.nodes[c] for c in self._children.get(node_id, [])]

    def invalidate(self) -> None:
        """Call after changing parent links."""
        self._children = None

    def ancestors(self, node_id: str) -> List[LayerNode]:
        out, current = [], self.nodes[node_id].parent_id
        while current is not None:
            out.append(self.nodes[current])
            current = self.nodes[current].parent_id
        return out

    def path(self, node_id: str) -> List[str]:
        """Ancestors from L1 down to the node."""
        return [a.id for a in reversed(self.ancestors(node_id))] + [node_id]

    def leaves_of(self, node_id: str) -> List[LayerNode]:
        node = self.nodes[node_id]
        if node.layer == 4:
            return [node]
        return [leaf for child in self.children(node_id) for leaf in self.leaves_of(child.id)]

    def stats(self) -> Dict[str, int]:
        out = {f"L{n}": sum(1 for x in self.nodes.values() if x.layer == n) for n in LAYERS}
        out["virtual_L4"] = sum(1 for x in self.nodes.values() if x.virtual)
        for kind in (HIERARCHICAL, SEQUENTIAL, REFERENCE):
            out[kind] = sum(1 for e in self.edges if e.edge_type == kind)
        return out


def _layer_of(graph: DocumentGraph, node_id: str) -> Optional[int]:
    node = graph.nodes[node_id]
    if node.node_type == DOCUMENT:
        return None
    if node.node_type in (SENTENCE, CLAUSE):
        return 4
    if node.node_type in (PARAGRAPH, TABLE, FIGURE):
        return 3
    if node.node_type in (CHAPTER, SECTION, SUBSECTION):
        parent = graph.parent(node_id)
        while parent is not None and not parent.is_section:
            parent = graph.parent(parent.id)
        return 1 if parent is None else 2
    return None


def four_layer(graph: DocumentGraph, doc_id: str = "doc") -> LayeredGraph:
    """The four-layer view of ``graph`` (validated and repaired)."""
    position = {nid: i for i, nid in enumerate(graph.order)}
    layer_of = {nid: _layer_of(graph, nid) for nid in graph.order}
    nodes: Dict[str, LayerNode] = {}
    for nid in graph.order:
        layer = layer_of[nid]
        if layer is None:
            continue
        node = graph.nodes[nid]
        # Parent in the layered tree: the nearest ancestor on a higher (smaller) layer.
        parent = graph.parent(nid)
        while parent is not None and (layer_of.get(parent.id) or 99) >= layer:
            parent = graph.parent(parent.id)
        nodes[nid] = LayerNode(nid, layer, node.node_type, node.text,
                               parent.id if parent is not None and layer_of.get(parent.id) else None,
                               position[nid], node.title, node.metadata.get("number"), node.page,
                               node.start, node.end)
    # Virtual L4 for paragraphs without atomic claims (D8).
    has_claims = {n.parent_id for n in nodes.values() if n.layer == 4}
    for nid, node in list(nodes.items()):
        if node.layer == 3 and nid not in has_claims:
            nodes[nid + VIRTUAL_SUFFIX] = LayerNode(
                nid + VIRTUAL_SUFFIX, 4, SENTENCE, node.text, nid, node.position,
                page=node.page, start=node.start, end=node.end, virtual=True)
    layered = LayeredGraph(doc_id, graph.title, nodes, [])
    layered.repairs = validate_and_repair(layered)
    layered.invalidate()
    layered.edges = _edges(graph, layered)
    return layered


def _edges(graph: DocumentGraph, layered: LayeredGraph) -> List[LayerEdge]:
    edges: List[LayerEdge] = []
    for node in sorted(layered.nodes.values(), key=lambda x: (x.position, x.layer)):
        if node.parent_id is not None:
            edges.append(LayerEdge(node.parent_id, node.id, HIERARCHICAL))
    for n in LAYERS:
        ordered = layered.layer(n)
        edges += [LayerEdge(a.id, b.id, SEQUENTIAL) for a, b in zip(ordered, ordered[1:])]
    seen = set()
    for e in graph.edges:
        if e.edge_type not in (CROSS_REFERENCE, DEFINITION_LINK):
            continue
        if e.source in layered.nodes and e.target in layered.nodes and (e.source, e.target) not in seen:
            seen.add((e.source, e.target))
            edges.append(LayerEdge(e.source, e.target, REFERENCE, e.edge_type))
    return edges


def validate_and_repair(layered: LayeredGraph) -> List[str]:
    """GVRM: every node's parent sits on a higher layer; L4 hangs off an L3; positions
    are unique per layer; offsets are ordered. Repairs are applied and returned."""
    repairs: List[str] = []
    nodes = layered.nodes
    for node in sorted(nodes.values(), key=lambda x: x.position):
        parent = nodes.get(node.parent_id) if node.parent_id else None
        if node.parent_id is not None and parent is None:
            repairs.append(f"{node.id}: parent {node.parent_id} missing; detached")
            node.parent_id = None
            parent = None
        if parent is not None and parent.layer >= node.layer:
            repairs.append(f"{node.id}: parent {parent.id} on layer {parent.layer}; re-attached")
            node.parent_id = _previous_on_layer_below(layered, node)
        if node.layer == 4 and (node.parent_id is None or nodes[node.parent_id].layer != 3):
            target = _previous_on_layer(layered, node, 3)
            if target is not None:
                repairs.append(f"{node.id}: L4 without an L3 parent; attached to {target}")
                node.parent_id = target
        if node.end < node.start:
            repairs.append(f"{node.id}: end {node.end} < start {node.start}; swapped")
            node.start, node.end = node.end, node.start
    return repairs


def _previous_on_layer(layered: LayeredGraph, node: LayerNode, layer: int) -> Optional[str]:
    # Linear scan: only called while repairing, which is rare.
    candidates = [x for x in layered.nodes.values() if x.layer == layer and x.position <= node.position
                  and x.id != node.id]
    return max(candidates, key=lambda x: x.position).id if candidates else None


def _previous_on_layer_below(layered: LayeredGraph, node: LayerNode) -> Optional[str]:
    for layer in range(node.layer - 1, 0, -1):
        found = _previous_on_layer(layered, node, layer)
        if found is not None:
            return found
    return None


def check(layered: LayeredGraph) -> List[str]:
    """Invariant violations (empty when the graph is a valid four-layer DAG)."""
    problems = []
    for node in layered.nodes.values():
        if node.layer not in LAYERS:
            problems.append(f"{node.id}: layer {node.layer}")
        if node.parent_id is not None:
            parent = layered.nodes.get(node.parent_id)
            if parent is None or parent.layer >= node.layer:
                problems.append(f"{node.id}: bad parent {node.parent_id}")
        if node.layer == 3 and not any(c.layer == 4 for c in layered.children(node.id)):
            problems.append(f"{node.id}: L3 without L4")
        if node.layer == 4 and (node.parent_id is None or layered.nodes[node.parent_id].layer != 3):
            problems.append(f"{node.id}: L4 not under L3")
    return problems


def edge_index(layered: LayeredGraph) -> Tuple[List[str], Dict[str, int]]:
    """Stable node order (document order, then layer) and id -> row index, for tensors."""
    ids = [n.id for n in sorted(layered.nodes.values(), key=lambda x: (x.position, x.layer, x.id))]
    return ids, {nid: i for i, nid in enumerate(ids)}
