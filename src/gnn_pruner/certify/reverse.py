"""U3. Reverse index: for every node, everything that points AT it (no model).

``pointing_at(node)`` returns the qualifier links whose target is the node, one of its
ancestors below the chapter (a link to "Section 12" also concerns 12.3), or one of its
descendants (a link to 12.3(b) concerns the 12.3 the reader holds). This is how the
exception 3,000 pages away is found by lookup instead of by reading forward.
"""

from __future__ import annotations

from typing import Dict, List, Set

from ..tree import CHAPTER, ROOT, DocTree
from .qualifiers import QualifierGraph, QualifierLink


class ReverseIndex:
    def __init__(self, tree: DocTree, graph: QualifierGraph):
        self.tree = tree
        self.graph = graph
        self._by_target: Dict[str, List[QualifierLink]] = {}
        for link in graph.links:
            for target in link.targets:
                self._by_target.setdefault(target, []).append(link)

    def _related(self, node_id: str) -> Set[str]:
        related = {node_id}
        node = self.tree.node(node_id)
        while node.parent not in (None, ROOT) and node.kind != CHAPTER:
            node = self.tree.node(node.parent)
            if node.kind != CHAPTER:
                related.add(node.id)
        related.update(n.id for n in self.tree.walk(node_id))
        return related

    def pointing_at(self, node_id: str) -> List[QualifierLink]:
        out, seen = [], set()
        related = self._related(node_id)
        for nid in related:
            for link in self._by_target.get(nid, []):
                key = id(link)
                if key not in seen and link.passage not in related:
                    seen.add(key)
                    out.append(link)
        return sorted(out, key=lambda l: l.sentence)

    def targets(self) -> int:
        return len(self._by_target)
