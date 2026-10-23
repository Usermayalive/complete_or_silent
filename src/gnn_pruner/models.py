"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Node types produced by the parser.
DOCUMENT = "document"
CHAPTER = "chapter"
SECTION = "section"
SUBSECTION = "subsection"
PARAGRAPH = "paragraph"
CLAUSE = "clause"
SENTENCE = "sentence"
TABLE = "table"
FIGURE = "figure"

SECTION_TYPES = (CHAPTER, SECTION, SUBSECTION)
CONTENT_TYPES = (PARAGRAPH, CLAUSE, SENTENCE, TABLE, FIGURE)
STRUCTURAL_TYPES = (DOCUMENT,) + SECTION_TYPES

# Edge types.
PARENT_CHILD = "parent_child"
NEXT_PREV = "next_prev"
SAME_SECTION = "same_section"
CROSS_REFERENCE = "cross_reference"
DEFINITION_LINK = "definition_link"
SEMANTIC_SIMILARITY = "semantic_similarity"


@dataclass
class Node:
    id: str
    node_type: str
    text: str
    parent_id: Optional[str] = None
    level: int = 0
    title: Optional[str] = None
    start: int = 0
    end: int = 0
    page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_content(self) -> bool:
        return self.node_type in CONTENT_TYPES

    @property
    def is_section(self) -> bool:
        return self.node_type in SECTION_TYPES


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    edge_type: str
    weight: float = 1.0
