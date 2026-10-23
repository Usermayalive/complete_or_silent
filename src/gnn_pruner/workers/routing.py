"""Routing (WORKFLOW B1): which chapters a question names, and what to look for in them.

A question like "On what date was the Content License Agreement of WebmdHealthCorp made?"
does two jobs:
- it names a unit ("Content License Agreement of WebmdHealthCorp")
- it asks for content ("date ... made")

Mixing them makes title words dominate the search: the step-2 benchmark found our pipeline
reaching the right contract but the wrong clause. Here the two are separated.

**Names.** A title is named when the question contains all of its *distinctive* words.
Those are words that appear in few titles: "webmdhealthcorp" counts, "agreement" does not.
A named part (a book in a library) names all of its chapters.

**Content.** The question minus the words of the titles it names. Workers search with it.

**Scope** (R2, R9):
- ``single``: one fact
- ``multi``: several facts or a comparison / reason
- ``global``: a summary or list over the whole document or unit

It sets how many chapters are opened and how much evidence the answer gets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from ..textutil import stem

_WORD = re.compile(r"[a-z0-9]+")
STOP = set("a an the of to in on for and or is are was were be by with that this it its as at from what who "
           "whom which how why when where did does do under any".split())
COMMON_SHARE = 0.2      # a title word in more than this share of titles is not distinctive


def words(text: str) -> List[str]:
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in STOP]


@dataclass
class Route:
    chapters: List[str] = field(default_factory=list)     # chapter ids the question names
    named: List[str] = field(default_factory=list)        # the titles matched
    content: List[str] = field(default_factory=list)      # content words to search with


def _distinctive(titles: Dict[str, List[str]]) -> Dict[str, set]:
    df: Dict[str, int] = {}
    for ws in titles.values():
        for w in set(ws):
            df[w] = df.get(w, 0) + 1
    n = max(1, len(titles))
    out = {}
    for node_id, ws in titles.items():
        rare = {w for w in ws if df[w] / n <= COMMON_SHARE or n <= 3}
        out[node_id] = rare or set(ws)
    return out


def route(tree, question: str) -> Route:
    q = set(words(question))
    # Placeholder chapters (title page, front matter) carry the document's own title; a
    # question naming the document does not name them. Slices "(k/n)" of a real unit do count.
    units = {c.id: c.title for c in tree.chapters()
             if not c.virtual or re.search(r"\(\d+/\d+\)$", c.title)}
    units.update({p.id: p.title for p in tree.parts()})
    # Titles without "Book:" prefixes or slice markers "(k/n)" (a slice is matched by its unit's title).
    clean = lambda t: re.sub(r"\s*\(\d+/\d+\)$", "", re.sub(r"^(book|part|volume)\s*:\s*", "", t, flags=re.I))
    title_words = {nid: words(clean(t)) for nid, t in units.items()}
    distinctive = _distinctive(title_words)
    hits = [nid for nid, need in distinctive.items() if need and need <= q]
    # Prefer the most specific match: a hit whose distinctive words contain another hit's.
    hits = [h for h in hits if not any(h != o and distinctive[h] < distinctive[o] for o in hits)]
    r = Route()
    for nid in hits:
        node = tree.node(nid)
        r.named.append(node.title)
        ids = node.children if node.kind == "part" else [nid]
        # A long chapter split into slices "(k/n)" is one unit: name every slice.
        if node.kind == "chapter" and re.search(r"\(\d+/\d+\)$", node.title):
            base = re.sub(r"\s*\(\d+/\d+\)$", "", node.title)
            ids = [c.id for c in tree.chapters() if re.sub(r"\s*\(\d+/\d+\)$", "", c.title) == base]
        r.chapters.extend(i for i in ids if i not in r.chapters)
    used = {w for nid in hits for w in title_words[nid]}
    r.content = [w for w in words(question) if w not in used]
    return r


_GLOBAL = re.compile(r"\b(summar\w*|overall|throughout|in general|main (?:themes?|points?|events?|characters?)|"
                     r"list (?:all|every|each|the)|all (?:of )?the \w+s\b|every \w+|each of the|how many|"
                     r"over the course|across the|how does \w+(?: \w+)? change|in the end)\b", re.I)
_MULTI = re.compile(r"\b(compare|comparison|difference|differ|between|both|relationship|why|how did|how does|"
                    r"what happens (?:to|when|if)|and (?:what|how|when|who|where|why))\b", re.I)


def scope(question: str) -> str:
    if _GLOBAL.search(question):
        return "global"
    if _MULTI.search(question):
        return "multi"
    return "single"
