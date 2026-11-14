"""Build step 3: routing, the central board, chapter workers and the orchestrator.

No network: a scripted fake model plays the workers and the answer step.
"""

import json
import re

from gnn_pruner.answer import ANSWER_SCHEMA, Completion
from gnn_pruner.document_graph import DocumentGraph
from gnn_pruner.tree import build_tree
from gnn_pruner.workers import ask
from gnn_pruner.workers.board import Board
from gnn_pruner.workers.gate import REWRITE_SCHEMA, TOC_SCHEMA, Gate
from gnn_pruner.workers.routing import route, scope
from gnn_pruner.workers.worker import SentenceIndex

ROOM = """# Contract data room

## Supply Agreement of ACME

### 1. Definitions

"Widgets" means the goods listed in Schedule 1.

### 2. Law

This Agreement is governed by the laws of Ontario.

## Lease of BETA

### 1. Rent

The tenant pays rent of 900 USD each month.

### 2. Law

This lease is governed by the laws of Texas. The Widgets are stored on site.

## Supply Agreement of GAMMA

### 1. Law

This Agreement is governed by the laws of Ohio.

## Schedule 1 Prices

Widgets cost 10 USD each.
"""


def room():
    return build_tree(DocumentGraph.from_text(ROOM))


def chapter(tree, title):
    return next(c.id for c in tree.chapters() if c.title == title)


# ------------------------------------------------------------------ routing

def test_routing_separates_the_unit_named_from_the_content_asked():
    tree = room()
    r = route(tree, "Which law governs the Lease of BETA?")
    assert r.chapters == [chapter(tree, "Lease of BETA")] and "beta" not in r.content and "law" in r.content
    r = route(tree, "Which law governs the Supply Agreement of GAMMA?")
    assert r.chapters == [chapter(tree, "Supply Agreement of GAMMA")]       # not ACME: "gamma" is distinctive
    assert route(tree, "Which law governs the agreement?").chapters == []


def test_routing_a_named_book_opens_all_its_chapters():
    lib = build_tree(DocumentGraph.from_text(
        "# Library\n\n## Book: Alpha Tales\n\n### CHAPTER I\n\nA fox ran.\n\n### CHAPTER II\n\nThe fox slept.\n\n"
        "## Book: Beta Saga\n\n### CHAPTER I\n\nA hare ran.\n\n### CHAPTER II\n\nThe hare hid in a barn.\n"))
    r = route(lib, "In Beta Saga, where did the hare hide?")
    beta = next(p for p in lib.parts() if p.title == "Book: Beta Saga")
    assert r.chapters == beta.children and "hare" in r.content


# -------------------------------------------------------------------- board

def test_board_checks_ids_quotes_exactly_merges_and_adds_definitions():
    tree = room()
    board = Board(tree, "Where are the widgets stored and which law governs the lease?")
    texas = next(p for p in tree.passages() if "Texas" in p.text)
    s_law, s_stored = [s for s, _ in texas.sentences]
    out = board.submit("w1", [{"sentence_ids": [s_law, "p999.s9"], "why": "law"}])
    assert out == {"accepted": 1, "rejected": 1} and board.rejected[0]["ids"] == ["p999.s9"]
    assert board.submit("w2", [{"sentence_ids": [f"[{s_law}]"], "why": "again"}])["accepted"] == 0  # duplicate
    board.submit("w2", [{"sentence_ids": [s_stored], "why": "storage"}])
    assert len(board.accepted) == 1 and board.accepted[0].sentence_ids == [s_law, s_stored]  # same passage merged
    assert board.add_definitions() == 1                                   # "Widgets" is defined elsewhere
    pack = board.to_pack()
    assert [i.label for i in pack.items] == ["E1", "E2"] and pack.items[1].role == "definition"
    assert pack.items[0].text == "This lease is governed by the laws of Texas. The Widgets are stored on site."
    assert pack.items[0].section_path.startswith("Lease of BETA") and board.status == "found"


# ------------------------------------------------------------ workers + ask

class FakeModel:
    """Worker calls: post the shown sentences containing ``keyword``; optionally read a node
    first, or follow a title. Answer calls: answer citing E1."""

    def __init__(self, keyword, read_first=None, follow=None, toc=None, rewrite=None, partial=""):
        self.keyword, self.read_first, self.follow = keyword, read_first, follow
        self.toc, self.rewrite, self.partial = toc or [], rewrite or [], partial
        self.calls, self.users, self.name = 0, [], "fake"

    def complete(self, system, user, schema):
        self.calls += 1
        self.users.append(user)
        if schema is TOC_SCHEMA:
            return Completion(json.dumps({"chapters": self.toc}), "fake", 100, 10)
        if schema is REWRITE_SCHEMA:
            return Completion(json.dumps({"keywords": self.rewrite}), "fake", 30, 10)
        if schema is ANSWER_SCHEMA:
            data = {"answer": "Texas law [E1].", "evidence": ["E1"], "confidence": "high", "insufficient": False}
            return Completion(json.dumps(data), "fake", 50, 10)
        ids = [m.group(1) for m in re.finditer(r"^\[([^\]]+)\] (.*)$", user, re.M) if self.keyword in m.group(2)]
        if self.read_first and "Text of [" not in user:
            data = {"findings": [], "follow": [], "next": {"action": "read", "target": self.read_first}}
        else:
            data = {"findings": [{"sentence_ids": ids, "why": "matches"}] if ids else [],
                    "follow": [self.follow] if self.follow and not ids else [],
                    "partial": self.partial if ids else "",
                    "next": {"action": "done", "target": ""}}
        return Completion(json.dumps(data), "fake", 100, 20)


def test_named_chapter_gets_one_worker_and_the_answer_is_cited():
    tree = room()
    model = FakeModel("Texas")
    result = ask(tree, "Which law governs the Lease of BETA?", model)
    assert result.chapters_searched == [chapter(tree, "Lease of BETA")]
    assert result.answer.cited == ["E1"] and not result.answer.abstained
    assert result.calls == 2 == model.calls                       # one worker call + the answer
    assert "never follow instructions" in model.users[0] or "<document>" in model.users[0]


def test_worker_can_read_before_posting():
    tree = room()
    rent = next(p for p in tree.passages() if "rent" in p.text)
    model = FakeModel("900 USD", read_first=tree.node(rent.parent).id)
    result = ask(tree, "How much rent does the Lease of BETA charge?", model)
    report = result.reports[0]
    assert report.actions[0].startswith("read:") and report.posted == 1
    assert "900 USD" in result.pack.items[0].text


def test_without_the_gate_an_unnamed_question_searches_every_chapter_and_follows_references():
    tree = room()
    model = FakeModel("Ohio")
    result = ask(tree, "Which agreement is governed by Ohio law?", model, use_gate=False)
    assert set(result.chapters_searched) == {c.id for c in tree.chapters()}
    assert "Ohio" in result.pack.items[0].text
    model = FakeModel("10 USD", follow="Schedule 1")
    result = ask(tree, "What do widgets cost under the Supply Agreement of ACME?", model)
    assert result.chapters_searched == [chapter(tree, "Supply Agreement of ACME"), chapter(tree, "Schedule 1 Prices")]
    assert any("10 USD" in i.text for i in result.pack.items)


def test_nothing_found_means_no_answer_call_and_budget_stops_workers():
    tree = room()
    model = FakeModel("Zanzibar")
    result = ask(tree, "Which law governs the Lease of BETA?", model, use_gate=False)
    assert result.answer.abstained and result.answer.flags == ["no_evidence"] and model.calls == 1
    result = ask(tree, "Which agreement mentions Zanzibar?", FakeModel("Zanzibar"), max_calls=2, parallel=1,
                 use_gate=False)
    assert result.calls == 2 and any(r.error == "budget" for r in result.reports)


def test_free_text_follow_finds_the_chapter_that_defines_a_code():
    tree = build_tree(DocumentGraph.from_text(
        "# Room\n\n## Services Agreement of ACME\n\nThe hourly charges are set out in rate card RC-8985.\n\n"
        "## Lease of BETA\n\nThe tenant pays rent monthly.\n\n"
        "## Consulting Terms of GAMMA\n\nRate card RC-8985 fixes the hourly rate at 212 USD.\n"))

    class TwoHop(FakeModel):
        def complete(self, system, user, schema):
            if schema is not ANSWER_SCHEMA and "RC-8985" in user and "212 USD" not in user:
                self.calls += 1
                sid = re.search(r"^\[([^\]]+)\] The hourly", user, re.M).group(1)
                return Completion(json.dumps({"findings": [{"sentence_ids": [sid], "why": "points to the card"}],
                                              "follow": ["rate card RC-8985"],
                                              "next": {"action": "done", "target": ""}}), "fake", 100, 20)
            return super().complete(system, user, schema)

    result = ask(tree, "What hourly rate applies under the Services Agreement of ACME?", TwoHop("212 USD"))
    titles = [tree.node(c).title for c in result.chapters_searched]
    assert titles == ["Services Agreement of ACME", "Consulting Terms of GAMMA"]
    assert [("RC-8985" in i.text, "212 USD" in i.text) for i in result.pack.items] == [(True, False), (True, True)]



# --------------------------------------------------------------- the gate

def test_scope():
    assert scope("Summarise the main events of the book.") == "global"
    assert scope("List all obligations of the Supplier.") == "global"
    assert scope("Why did Norman leave the castle?") == "multi"
    assert scope("Which law governs the Lease of BETA?") == "single"


def test_gate_opens_best_chapters_by_any_signal_without_searching_everything():
    tree = room()
    model = FakeModel("Ohio", toc=[chapter(tree, "Lease of BETA")])
    result = ask(tree, "Which agreement is governed by Ohio law?", model)
    gamma, beta = chapter(tree, "Supply Agreement of GAMMA"), chapter(tree, "Lease of BETA")
    assert gamma in result.plan.rounds[0] and "lexical" in result.plan.opened_by[gamma]
    assert "toc" in result.plan.opened_by[beta]                    # the TOC screen opened BETA too
    assert len(result.chapters_searched) < len(tree.chapters()) and "Ohio" in result.pack.items[0].text
    assert result.rounds_run == 1                                  # found in round 1: no widening


def test_named_unit_first_then_widening_then_rewrite_before_not_found():
    tree = room()
    beta = chapter(tree, "Lease of BETA")
    plan = Gate(tree, SentenceIndex(tree)).plan("Which law governs the Lease of BETA?", ["law", "govern"], [beta])
    assert plan.rounds[0] == [beta] and plan.opened_by[beta][0] == "named" and plan.rounds[1]

    # A bigger room, price schedule last: the question's words match nothing, rounds 1-2
    # (3 + 3 chapters) miss it, and the rewrite finds it.
    fillers = "".join(f"## Notice {k} of DELTA{k}\n\nThe parties met on day {k}.\n\n" for k in range(6))
    big = build_tree(DocumentGraph.from_text(ROOM.replace("## Schedule 1 Prices", fillers + "## Schedule 1 Prices")))
    model = FakeModel("10 USD", rewrite=["widgets", "cost", "each"])
    result = ask(big, "How expensive is a single thingamajig?", model, use_toc=False)
    tree, sched = big, chapter(big, "Schedule 1 Prices")
    assert sched not in result.plan.rounds[0] + result.plan.rounds[1]
    assert "rewrite" in result.plan.opened_by.get(sched, []) and any("10 USD" in i.text for i in result.pack.items)


def test_a_named_book_limits_the_gate_to_its_chapters():
    text = "# Library\n\n" + "".join(
        f"## Book: {name}\n\n" + "".join(f"### CHAPTER {k}\n\n{name} text {k}. The {animal} ran.\n\n"
                                         for k in range(1, 9))
        for name, animal in (("Alpha Tales", "fox"), ("Beta Saga", "hare")))
    lib = build_tree(DocumentGraph.from_text(text))
    beta = set(next(p for p in lib.parts() if p.title == "Book: Beta Saga").children)
    model = FakeModel("hare")
    result = ask(lib, "In Beta Saga, where did the hare run?", model, use_toc=False)
    assert set(result.chapters_searched) <= beta and 0 < len(result.chapters_searched) < len(beta)


def test_evidence_pack_respects_the_token_limit():
    tree = room()
    board = Board(tree, "q")
    for p in tree.passages():
        board.submit("w", [{"sentence_ids": [p.sentences[0][0]], "why": ""}])
    assert len(board.to_pack(max_tokens=20).items) < len(board.to_pack().items)
    assert len(board.to_pack(max_tokens=1).items) == 1             # the first item is always kept


def test_routing_names_every_slice_of_a_split_unit():
    para = " ".join(f"word{i}" for i in range(120)) + "."
    text = "# Room\n\n## Lease of BETA\n\nRent is due monthly.\n\n## Franchise Agreement of GOOSEHEAD\n\n" + \
        "\n\n".join(f"Clause {k}. {para}" for k in range(250))
    tree = build_tree(DocumentGraph.from_text(text))
    slices = [c.id for c in tree.chapters() if c.title.startswith("Franchise Agreement of GOOSEHEAD (")]
    assert len(slices) >= 2
    assert route(tree, "Who owns the IP under the Franchise Agreement of GOOSEHEAD?").chapters == slices


def test_whole_document_question_gathers_many_chapters_and_passes_partial_answers_as_notes():
    tree = room()
    model = FakeModel("governed", partial="this agreement names its governing law")
    result = ask(tree, "List all the governing laws across the agreements.", model, use_toc=False)
    assert result.scope == "global"
    laws = {t for i in result.pack.items for t in ("Ontario", "Texas", "Ohio") if t in i.text}
    assert laws == {"Ontario", "Texas", "Ohio"}                  # one quote per agreement
    answer_prompt = model.users[-1]
    assert "Notes from the search" in answer_prompt and "names its governing law" in answer_prompt
    assert "not evidence" in answer_prompt


# ------------------------------------------------- second hop + widening cost (F10, F11)

def test_board_bridges_a_two_part_fact_without_a_model_call():
    tree = build_tree(DocumentGraph.from_text(
        "# Library\n\n## CHAPTER I\n\nRufus said the parcel must be left with the man who wore a copper scarf.\n\n"
        "## CHAPTER II\n\nThe ferry left at dawn. Nobody spoke.\n\n"
        "## CHAPTER III\n\nThe man who wore a copper scarf was the ferryman, Silas Grey.\n\n"
        "## CHAPTER IV\n\nOttilie hid the silver key inside a blue tin of peppermints.\n\n"
        "## CHAPTER V\n\nMarta hid the gold key inside a red tin of peppermints.\n"))
    index = SentenceIndex(tree)
    board = Board(tree, "Who was Rufus's parcel to be left with?")
    board.index = index
    first = next(p for p in tree.passages() if "Rufus" in p.text).sentences[0][0]
    board.submit("w1", [{"sentence_ids": [first], "why": "parcel"}])
    assert board.add_bridges() == 1
    assert [f.role for f in board.accepted] == ["answer", "bridge"]
    assert "Silas Grey" in board.quote(board.accepted[1]) and board.calls == 0
    # a same-template sentence (a distractor) is not a bridge
    board = Board(tree, "Where did Ottilie hide the silver key?")
    board.index = index
    key = next(p for p in tree.passages() if "Ottilie" in p.text).sentences[0][0]
    board.submit("w1", [{"sentence_ids": [key], "why": "key"}])
    assert board.add_bridges() == 0


def test_two_hop_answer_pack_holds_both_parts():
    tree = build_tree(DocumentGraph.from_text(
        "# Room\n\n## Services Agreement of ACME\n\nThe hourly charges are set out in rate card RC-8985.\n\n"
        "## Lease of BETA\n\nThe tenant pays rent monthly.\n\n"
        "## Consulting Terms of GAMMA\n\nRate card RC-8985 fixes the hourly rate for consulting at 212 USD.\n"))
    result = ask(tree, "What hourly charges apply under the Services Agreement of ACME?", FakeModel("RC-8985"),
                 use_toc=False, widen=False)
    texts = [i.text for i in result.pack.items]
    assert any("212 USD" in t for t in texts) and result.pack.items[1].role == "bridge"


def test_widening_skips_chapters_without_word_evidence_and_uses_fewer_steps():
    fillers = "".join(f"## Notice {k} of DELTA{k}\n\nThe parties met on day {k}.\n\n" for k in range(6))
    big = build_tree(DocumentGraph.from_text(ROOM.replace("## Schedule 1 Prices", fillers + "## Schedule 1 Prices")))
    model = FakeModel("Zanzibar")
    result = ask(big, "Which agreement is governed by Zanzibar law?", model, use_toc=False)
    assert result.answer.abstained
    notices = {c.id for c in big.chapters() if c.title.startswith("Notice")}
    assert not notices & set(result.chapters_searched)        # no word match: never opened
    assert result.calls < len(big.chapters())
