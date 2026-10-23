"""Build step 2: benchmark construction helpers, scoring, and the plain vector-search baseline.

No network and no downloads: synthetic units stand in for CUAD contracts and Gutenberg books.
"""

import random

from gnn_pruner.longbench import (Unit, _assign_tiers, _book_blocks, _contract_blocks, _contract_title,
                                  plant_needles, render)
from gnn_pruner.longbench_eval import _spread, answer_overlap, contains_answer, coverage


def test_coverage_and_answer_checks():
    gold = "This Agreement shall be governed by the laws of the State of Delaware."
    assert coverage(gold, "... 12.3 This Agreement shall be governed by the laws of the State of Delaware. ...") == 1.0
    assert coverage(gold, "governed by the laws of England") < 0.5
    assert coverage("RC-4417", "rate card RC-4417 fixes") == 1.0     # short gold: substring rule
    assert answer_overlap(["Ben Keene, Delmar's valet"], "Arabella married Ben Keene, the valet.") >= 0.75
    assert contains_answer("It is fixed at 212 USD per hour [E2].", "212 USD")
    assert contains_answer("Harriet Okonkwo of Lisbon is named.", "Harriet Okonkwo of Lisbon")
    assert contains_answer("It was signed only with a small drawing of a fox.", "with a drawing of a fox")
    assert not contains_answer("It was signed with a drawing of a heron.", "with a drawing of a fox")
    assert not contains_answer("left with the man in the copper scarf", "Henrike Holloway, the ferryman")
    assert not contains_answer("The document does not say.", "212 USD")


def test_contract_and_book_text_become_nested_markdown():
    blocks = _contract_blocks("EXHIBIT 10.6\n\n1. DEFINITIONS\n\n\"Goods\" means the products.\n\n"
                              "ARTICLE V - TERM\n\nThe term is ten years.")
    assert "### 1. DEFINITIONS" in blocks and "### ARTICLE V - TERM" in blocks
    assert not any(b.startswith("#") and not b.startswith("### ") for b in _contract_blocks("Text.\n\n#\n\n# #\n\nMore."))
    assert _contract_title("LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT", "DISTRIBUTOR AGREEMENT") == \
        "Distributor Agreement of LIMEENERGYCO (1999-09-09)"
    book = _book_blocks("BOOK I\n\nCHAPTER I. The Period\n\nIt was the best of times.\n\nCHAPTER II\n\nMore.")
    assert book[0] == "BOOK I" and "### CHAPTER I. The Period" in book and "### CHAPTER II" in book


def _units(n, chars):
    return [Unit(f"u{i}", f"Unit {i}", ["### Part A"] + [f"Paragraph {i}.{k} text." for k in range(20)],
                 chars=chars) for i in range(n)]


def test_tiers_nest_and_needles_are_planted_with_questions():
    units = _units(30, 60_000)              # 20 pages each: S = 5 units, M = 50, ...
    _assign_tiers(units)
    tiers = [u.tier for u in units]
    assert tiers[:5] == ["S"] * 5 and set(tiers[5:]) == {"M"}
    cases = plant_needles(units, "contracts", random.Random(1), per_tier=3, two_hop=1, absent=2)
    kinds = [c["tags"]["kind"] for c in cases]
    assert kinds.count("needle") == 6 and kinds.count("needle_2hop") == 2 and kinds.count("needle_absent") == 4
    s_text, m_text = render(units, "contracts", "S"), render(units, "contracts", "M")
    assert s_text.startswith("# Contract data room (5 contracts)") and len(m_text) > len(s_text)
    for c in cases:
        text = m_text if c["tags"]["unit_tier"] == "M" else s_text
        assert all(g in text for g in c["gold_evidence"])          # every planted sentence is present
        if c["tags"]["kind"] == "needle_2hop":
            assert c["tags"]["unit"] != c["tags"]["second_unit"]     # contracts: rate card in another one
    absent = [c for c in cases if c["tags"]["kind"] == "needle_absent"]
    assert all(not c["answerable"] and not c["gold_evidence"] for c in absent)


def test_spread_balances_kinds():
    class C:
        def __init__(self, kind):
            self.tags = {"kind": kind}

    picked = _spread([C("a")] * 10 + [C("b")] * 2 + [C("c")] * 5, 6)
    assert sorted(c.tags["kind"] for c in picked) == ["a", "a", "b", "b", "c", "c"]


def test_rag_baseline_retrieves_title_prefixed_chunks(tmp_path):
    from gnn_pruner.embeddings import load_embedder
    from gnn_pruner.longbench_eval import RagBaseline

    doc = tmp_path / "room.md"
    doc.write_text("# Data room\n\n## Supply Agreement of ACME\n\n"
                   + "\n\n".join(f"Clause {k}: the supplier ships widgets weekly." for k in range(5))
                   + "\n\nThis Agreement is governed by the laws of Ontario.\n\n"
                   "## Lease of BETA\n\nThe tenant pays rent monthly.\n\nThis lease is governed by the laws of Texas.\n")
    rag = RagBaseline(str(doc), load_embedder("tfidf"), chunk_tokens=40, k=1)
    assert any(path.startswith("Supply Agreement of ACME") for _, path, _ in rag.chunks)
    pack, _ = rag.ask("Which law governs the Lease of BETA?")
    assert "Texas" in pack.items[0].text and pack.items[0].label == "E1"
