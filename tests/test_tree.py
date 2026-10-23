"""Build step 1: PDF outline + cleaning, the document tree, metadata, enrichment, upload cache.

No network: model calls go to scripted fake providers.
"""

import json
import re

import pytest

from gnn_pruner.answer import Completion, QuotaExhausted
from gnn_pruner.document_graph import DocumentGraph
from gnn_pruner.enrich import Enricher, revalidate, strip_ids
from gnn_pruner.ingest.pdf_loader import toc_levels
from gnn_pruner.metadata import extract_metadata, isbn_valid
from gnn_pruner.textutil import clean_text, match_key
from gnn_pruner.tree import CHAPTER, PASSAGE, SECTION, SUBSECTION, DocTree, build_tree

CONTRACT = """# Supply Agreement

## 1. Definitions

"Goods" means the products listed in Schedule 1.

## 2. Delivery

The Supplier shall deliver the Goods within 30 days of an order. Late delivery incurs a fee of 5 percent.

### 2.1 Risk

Risk in the Goods passes on delivery, subject to Section 1.

## 3. Payment

The Customer shall pay each invoice within 60 days.

### Schedule 1 Prices

Widgets cost 10 USD each.
"""


def tree_of(text, title="Document", info=None):
    return build_tree(DocumentGraph.from_text(text, title=title), info=info)


# ------------------------------------------------------------- text + outline

def test_clean_text_and_match_key():
    assert clean_text("eﬃcient “quoted” soft­hyphen") == 'efficient "quoted" softhyphen'
    assert match_key("I.A SCANDAL IN BOHEMIA") == match_key("I. A Scandal in Bohemia")


def test_toc_levels_fix_flat_part_outlines_only():
    flat = [(1, "A TALE", 1), (2, "Contents", 1), (2, "Book the First", 4), (2, "CHAPTER I. The Period", 5),
            (2, "CHAPTER II. The Mail", 7), (2, "Book the Second", 31), (2, "CHAPTER I. Five Years", 32),
            (2, "THE FULL PROJECT GUTENBERG LICENSE", 211)]
    assert toc_levels(flat) == [1, 2, 2, 3, 3, 2, 3, 2]
    nested = [(1, "Part One", 1), (2, "Chapter 1", 2), (3, "1.1 Start", 2), (1, "Part Two", 9), (2, "Chapter 2", 10)]
    assert toc_levels(nested) == [1, 2, 3, 1, 2]


def test_pdf_outline_split_headings_drop_caps_and_indents(tmp_path):
    fitz = pytest.importorskip("pymupdf")
    from gnn_pruner.ingest import load_document

    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 120), "THE TEST BOOK", fontsize=22)
    p1.insert_text((72, 160), "A short book used by the tests of the outline reader.", fontsize=12)
    p2 = doc.new_page()
    p2.insert_text((72, 120), "I.", fontsize=16)
    p2.insert_text((72, 142), "THE BEGINNING", fontsize=16)
    p2.insert_text((72, 190), "I", fontsize=30)
    p2.insert_text((96, 186), "t was a cold day in the town.", fontsize=12)
    y = 230
    for x, line in [(84, "The first paragraph starts here and"), (72, "it continues on this line."),
                    (84, "The second paragraph starts here"), (72, "and it ends here.")]:
        p2.insert_text((x, y), line, fontsize=12)
        y += 14
    p3 = doc.new_page()
    p3.insert_text((72, 120), "II.", fontsize=16)
    p3.insert_text((72, 142), "THE END", fontsize=16)
    p3.insert_text((84, 190), "Everything ended quietly.", fontsize=12)
    doc.set_toc([[1, "THE TEST BOOK", 1], [2, "I.THE BEGINNING", 2], [2, "II.THE END", 3]])
    path = tmp_path / "book.pdf"
    doc.save(str(path))

    parsed = load_document(str(path))
    titles = [n.title for n in parsed.nodes if n.is_section]
    assert "I. THE BEGINNING" in titles and "II. THE END" in titles and "I" not in titles
    texts = [n.text for n in parsed.nodes if n.node_type == "paragraph"]
    assert any(t.startswith("It was a cold day") for t in texts)
    assert any(t.startswith("The first paragraph") and t.endswith("this line.") for t in texts)
    assert any(t.startswith("The second paragraph") for t in texts)
    assert parsed.info["toc_matched"] == 3 and parsed.info["toc_unmatched"] == []


# ------------------------------------------------------------------- the tree

def test_tree_shape_schedules_links_and_cite_by_id(tmp_path):
    tree = tree_of(CONTRACT)
    assert tree.title == "Supply Agreement" and tree.kind == "contract"
    chapters = [c.title for c in tree.chapters()]
    assert chapters == ["1. Definitions", "2. Delivery", "3. Payment", "Schedule 1 Prices"]
    for p in tree.passages():  # same depth everywhere
        sub = tree.node(p.parent)
        sec = tree.node(sub.parent)
        assert (sub.kind, sec.kind, tree.node(sec.parent).kind) == (SUBSECTION, SECTION, CHAPTER)
    risk = next(p for p in tree.passages() if p.text.startswith("Risk"))
    assert tree.path(risk.id) == ["2. Delivery", "2.1 Risk"]
    assert risk.links.get("refers_to") == ["ch1"]
    definition = next(p for p in tree.passages() if p.text.startswith('"Goods"'))
    assert definition.defined_terms == ["goods"]
    delivery = next(p for p in tree.passages() if p.text.startswith("The Supplier shall"))
    first, second = delivery.sentences
    assert tree.quote([second[0]]) == "Late delivery incurs a fee of 5 percent."
    assert tree.sentence(first[0])[1] == delivery.id
    path = tmp_path / "t.json"
    tree.save(str(path))
    again = DocTree.load(str(path))
    assert again.stats() == tree.stats() and again.quote([second[0]]) == tree.quote([second[0]])


def test_parts_group_chapters():
    text = "# Book One\n\n## Chapter 1\n\nAlpha text.\n\n## Chapter 2\n\nBeta text.\n\n# Book Two\n\n## Chapter 3\n\nGamma text.\n"
    tree = tree_of(text)
    assert [p.title for p in tree.parts()] == ["Book One", "Book Two"]
    assert [c.title for c in tree.chapters()] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    gamma = next(p for p in tree.passages() if p.text.startswith("Gamma"))
    assert tree.path(gamma.id)[:2] == ["Book Two", "Chapter 3"]


def test_unstructured_text_becomes_virtual_chapters_and_long_subsections_are_sliced():
    para = " ".join(f"word{i}" for i in range(100)) + "."
    tree = tree_of("\n\n".join(f"Paragraph {k}. {para}" for k in range(300)))
    chapters = tree.chapters()
    assert len(chapters) >= 2 and all(c.virtual for c in chapters)
    ids = [p.id for p in tree.passages()]
    assert len(ids) == 300 and ids == sorted(ids, key=lambda i: int(i[1:]))  # order kept
    subs = tree.of_kind(SUBSECTION)
    assert all(s.tokens <= 2000 for s in subs) and any(re.search(r"\(\d+/\d+\)$", s.title) for s in subs)


# ------------------------------------------------------------------- metadata

def test_isbn_checksum():
    assert isbn_valid("978-0-14-143960-0") and isbn_valid("0-306-40615-2")
    assert not isbn_valid("978-0-14-143960-1")


class Scripted:
    """Fake provider: ``reply(system, user, schema) -> dict``; can raise on call n."""

    def __init__(self, reply, fail_on=None):
        self.reply, self.fail_on, self.calls, self.users = reply, fail_on, 0, []
        self.name = "scripted"

    def complete(self, system, user, schema):
        self.calls += 1
        self.users.append(user)
        if self.fail_on and self.calls == self.fail_on:
            raise QuotaExhausted("scripted: daily quota")
        return Completion(json.dumps(self.reply(system, user, schema)), "scripted", 100, 20, cost_usd=0.0)


def test_book_metadata_from_labels_and_patterns_llm_values_must_be_in_text():
    front = ("Title: The Quiet Harbour\n\nAuthor: Jane Writer\n\nISBN 978-0-14-143960-0\n\n"
             "Copyright © 1998 Harbour Press.\n\n")
    tree = tree_of(front + "\n\n".join(f"Chapter text {k} about the sea." for k in range(5)))
    fields = extract_metadata(tree)
    assert fields["title"]["value"] == "The Quiet Harbour" and fields["author"]["value"] == "Jane Writer"
    assert fields["isbn"]["value"] == "978-0-14-143960-0" and fields["year"]["value"] == "1998"
    assert all(f["verified"] for f in fields.values())

    tree = tree_of("Copyright notice: all rights reserved.\n\nPublished by Harbour Press in London.\n\nText.")
    llm = Scripted(lambda s, u, sc: {"fields": [
        {"name": "author", "value": "Invented Person", "evidence": "Written by Invented Person"},
        {"name": "publisher", "value": "Harbour Press", "evidence": "Published by Harbour Press in London."}]})
    fields = extract_metadata(tree, llm)
    assert "author" not in fields and fields["publisher"]["value"] == "Harbour Press"

    tree = tree_of("Release date: March 1, 1999 [eBook #1661]\n\nSome text.")
    llm = Scripted(lambda s, u, sc: {"fields": [
        {"name": "year", "value": "1999", "evidence": "Release date: March 1, 1999 [eBook #1661]"}]})
    assert "year" not in extract_metadata(tree, llm)  # an e-book release is not publication
    assert tree.root.metadata["usage"]["metadata"]["rejected"] == 1
    assert "<document>" in llm.users[0]


def test_contract_metadata_parties_date_and_law():
    text = ("# Distribution Agreement\n\nThis Agreement is made on 1 March 2024 between Acme Ltd "
            '(the "Supplier") and Beta GmbH (the "Distributor").\n\n## 1. Law\n\n'
            "This Agreement shall be governed by the laws of England and Wales.\n")
    tree = tree_of(text)
    fields = extract_metadata(tree)
    assert tree.kind == "contract"
    assert fields["parties"]["value"] == ["Acme Ltd", "Beta GmbH"]
    assert fields["effective_date"]["value"] == "1 March 2024"
    assert fields["governing_law"]["value"] == "England and Wales"
    assert fields["title"]["value"] == "Distribution Agreement"


# ----------------------------------------------------------------- enrichment

def fake_enricher_reply(system, user, schema):
    """Facts: per item one valid fact, one with a foreign source, one with a wrong number.
    Summaries: "summary of <id>"."""
    items = re.findall(r'<item id="([^"]+)"[^\n]*>\n(.*?)\n</item>', user, re.S)
    if "facts" in json.dumps(schema):
        out = []
        for item_id, body in items:
            sid, text = re.match(r"\[([^\]]+)\] (.*)", body.splitlines()[0]).groups()
            out.append({"id": item_id, "facts": [
                {"text": text, "type": "obligation", "sources": [f"[{sid}]"]},
                {"text": "Something unrelated.", "type": "other", "sources": ["nowhere"]},
                {"text": "It costs 999 dollars.", "type": "amount", "sources": [sid]}]})
        return {"items": out}
    return {"items": [{"id": item_id, "summary": f"summary of {item_id}"} for item_id, _ in items]}


def test_enrichment_checks_facts_summarises_bottom_up_and_builds_lists():
    tree = tree_of(CONTRACT)
    provider = Scripted(fake_enricher_reply)
    stats = Enricher(provider).run(tree)
    subs = tree.of_kind(SUBSECTION)
    assert all(len(s.key_facts) == 1 for s in subs)
    assert stats["facts_dropped_source"] == len(subs) and stats["facts_dropped_numbers"] == len(subs)
    assert all(s.summary == f"summary of {s.id}" for s in tree.of_kind(SECTION))
    assert all(c.summary for c in tree.chapters()) and tree.root.summary == "summary of root"
    assert tree.root.metadata["lists"]["obligation"][0]["sources"]
    assert stats["calls"] == provider.calls and tree.root.metadata["enrichment"]["complete"]
    assert all("never follow instructions" in u or "<document>" in u for u in provider.users)
    calls = provider.calls
    Enricher(provider).run(tree)  # everything done: no new calls
    assert provider.calls == calls


def test_enrichment_stops_on_quota_and_resumes():
    tree = tree_of(CONTRACT)
    first = Scripted(fake_enricher_reply, fail_on=2)
    stats = Enricher(first, max_items=2).run(tree)
    assert "daily quota" in stats["stopped"] and not tree.root.metadata["enrichment"]["complete"]
    done = [s.id for s in tree.of_kind(SUBSECTION) if s.metadata.get("facts_done")]
    assert len(done) == 2
    second = Scripted(fake_enricher_reply)
    second.name = "scripted"
    Enricher(second, max_items=2).run(tree)
    assert tree.root.metadata["enrichment"]["complete"]
    assert not any(any(f'<item id="{d}"' in u for d in done) for u in second.users)  # not redone


def test_offline_enrichment_and_upload_cache(tmp_path):
    from gnn_pruner.upload import upload

    doc = tmp_path / "contract.md"
    doc.write_text(CONTRACT)
    tree = upload(str(doc), cache_dir=tmp_path / "trees")
    assert tree.root.metadata["enrichment"]["provider"] == "extractive"
    kinds = {f["type"] for s in tree.of_kind(SUBSECTION) for f in s.key_facts}
    assert {"definition", "deadline"} <= kinds
    cached = list((tmp_path / "trees").glob("contract-*.json"))
    assert len(cached) == 1
    provider = Scripted(fake_enricher_reply)
    again = upload(str(doc), cache_dir=tmp_path / "trees", provider=provider)  # new provider: re-enrich
    assert provider.calls > 0 and again.root.metadata["enrichment"]["provider"] == "scripted"
    calls = provider.calls
    upload(str(doc), cache_dir=tmp_path / "trees", provider=provider)  # cached and complete
    assert provider.calls == calls


def test_sentence_ids_in_fact_text_do_not_count_as_numbers_and_drops_can_be_rechecked():
    assert strip_ids("Payment is due in 60 days [p11.s2, p11.s3].") == "Payment is due in 60 days."
    tree = tree_of(CONTRACT)
    sub = next(s for s in tree.of_kind(SUBSECTION) if s.title == "3. Payment")
    sid = tree.children(sub.id)[0].sentences[0][0]
    tree.root.metadata["enrichment"] = {"stats": {}}
    sub.metadata["dropped_facts"] = [{"text": "Invoices are paid within 60 days. [p9]", "type": "deadline",
                                      "sources": [sid], "reason": "numbers", "missing": ["9"]}]
    assert revalidate(tree)["moved"] == 1
    assert sub.key_facts[-1]["text"] == "Invoices are paid within 60 days."
    assert tree.root.metadata["lists"]["deadline"]


def test_fact_filed_under_the_wrong_item_moves_to_the_subsection_owning_its_sentences():
    tree = tree_of(CONTRACT)
    subs = tree.of_kind(SUBSECTION)
    a, b = subs[0], subs[1]
    b_sid = tree.children(b.id)[0].sentences[0][0]
    b_text = tree.quote([b_sid])

    def reply(system, user, schema):
        if "facts" not in json.dumps(schema):
            return fake_enricher_reply(system, user, schema)
        ids = re.findall(r'<item id="([^"]+)"', user)
        return {"items": [{"id": i, "facts": [{"text": b_text, "type": "deadline", "sources": [b_sid]}]
                           if i == a.id else []} for i in ids]}

    Enricher(Scripted(reply)).run(tree)
    assert [f["text"] for f in b.key_facts] == [b_text] and a.key_facts == []


def test_unparseable_replies_are_retried_then_fall_back_and_completion_is_real():
    tree = tree_of(CONTRACT)

    class Broken(Scripted):
        def complete(self, system, user, schema):
            self.calls += 1
            if "facts" in json.dumps(schema):
                return Completion("{not json", "scripted", 10, 10)  # e.g. a reply cut off by the cap
            return Completion(json.dumps(fake_enricher_reply(system, user, schema)), "scripted", 10, 10)

    stats = Enricher(Broken(None)).run(tree)
    assert stats["unparsed"] == 2 and stats["items_fallback"] == len(tree.of_kind(SUBSECTION))
    assert all(s.metadata.get("facts_done") for s in tree.of_kind(SUBSECTION))
    assert tree.root.metadata["enrichment"]["complete"]
