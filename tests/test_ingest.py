from pathlib import Path

import pytest

from gnn_pruner import DocumentGraph, DocumentQA
from gnn_pruner.evaluation import evaluate_case, load_cases
from gnn_pruner.ingest import load_document
from gnn_pruner.ingest.html_loader import extract_html
from gnn_pruner.parser import normalize_ws

from renderers import RENDERERS

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
DATASET = ROOT / "eval" / "contracts_v1.jsonl"
CONTRACTS = ["sample_contract.md", "saas_agreement.md"]


def _available(suffix):
    module = {".pdf": "pymupdf", ".docx": "docx", ".html": "html"}[suffix]
    try:
        __import__(module)
        return True
    except ImportError:
        return False


FORMATS = [pytest.param(s, marks=pytest.mark.skipif(not _available(s), reason=f"{s} extra missing"))
           for s in RENDERERS]


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """{(contract_name, suffix): path} for every renderable format."""
    out = tmp_path_factory.mktemp("rendered")
    paths = {}
    for name in CONTRACTS:
        text = (EXAMPLES / name).read_text()
        for suffix, render in RENDERERS.items():
            if _available(suffix):
                paths[(name, suffix)] = render(text, out / Path(name).with_suffix(suffix).name)
    return paths


def _sections(graph):
    return [(n.node_type, n.title, n.metadata.get("number")) for n in graph.sections()]


def _structure(graph):
    """(section path, text) for every paragraph/table: the format-independent content."""
    return sorted((graph.path(n.id), normalize_ws(n.text).replace(" ", ""))
                  for n in graph.content_nodes(("paragraph", "table")))


@pytest.mark.parametrize("suffix", FORMATS)
@pytest.mark.parametrize("name", CONTRACTS)
def test_format_matches_markdown_structure(rendered, name, suffix):
    reference = DocumentGraph.from_file(str(EXAMPLES / name))
    graph = DocumentGraph.from_file(str(rendered[(name, suffix)]))
    assert graph.title == reference.title
    assert _sections(graph) == _sections(reference)
    assert _structure(graph) == _structure(reference)
    assert graph.stats() == reference.stats()
    assert graph.validate() == []


@pytest.mark.parametrize("suffix", FORMATS)
def test_offsets_point_into_canonical_text(rendered, suffix):
    doc = load_document(str(rendered[("saas_agreement.md", suffix)]))
    for node in doc.nodes:
        if node.node_type in ("paragraph", "sentence"):
            assert normalize_ws(doc.text[node.start:node.end]) == node.text


@pytest.mark.parametrize("suffix", FORMATS)
def test_eval_parity_with_markdown(rendered, suffix):
    """Phase 1 acceptance: the Phase 0 gold set scores the same on every format."""
    cases = load_cases(str(DATASET))
    systems = {}
    for case in cases:
        name = Path(case.document).name
        md_qa = systems.setdefault(("md", name), DocumentQA.from_file(case.document))
        fmt_qa = systems.setdefault((suffix, name), DocumentQA.from_file(str(rendered[(name, suffix)])))
        expected = evaluate_case(md_qa, case)
        actual = evaluate_case(fmt_qa, case)
        assert actual.recall == expected.recall, case.id
        assert actual.abstained == expected.abstained, case.id


# ------------------------------------------------------------------- PDF only

pdf = pytest.mark.skipif(not _available(".pdf"), reason="pymupdf not installed")


@pdf
def test_pdf_drops_headers_footers_and_keeps_pages(rendered):
    path = rendered[("sample_contract.md", ".pdf")]
    doc = load_document(str(path))
    assert doc.page_count and doc.page_count > 1
    texts = [n.text for n in doc.nodes]
    assert not any("CONFIDENTIAL" in t or t.startswith("Page ") for t in texts)
    content = [n for n in doc.nodes if n.node_type == "paragraph"]
    assert all(n.page for n in content)
    assert {n.page for n in content} == set(range(1, doc.page_count + 1))
    assert all(n.metadata["source"].startswith(f"{path}#page=") for n in content)


@pdf
def test_pdf_merges_paragraph_split_across_pages(tmp_path):
    from renderers import render_pdf

    clause = " ".join(f"obligation number {i} applies to each party" for i in range(60))
    long_paragraph = f"The parties agree that {clause} without exception."
    path = render_pdf(f"# Agreement\n\n## 1. Obligations\n\n{long_paragraph}\n\nShort closing paragraph.\n",
                      tmp_path / "long.pdf")
    doc = load_document(str(path))
    assert doc.page_count > 1
    paragraphs = [n for n in doc.nodes if n.node_type == "paragraph"]
    assert [p.text for p in paragraphs] == [long_paragraph, "Short closing paragraph."]
    assert paragraphs[0].metadata["pages"] == list(range(1, doc.page_count + 1))


@pdf
def test_pdf_table_extracted(rendered):
    doc = load_document(str(rendered[("saas_agreement.md", ".pdf")]))
    tables = [n for n in doc.nodes if n.node_type == "table"]
    assert len(tables) == 1
    assert "| Enterprise | 4,000 USD | Unlimited |" in tables[0].text


@pdf
def test_pdf_figure_with_caption_and_numbered_heading_nesting(tmp_path):
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page(width=420, height=520)
    # Same font size and weight for "4." and "4.1": nesting must come from numbering.
    page.insert_text((50, 60), "4. Liability", fontname="hebo", fontsize=12)
    page.insert_text((50, 90), "4.1 Cap", fontname="hebo", fontsize=12)
    page.insert_text((50, 115), "Liability is capped at the fees paid.", fontname="helv", fontsize=10)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), 0)
    pix.clear_with(200)
    page.insert_image(fitz.Rect(50, 140, 150, 240), pixmap=pix)
    page.insert_text((50, 260), "Figure 1: Liability waterfall", fontname="helv", fontsize=9)
    page.insert_text((50, 290), "5. Indemnity", fontname="hebo", fontsize=12)
    path = tmp_path / "fig.pdf"
    doc.save(str(path))

    graph = DocumentGraph.from_file(str(path))
    cap = graph.lookup("4.1 Cap")
    assert graph.parent(cap.id).title == "4. Liability"
    assert graph.parent(graph.lookup("5. Indemnity").id).node_type == "document"
    figures = graph.content_nodes(("figure",))
    assert [f.text for f in figures] == ["Figure 1: Liability waterfall"]
    assert graph.parent(figures[0].id).id == cap.id


@pdf
def test_pdf_scanned_page_reports_ocr_status(tmp_path):
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page(width=420, height=520)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), 0)
    pix.clear_with(255)
    page.insert_image(page.rect, pixmap=pix)
    path = tmp_path / "scan.pdf"
    doc.save(str(path))

    parsed = load_document(str(path))
    assert any("no text layer" in w for w in parsed.warnings)


# ------------------------------------------------------------------ HTML / misc

def test_html_skips_page_furniture():
    blocks, title, _ = extract_html(
        "<html><head><title>T</title><script>x()</script></head><body>"
        "<nav>Home | Docs</nav><h2>1. Scope</h2><p>Body <b>text</b>.</p>"
        "<figure><img alt='diagram'><figcaption>Figure 2: Flow</figcaption></figure>"
        "<footer>(c) 2026</footer></body></html>")
    assert title == "T"
    assert [(b.kind, b.text) for b in blocks] == [
        ("heading", "1. Scope"), ("paragraph", "Body text."), ("figure", "Figure 2: Flow")]


def test_unsupported_extension(tmp_path):
    path = tmp_path / "x.rtf"
    path.write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(str(path))
