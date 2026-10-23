# GNN Pruner

Structure-aware document QA: parse a document into a hierarchy graph, retrieve
candidates (BM25 + dense + structural), expand to a local subgraph, score with
explainable heuristics, prune while keeping definitions and referenced clauses,
and build a cited evidence pack for an LLM.

```
Document -> Parse -> Graph -> Embeddings -> Query plan -> Candidates
        -> Graph expansion -> Scoring -> Pruning -> Evidence pack -> Answer
```

## Run

```bash
uv run --no-project --python 3.11 --with numpy --with scikit-learn --with pytest \
  --with pymupdf --with python-docx --with networkx pytest -q

PYTHONPATH=src uv run --no-project --python 3.11 --with numpy --with scikit-learn \
  python -m gnn_pruner examples/sample_contract.md "what is the definition of Losses" --explain --prompt
```

Embeddings: TF-IDF works with no extra installs but misses paraphrased questions. Install the dense
model (`pip install sentence-transformers faiss-cpu`) and the CLIs pick it up automatically
(`--embedder auto`, MiniLM). Use `--store graphs.db` to cache parsed graphs and embeddings.

Evaluate against the gold set (see [docs/USE_CASE.md](docs/USE_CASE.md)):

```bash
PYTHONPATH=src uv run --no-project --python 3.11 --with numpy --with scikit-learn \
  python -m gnn_pruner.evaluation eval/contracts_v1.jsonl --verbose
# compare embedders (needs sentence-transformers):
#   ... -m gnn_pruner.evaluation eval/contracts_v1.jsonl --compare tfidf st:sentence-transformers/all-MiniLM-L6-v2
```

```python
from gnn_pruner import DocumentQA

qa = DocumentQA.from_file("examples/sample_contract.md", llm=my_llm)  # llm: str -> str, optional
result = qa.ask("is the indemnity subject to the liability cap")
result.evidence.render_prompt(); result.answer; result.scored  # features, contributions, reasons
result.confidence.explain()  # is there evidence at all? DocumentQA(abstain=True) acts on it
```

Inputs: `.md`, `.txt`, `.pdf` (PyMuPDF), `.docx` (python-docx), `.html`. Scanned PDF pages
are OCR'd when [Tesseract](https://github.com/tesseract-ocr/tesseract) is installed
(`brew install tesseract`); otherwise the loader records a warning. To generate PDF/DOCX/HTML
versions of the example contracts, run `python tests/renderers.py`.

## Layout

| Module | Phase |
|---|---|
| `ingest/` | 1: PDF (font-size headings, header/footer removal, tables, figures, cross-page paragraphs, OCR), DOCX, HTML loaders -> blocks |
| `parser.py` | 1: blocks -> section / paragraph / sentence / table / figure nodes with offsets, pages, numbered-heading nesting |
| `document_graph.py` | 2: chapter/section/subsection/clause nodes; parent_child, next_prev, same_section, cross_reference ("Sections 4.1 and 4.2", "2(c)"), definition_link edges; `validate()`, `outline()`, JSON `save`/`load`, `to_networkx()` |
| `store.py` | 2-3: SQLite `GraphStore` for many documents; node/edge queries; `load_or_ingest` skips unchanged files; per-node embeddings + cross-document vector `search`; embedding cache |
| `inspect_graph.py` | 2: `python -m gnn_pruner.inspect_graph FILE --outline [--save g.json] [--store g.db]` |
| `embeddings.py` | 3: `load_embedder("tfidf" \| "st:<model>")`; query/document prefixes; `CachedEmbedder` (never embeds a text twice) |
| `index.py` | 3: contextual node embeddings ("section path. text"), section vectors, BM25, `records()` (node_id, vector, type, text, metadata), semantic_similarity edges tied to the model |
| `vector_index.py` | 3: exact numpy search, or FAISS (exact; HNSW above 10k vectors) |
| `query_planner.py` | 4: question type (rules + semantic fallback), defined-term / section / comparison-side / condition resolution against the document, retrieval strategy (`hybrid`, `scoped`, `definition_first`, `follow_references`, `per_side`, `section_summary`, `pairwise`); `plan.explain()`, CLI `--plan` |
| `retrieval.py` | 5-6: hybrid candidates (weighted RRF), best-first graph expansion |
| `scoring.py` | 7: 12 node features, `LinearScorer` with per-feature contributions (`--explain`), pluggable `Scorer`; `EvidenceConfidence` abstention signal with per-embedder calibration (`calibration.json`) |
| `pruning.py` | 8: adaptive budget (score floor + gap), low-signal / provenance / sibling filters, dependency rules (continuations, definitions, both ends of references, qualifiers, numeric conflicts), drop reasons; `PruneConfig`, per-embedder defaults |
| `prompt.py` | 9: topology routing (PATH / SIBLING / SYNTHESIS / QA), route templates, per-route token budget |
| `evidence.py` | 9-10: evidence pack (roles, citations, excerpts, citation map), rendered prompt, LLM or extractive answer |
| `answer.py`, `answer_eval.py` | 10: Claude / OpenAI / extractive providers, grounded `Answer` with citation validation and abstention; answer-quality readings (`python -m gnn_pruner.answer_eval`, `--dry-run` for cost) |
| `pipeline.py` | `DocumentQA` end-to-end |
| `layers.py` | G1: spec four-layer DAG view (L1 sections … L4 claims), hierarchical / sequential / reference edges, GVRM validate-and-repair |
| `parquet_store.py` | G1: Parquet shards (nodes, edges, embeddings) with memory-mapped reads |
| `corpus.py` | G1: generated training contracts (any length, up to 10,000+ pages) with known-answer questions; `python -m gnn_pruner.corpus` |
| `silver.py` | G1: Silver Path labels (BM25 + dense fallback, up to L1) and label-quality readings; `python -m gnn_pruner.silver` |
| `ingest/docling_loader.py` | G1: Docling parser (`parser="docling"`) |
| `gnn/cross_attention.py` | G4: CrossEdgeAttention (spec §4.4, windowed local attention + global summary); `python -m gnn_pruner.gnn.cross_attention train|evaluate` |
| `gnn/g6.py` | G6: spec tables 8–10 (scale, quality, adversarial suite, human-preference sheet); `python -m gnn_pruner.gnn.g6 scale|quality|adversarial|sheet` |
| `gnn/distributed.py` | G5: tree-cut / METIS partitioning, spec or exact boundary replication, Parquet shards, Ray workers + aggregator; `python -m gnn_pruner.gnn.distributed bench` |
| `gnn/two_stage.py`, `gnn/spec_qa.py`, `gnn/bench.py` | G3: coarsening (pooled paragraph super-nodes, keep rule) + FineGNN (spec §3.4) + merge; the spec path end to end (`python -m gnn_pruner.gnn.spec_qa`); scale benchmark |
| `gnn/` | G2: CoarseGNN (typed GraphSAGE) on the coarse graph; `python -m gnn_pruner.gnn.train` (supervised + PPO), `python -m gnn_pruner.gnn.evaluate` (readings), `GNNScorer` (`--scorer gnn`). Needs torch + torch_geometric. |
| `evaluation.py` | 0+: gold-set metrics (recall, hit@1, precision, abstention); `--ranking` (MRR, nDCG), `--fit-calibration`, `--abstain` |

## Status

The full workflow (upload -> document tree -> relevance gate -> chapter workers -> central
board -> cited answer) is described in `WORKFLOW.md`; what is built and what is next is in
`REQUIREMENTS.md`, and every measurement is in `docs/READINGS.md`.

## Data and licences

Large data is not in the repository; the build scripts recreate it:

| Data | Source | Licence | Rebuild |
|---|---|---|---|
| `data/longbench/` (contract data rooms, book libraries) | CUAD v1 (The Atticus Project) and NarrativeQA test set (DeepMind) with Project Gutenberg texts | CUAD: CC BY 4.0; NarrativeQA: Apache 2.0; Gutenberg texts: public domain in the US | `python -m gnn_pruner.longbench build` |
| `eval/long/*.jsonl` (questions; CUAD clause spans and NarrativeQA questions / answers) | as above | as above | written by the same build |
| `data/corpus_v1/` (generated contracts) | generated | — | `python -m gnn_pruner.corpus` |
| `models/` (trained GNN weights) | trained here | — | `python -m gnn_pruner.gnn.train ...` (see REQUIREMENTS G2–G4) |

API keys are read from the environment (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) and are never stored in the project.
