# GNN PRUNER — Requirements & Phase Tracker

Source of truth for what to build next. Work proceeds **one phase at a time**; a phase
is "done" only when every checklist item is implemented, tested, and evaluated.

**Target architecture:** *GNN-Pruner Architecture Specification* (Definitively
Corrected Edition), adopted 2026-10-17. Keep a copy at `docs/GNN_Pruner_v3_Architecture.pdf`.
We implement the spec as written. Where it is contradictory or incomplete, the decision
taken is recorded under "Design decisions" below.

**Goal:** build the v3.0 system, take our own readings at every stage, and make those
readings as good as we can. The spec's figures (95% Recall@10, 94% cross-chapter recall,
80 ms, 5% of tokens) are reference points, not pass/fail gates. Every phase records its
measured readings in the readings log (`docs/READINGS.md`), with the setup stated
(documents, size, embedder, hardware).

**End-to-end flow (spec §5)**

1. Document ingestion: 4-layer DAG, Parquet
2. Graph coarsening: paragraph super-nodes, METIS partitioning
3. Distributed coarse GNN: GraphSAGE, boundary replication
4. Fine-grained expansion: FineGNN with parent context
5. Cross-edge attention: local window plus global summary
6. Adaptive merging: threshold, stitching
7. Topology-routed prompt
8. Final LLM

Training: Silver Paths → supervised pre-training → PPO → curriculum → feedback retraining.

**v4 (adopted 2026-10-22): chapter workers + central board.**
- **Design:** full design in `docs/ARCHITECTURE.md`. It is the user's design, built on top
  of v3.0, not replacing it.
- **Tree:** the document becomes a tree: book (metadata) → chapters → sections →
  subsections → original text, with summaries / key facts on every node.
- **Workers:** one worker per chapter. A free local gate decides relevance; relevant
  chapters are searched step by step with a reasoning model.
- **Board:** workers post word-for-word quotes to a central board, which verifies them.
- **Answer:** a final LLM answers only from accepted quotes, with page / section
  citations.
- **v3.0's role:** v3.0's pruner and GNN become the worker gate.
- **Goal:** grounded answers on documents larger than any model's context window,
  proven against a long-context LLM on a long-document benchmark.

The rule-based pipeline built in Phases 0–8 (planner → hybrid retrieval → expansion →
linear scoring → rule pruning) is **not** part of the spec's inference path. It stays in
three roles:
- the dense / BM25 source for Silver Paths
- the baseline every reading is compared against
- the fallback when no trained model is present

**Domain:** legal contracts first (see `docs/USE_CASE.md`); the spec also targets
financial and technical documents.

---

## Working rules

1. One phase per request. Treat the phase as its full scope below, not just an MVP.
2. Every phase ships with: code, unit tests in `tests/`, and readings from `eval/`,
   logged in `docs/READINGS.md`.
3. Every spec component is compared with the rule-based baseline on the same evals, and
   with itself switched off (ablation: the spec's "v2.0" column = v3.0 without
   coarsening / cross-edge attention). Improve the component until it is as good as we
   can make it; report the readings honestly, whatever they are.
4. **Test sets are never used for training, labels, or tuning:**
   `eval/contracts_v1.jsonl`, `eval/contracts_scale_v1.jsonl`,
   `eval/planner_holdout_v1.jsonl`, and any set marked *test* later. Silver labels and
   tuning use training documents or dev sets (`eval/calibration_v1.jsonl`,
   `eval/dev_*.jsonl`, `eval/planner_v1.jsonl`), split by document. This keeps the
   readings real.
5. At the end of a phase, report what is done and what remains, then update the status
   table below.
6. Environment:
   - System Python is 3.9, so run via `uv run --no-project --python 3.11 --with <deps> ...`.
   - torch 2.14 and torch_geometric 2.8 are already cached by uv.
   - Pin `pymupdf<2`: PyMuPDF 2.0 changed table/text extraction and breaks 3 PDF ingest
     tests (found 2026-10-20). Supporting 2.0 is an open item.
     On the `uv run` command line write `--with pymupdf~=1.26`, not `pymupdf<2`: Ray
     re-runs that command through a shell for each worker, and `<2` becomes a redirect.
   - HF models are cached in `~/.cache/huggingface`.
   - The network is slow, so start large installs in the background early.
   - Hardware is one Mac, CPU only. Ray runs locally with several worker processes: the
     same code as a cluster, but not a multi-machine deployment. The spec's 4× A100 is
     replaced by CPU training at model sizes that fit.

---

## Status

**Foundation (rule-based pipeline: baseline, Silver Path source, fallback)**

| Phase | Name                        | Status |
|-------|-----------------------------|--------|
| 0     | Define the use case         | ✅ Done (real-user questions still missing) |
| 1     | Ingestion & parsing         | ✅ Done |
| 2     | Document graph              | ✅ Done (covers the spec's 4-layer DAG) |
| 3     | Embeddings & metadata       | ✅ Done |
| 4     | Query planner               | ✅ Done |
| 5     | Hybrid retrieval            | ✅ Done (2026-10-11) |
| 6     | Graph expansion             | ✅ Done (2026-10-11) |
| 7     | Relevance scoring           | ✅ Done (2026-10-14) |
| 8     | Adaptive pruning            | ✅ Done (2026-10-17) |

**v3.0 build (spec), in order**

| Phase | Name                                                     | Spec      | Status |
|-------|----------------------------------------------------------|-----------|--------|
| G1    | Corpus, 4-layer export, Silver Path generation           | 2.1, 2.2  | ✅ Done (2026-10-17) |
| G2    | CoarseGNN: supervised pre-training + PPO                 | 2.2, 2.3, 3.1 | ✅ Done (2026-10-17) |
| G3    | Coarsening + FineGNN (two-stage scoring and merge)       | 3         | ✅ Done (2026-10-17) |
| G4    | Cross-edge attention (local window + global summary)     | 4         | ✅ Done (2026-10-18) |
| G5    | Distribution: METIS, boundary replication, Ray, Parquet  | 2.3, 5    | ✅ Done (2026-10-20) |
| G6    | Benchmarks: 100 → 10,000 pages, full readings            | 7, 8      | ✅ Done (2026-10-21) |
| 9     | Topology-routed prompt + evidence pack                   | 5 (ph. 7) | ✅ Done (2026-10-21) |
| 10    | Final LLM inference (+ MCP sampling)                     | 5 (ph. 8) | ✅ Done (2026-10-22) — live readings on Gemini (free); MCP sampling → Phase 12; paid-model runs optional |
| 11    | Training pipeline: curriculum + feedback retraining      | 5 (training), 9 | ⬜ (merged with v4-6) |
| 12    | Production hardening: FastAPI, dashboard, regression suite | 6, 9    | ⬜ |

**v4 build (user design, `docs/ARCHITECTURE.md`), in order**

| Phase | Name                                                              | Architecture § | Status |
|-------|-------------------------------------------------------------------|----------------|--------|
| v4-1  | Tree + enrichment: TOC structure, virtual chapters, root metadata, bottom-up summaries / key facts | 2, 3 | 🟡 Built and measured; human fact spot check pending |
| v4-2  | Long-document benchmark (100 → 10,000+ pages) + baselines 1–3     | 8              | 🟡 Built; S/M/L retrieval measured, S answers measured; M/L answers and the strong full-document baseline (Claude) pending |
| v4-3  | Central board + chapter workers (descend loop, quote checks, references) | 4 [7]–[9] | 🟡 Built and tested; first readings (contracts S); run 2 resumes on quota reset |
| v4-4  | Worker gate (local score + TOC screen) + 5-system comparison      | 4 [8], 8       | ⬜ |
| v4-5  | Entity links for books                                            | 2.3            | ⬜ |
| v4-6  | Learning loop: board labels retrain the gate (= Phase 11)         | 4 [13]         | ⬜ |
| v4-7  | Demo app: local chat interface (Streamlit) with citation viewer  | WORKFLOW B8    | ⬜ Planned; starts once the core engine is proven |
| R     | **Research track: "Complete or Silent"** (closure + absence certificates, trap benchmark, paper) | `docs/PAPER_PLAN.md`, `docs/COMPLETE_OR_SILENT.md` | ⬜ Planned: P0–P7, ~5–7 weeks focused; P4 needs ~$200–400 API budget |

**Next up: the research track (`docs/PAPER_PLAN.md`): P0 literature review + P1 certificates + P2 trap benchmark, in parallel. It includes proving the core engine. Finish the v4-3 / v4-4 readings (books, judge), fix the open weaknesses (widening cost on unanswerable questions, two-hop facts, story-level questions), and run the strong whole-document baseline (Claude, budget pending). Then v4-7 (demo app). v4-1 waits on the human fact spot check.**

---

## Design decisions (where the spec is contradictory or incomplete)

| # | Spec issue | Decision |
|---|---|---|
| D1 | Latency: the headline 80 ms (30 + 20 + 30) does not match the phase table sum (~116 ms) | Measure and report every stage separately, plus the total. Both spec figures are kept as reference lines in the readings. |
| D2 | Keep rule: "top 10–20%" vs "score > θ · mean" (Stage 1), and "> 80% of global mean" vs "< 5% of nodes" (merging) | Implement the spec rules as written, with θ and the percentages configurable. Stage 1 = θ · mean, capped to the top 10–20%. Merging = > 0.8 · global mean. θ is tuned on dev for the best recall at the smallest share kept. The share of nodes kept is reported against the 5% reference. |
| D3 | Nodes per page disagree (5 vs 50 sentences per page) | Benchmark documents are generated by page count (~500 words per page). Actual sentence, paragraph and node counts are recorded, not assumed. |
| D4 | The GNN formulas and code take no query input, so every question would get the same scores | Each node's input is its embedding concatenated with the query embedding and query–node similarity features. This is the minimum addition needed for retrieval to work. |
| D5 | PPO is named but its actions and reward are undefined | Two stages. (1) Supervised pre-training on Silver Path labels. (2) PPO fine-tuning: the policy makes top-down keep/expand decisions (Layer 1 → 4), and the reward is overlap with the Silver Positive Set, with a size penalty. Readings are kept for both stages. |
| D6 | `CrossEdgeAttention` uses `self.alpha` without defining it | `self.alpha = nn.Parameter(torch.tensor(0.3))`, matching FineGNN. |
| D7 | "DocSlicer" and "GVRM validation" could not be identified | Docling as the spec parser (alongside the existing loaders, compared on the evals). GVRM = `DocumentGraph.validate()` plus repair of broken nesting and offsets. Replace these if the user provides a definition. |
| D8 | "Exactly four layers" vs documents without subsections, plus tables and the root | Map onto the 4 layers: L1 chapter/section, L2 sub-section, L3 paragraph/table, L4 sentence/clause. A missing level is left empty; a paragraph with one sentence gets one L4 node. |
| D9 | The spec has no planner or candidate retrieval step | The spec path scores the whole (coarsened) graph. The rule-based pipeline is the baseline, Silver Path source and fallback (see above). |
| D10 | 4× A100, multi-machine Ray, 100,000 pages on 100 workers | CPU training. Ray local cluster with N worker processes. Benchmarks up to the largest size that fits in memory on this Mac, at least 10,000 pages. |
| D12 | Prompt routing: the rule list (depth first) contradicts the worked example (depth 4 + 2 cross-refs → SYNTHESIS) | Spec thresholds (> 3, > 3, > 2) with precedence SYNTHESIS > SIBLING > PATH > QA. Cross-refs = clause citations between kept items, not definitions (nearly every clause uses a defined term, so counting them would route everything to SYNTHESIS). Depth = chapter → … → item levels. |
| D11 | Metrics the spec does not define: structural coherence, adversarial robustness, human preference | Defined in G6: structural coherence = share of kept nodes whose ancestors and required dependencies are kept; adversarial = a written suite of distractor/negation/paraphrase queries; human preference = a rating sheet for you to fill in. |

---

## Readings log

Measured values go in `docs/READINGS.md`, one table per phase, with spec reference
values alongside:
- Recall@10
- cross-chapter recall
- nodes kept %
- tokens to the LLM %
- per-stage latency P50 / P99
- training time
- cost per query

Current baseline readings (rule-based pipeline, MiniLM, 2 contracts, 36 answerable):

| Reading | Value | Spec reference |
|---|---|---|
| Evidence recall (small / scale set) | 1.0 / 1.0 | 95% Recall@10 |
| Scored recall@5 | 0.958 / 0.944 | — |
| Pack words as a share of the document | ~23% short contracts, ~2% on 4,500-word docs | 5% of tokens |
| Latency | A few ms per question on ≤ 358 nodes | 80 ms on 10,000 pages |
| Cross-chapter recall | Not yet measurable (G4 adds the set) | 94% |

---

## Foundation — completed phases (summary)

Details, tables and decisions are in `docs/USE_CASE.md`.

- **Phase 0:** Legal-contracts domain. Gold set `eval/contracts_v1.jsonl` (40 cases:
  36 answerable, 4 unanswerable). *Open:* real-user questions.
- **Phase 1:** PDF, DOCX and HTML loaders with OCR fallback. Nodes carry id, parent,
  type, text, page, offsets and metadata.
- **Phase 2:** Chapter, subsection, paragraph, clause and sentence nodes. Edges:
  hierarchical, sequential and reference, plus same_section and definition_link. JSON and
  SQLite GraphStore; `validate`/`outline`; `inspect_graph` CLI.
- **Phase 3:** Embedder interface and CachedEmbedder (MiniLM, mpnet), FAISS/numpy index.
  Dense recall 1.0 vs 0.79 for TF-IDF.
- **Phase 4:** Doc-aware planner with 7 strategies and a semantic fallback. Dev set 1.0,
  held-out 0.739.
- **Phase 5:** Hybrid retrieval (BM25, phrase, dense, structural, plan and graph sources)
  fused by weighted RRF.
- **Phase 6:** Best-first bounded expansion with a recorded path per node.
- **Phase 7:** 12-feature `LinearScorer` with contributions, behind `Scorer`. Calibrated
  evidence-confidence signal.
- **Phase 8:** Adaptive floor + gap pruning with recorded drop reasons, and dependency
  rules (continuations, definitions, references, qualifiers, numeric conflicts).

Reused by the v3.0 build:
- the `Scorer` interface and `subgraph_features` (GNN input features)
- BM25 + dense retrieval (Silver Paths)
- `--ranking` metrics
- Phase 8 dependency rules (kept as a post-merge step)
- the evidence-confidence signal (Phase 10 abstention)
- `make_scale_docs.py` (G1 corpus generator)

---

## G1 — Corpus, 4-layer export, Silver Paths (spec §2.1, §2.2) ✅

**Goal:** the documents, graph format and training labels everything else learns from.

- [x] **Training corpus** `data/corpus_v1`:
  - 48 generated contracts (`corpus.py`), 2–120 pages each (985 pages, 503k words),
    split by document 38 train / 10 val
  - the generator scales to 10,000 pages (510k nodes, valid four-layer DAG)
  - new topics and templates, with a leakage test against every `examples/` document and
    eval question
  - CUAD (optional) is not added
- [x] **Four-layer DAG** (`layers.py`, D8): L1–L4 tags, virtual L4 claims, hierarchical,
      sequential and reference edges. **GVRM** (`validate_and_repair`, D7). **Parquet**
      shards (`parquet_store.py`: nodes, edges and embeddings, memory-mapped reads).
      Shard-load times recorded.
- [x] **Docling parser** (`ingest/docling_loader.py`, `parser="docling"`, `--parser`),
      compared with the native loaders on `contracts_v1` in all four formats
- [x] **Training questions:** 1,340 with known answers. Lexical, paraphrase (topic and
      verb words replaced), definition and cross-chapter (a duty plus its exception in
      another chapter).
- [x] **Silver Paths** (`silver.py`): BM25 top-10 L4 leaves plus the dense fallback,
      walked up to L1 → Silver Positive Set. Graded labels, 20 hard negatives each.
      Modes bm25 / fallback / hybrid; fusion weighted by BM25 term coverage.
      `silver_{train,val}.jsonl` are written in hybrid mode.
- [x] Readings (`docs/READINGS.md`):
  - hybrid path recall 0.947 train / 0.966 val (BM25 only: 0.68 / 0.70)
  - paraphrases 0.90 / 0.91; cross-chapter 0.83 / 0.94
  - dev sets: 1.0, 1.0, 1.0 and 0.91
  - corpus build 58 s
- [x] Tests (`tests/test_g1.py`): layer mapping and edges, GVRM repairs, Parquet round
      trip, deterministic generation with answers present exactly once, leakage, split by
      document, Silver Path invariants, Docling structure

*Carried forward:*
- Cross-chapter questions are the weakest label group (0.83 train), because the spec's
  Silver Path walks the hierarchy only. G4 (cross-edge attention) and its cross-chapter
  question set build on the `cross_chapter` tag already in the corpus.
- The generated corpus is templated. Real-document readings come from the dev and test
  sets. CUAD would add real contracts for training.
- Silver precision is low by construction (10 leaves for 1–2 answers); the grades carry
  the strength.

## G2 — CoarseGNN: supervised + PPO (spec §2.2, §2.3, §3.1) ✅

**Goal:** a trained GraphSAGE node scorer, conditioned on the query (D4).

- [x] `CoarseGNN` (`gnn/model.py`): typed GraphSAGE (one SAGEConv per edge type),
      residual layers and a skip connection. Input (`gnn/features.py`, D4): embedding,
      embedding × query, cosine, BM25, coverage, layer/kind/position.
- [x] **Stage 1, supervised** on Silver Positive Sets: listwise loss with sharpened
      targets (τ = 0.1) plus graded BCE, with early stopping on the validation
      documents. Settings chosen by a 7-run sweep and 3 seeds per architecture.
- [x] **Stage 2, PPO** (D5): top-down keep/drop policy (L1 → L2 → L3). Reward per
      layer = silver grade kept − size penalty × share kept, weighted (0.25, 0.25, 1.0).
      Clipped objective, entropy bonus, supervised anchor.
- [x] `GNNScorer` (`gnn/scorer.py`) behind `Scorer` (`--scorer gnn`): whole-graph
      scoring, gradient × input explanations per feature group plus neighbours, and
      `LinearScorer` when no model is configured. `score_subgraph` supports scorers that
      score all nodes at once.
- [x] Model artefacts `models/coarse_gnn_v2/{supervised,ppo}`: weights, config,
      settings, corpus and silver hashes, history
- [x] Readings (`gnn/evaluate.py`, `docs/READINGS.md`):
  - paragraph Recall@10, MRR, nDCG@10 and hit@1 for BM25, dense, the rule-based
    pipeline, supervised, PPO and a gold-label upper bound, on corpus val, 4 dev sets
    and 2 test sets
  - ablations: no edges, no query, untyped, skip
  - policy kept share vs answers kept, training time, latency, end to end
- [x] Tests (`tests/test_gnn.py`): features, metrics, training fits and is seeded,
      policy hierarchy and rewards, scorer in the pipeline, embedder mismatch

**Result:**
- **In domain** (generated validation documents, multi-passage dev sets) the GNN is the
  best ranker: MRR 0.794 vs 0.776 rule-based (0.955 vs 0.864 on dev_pruning).
- **On the real test contracts** it is second to the rule-based pipeline (0.935 vs 0.965;
  long documents 0.852 vs 0.918), but well ahead of BM25 and dense alone.
- **PPO policy on long documents:** keeps 2.1% of paragraphs and 90% of answers.
- The question input is essential (without it MRR is 0.09); edge types beat the spec's
  untyped GraphSAGE.
- Fix along the way: FAISS and PyTorch clash on macOS (two OpenMP runtimes). FAISS runs
  single-threaded, and "auto" uses numpy, once PyTorch is loaded.

*Carried forward:*
- **Domain shift to real contracts.** The gold-label model is no better on test (0.908),
  so the gap is the templated training documents, not label noise. Remedies:
  - real training contracts (CUAD)
  - give the GNN the planner/linear signals (definition, cited-section and fusion
    features) the rule-based pipeline uses
  - an ensemble with `LinearScorer`
- **Silver noise costs ~0.14 MRR in domain** (gold-trained 0.93). G4's cross-chapter
  links should improve the labels.
- **End to end with the GNN scorer**, packs are 30% smaller but recall is 0.94 / 0.89.
  The Phase 8 thresholds need re-tuning on dev for GNN probabilities (G3 merging
  replaces this path anyway).
- Run-to-run variation is ±0.01 MRR; report means over seeds for close calls.

## G3 — Graph coarsening + FineGNN (spec §3) ✅

**Goal:** the spec's three-stage scoring.

- [x] **Stage 1** (`gnn/two_stage.py`): paragraph super-node = mean of its sentence
      embeddings (`paragraph_vectors`, pooling `mean` / `direct`). CoarseGNN is retrained
      on pooled nodes. Keep rule (D2): `p > θ · mean`, capped at the top 20%, minimum 5;
      θ chosen on corpus val. Optional absolute cap `max_keep`. Attention pooling was not
      built: mean and direct pooling are within noise, so the spec's fallback was not
      needed.
- [x] **Stage 2:** `FineGNN` as in spec §3.4 (context projection, trainable alpha = 0.3,
      GraphSAGE 2 layers, output projection), plus the query features (D4) and the parent
      coarse probability. The fine graph holds the kept paragraphs, their sentences and
      headings, with hierarchical, sequential (re-linked) and reference edges. Trained
      with the coarse model frozen, then jointly.
- [x] **Stage 3:** sentences of discarded paragraphs are dropped; sentences above
      0.8 × mean are merged. The spec path end to end (`gnn/spec_qa.py`): stages →
      merge → Phase 8 dependencies → evidence pack.
- [x] Readings (`docs/READINGS.md`):
  - stage-1 recall, nodes per stage, merged share and recall, sentence MRR
  - comparisons: no coarsening (spec v2.0), no context, direct pooling, staged vs
    joint, dense / BM25
  - end-to-end evidence metrics
  - scale benchmark at 100 / 1,000 / 10,000 pages (`gnn/bench.py`)
- [x] Tests (`tests/test_g3.py`): pooling, keep rule, discard rule, fine inputs and
      context, FineGNN shape and trainable alpha, stages and merge, spec path end to end

**Result:**
- **Coarsening:** ~90% fewer nodes in the fine stage at about −1 point of MRR vs no
  coarsening (spec reference +2–3%).
- **Merged output on long documents:** 2.5% of sentences, 93% of answers.
- **Spec path end to end:** evidence recall 0.986 / 0.958 on the test sets, coarse + fine
  3–5 ms.
- **10,000 pages:** 4.1 s with the spec's percentage rule, 1.34 s with `max_keep = 200`
  (same recall). The no-coarsening baseline does not fit in 8 GB at that size.

*Carried forward:*
- **The coarse pass dominates at 10,000 pages** (1.2 s over 239k nodes). This is what G5
  distributes.
- **Stage-1 misses** drop 1–2 test questions (stage-1 recall 0.958). Raising θ / cap
  costs fine nodes; better coarse ranking (real training data, planner signals; see G2)
  is the lever.
- **End-to-end hit@1 trails the rule-based pipeline** (0.83 vs 0.94), the same domain
  shift as G2.
- **Parent-context injection and the alpha weight show no clear benefit** at this scale;
  the spec component is kept.

## G4 — Cross-edge attention (spec §4) ✅

**Goal:** cross-chapter relevance.

- [x] Cross-chapter question sets:
  - dev: `eval/dev_crosschapter_v1.jsonl` and `_scale_v1`, 10 real-contract questions
  - test: automatic 2+-chapter subsets of `contracts_v1` / `contracts_scale_v1`
  - corpus: tagged questions
  - metric: complete (every chapter's answer) at paragraph top-10, stage 1, merge and
    end to end
- [x] **Local sparse attention:** 512-node sliding window, exact band attention computed
      in blocks (O(n · window)), using document positions so G5 workers with
      non-contiguous rows stay exact
- [x] **Global summary:** mean of the top-K (100) hidden states. The §4.2 score
      `local + head + β·sigmoid(cos(h, GS))`, with the global boost bounded to [0, β].
      `CrossEdgeAttention` as in spec §4.4, with D6 applied.
- [x] Trained on top of G3 (curriculum: GNN frozen, then joint), at the coarse and/or fine
      stage; 3 seeds plus local-only, global-only, coarse-only and fine-only ablations
- [x] Readings (`docs/READINGS.md`): cross-chapter complete with and without attention,
      vs the rule-based reference following; end to end; latency overhead at 1,000 /
      10,000 pages
- [x] Tests (`tests/test_g4.py`): spec module shape and D6, windowed = band attention,
      locality, bounded boost, external global summary, ablations, save/load,
      pipeline / spec path

**Result:**
- **In domain:** cross-chapter complete@10 0.85 → 0.92, stage-1 complete 0.95 → 1.0,
  MRR +0.013 (3 seeds).
- **On the real contracts:** within noise (7–10 questions). End-to-end complete evidence
  rises (scale test 0.944 → 0.972) with packs 15–25% larger.
- **Cost:** the coarse stage costs 5× more on CPU (1,000 pages: 72 → 376 ms), but the
  fine stage shrinks, so 10,000 pages stays at ~8 s.

*Carried forward:*
- **The rule-based reference following still wins cross-chapter on the real long test
  documents** (1.0 vs 0.71). Domain shift again (G2). Options:
  - reference-aware Silver Paths (walk reference edges as well as the hierarchy)
  - real training contracts
- **Attention latency on CPU is far above the spec's 30 ms.** A global-only variant
  costs ~1 ms for most of the benefit.

## G5 — Distribution (spec §2.3, §5) ✅

**Goal:** the spec's distributed execution, running on a local Ray cluster (D10).

- [x] **Partitioning** (`gnn/distributed.py`): hierarchical tree-cut by chapter
      (contiguous, balanced) and METIS (`pymetis`)
- [x] **Boundary replication**, two modes:
  - `spec`: last/first N = 3 nodes per shard edge, exactly as written; no cross-worker
    calls during the GNN pass
  - `exact`: attention window plus 2-hop halo; distributed = single-process
- [x] **Ray actor per shard** (plus an in-process backend) and a central aggregator.
      Protocol:
  1. stats reduce (feature normalisation)
  2. GNN pass + local top-K
  3. GlobalSummary broadcast + cross-edge attention
  4. keep rule (D2), FineGNN, adaptive merge (> 0.8 × global mean), boundary stitching
- [x] **Parquet shards** (nodes with embeddings and structure, edges, document-level BM25
      slice), memory-mapped. The 1,000 / 10,000-page documents are cached under
      `data/bench/`.
- [x] **Parity:** distributed = single-process to ≤ 1e-5 in exact mode (tests: 1/3/4
      workers, tree and METIS, plain and attention models, real Ray actors; bench: 1,000 and
      10,000 pages)
- [x] Readings (`docs/READINGS.md`): latency vs workers, halo mode and partitioner; shard
      load / write / startup; aggregator rounds; parity and top-50 agreement; recall
- [x] Tests (`tests/test_g5.py`, 13)

**Result:**
- **10,000 pages run distributed on 8 GB.**
  - Exact mode is exact but gets slower with more workers (halos +43–96%).
  - Spec N = 3 mode with 4 workers is the fastest: coarse 5.8 s, total 7.9 s, 1.3× over
    1 worker. Its logits drift by up to 4.7, but recall is unchanged.
- **On one machine the gain is memory per worker, not speed.**

*Carried forward:*
- **The spec's N = 3 replication is approximate.** A smaller exact halo would need a
  graph shape where headings are not hubs. Options: replicate heading summaries instead
  of whole chapters, or 1-hop halo plus cached heading states.
- **The aggregator does 2–3 s of work at 10,000 pages** (fine stage, merge). Use
  `max_keep` (G3) and move the fine stage to workers when it matters.
- **A real multi-machine run** (or GPU workers) would be needed to measure the spec's
  speed claims.
- **Ray packages the working directory under uv.** Large `data/` and `models/` folders
  are excluded, and workers receive absolute paths.

## G6 — Benchmarks and full readings (spec §7, §8) ✅

**Goal:** the spec's tables, filled in with our measurements.

- [x] Documents of 100 / 1,000 / 10,000 pages (5.4k / 51k / 511k nodes measured, D3).
      Larger documents don't fit: 10,000 pages already needs one process per system on
      8 GB.
- [x] **Table 8:** latency p50 / p99 per size for v3.0, v3.0+cap, G3, "v2.0" and
      rule-based (`gnn/g6.py scale`)
- [x] **Table 9:**
  - Recall@10, structural coherence and cross-chapter complete@10 on test / dev /
    generated sets
  - adversarial robustness on the new 100-query suite (`eval/adversarial_v1.jsonl` +
    `_base_v1`)
  - human preference: a blind A/B sheet (`docs/human_preference_sheet.md`), rated by you:
    **50%** (6 / 6 / 8 ties), all preferred packs sufficient, shorter packs preferred 11/12
- [x] **Table 10:** pack tokens vs document tokens (WordPiece), throughput, and the cost
      formula (prices left to you)
- [x] `docs/READINGS.md` complete, with spec reference values next to ours

**Result:**
- **Speed at 10,000 pages:** G3 4.7 s, v3.0+cap 6.7 s, v3.0 9.9 s vs rule-based 167 s
  (spec 80 ms on GPUs).
- **Real test contracts:** rule-based still best (Recall@10 1.0, cross-chapter 1.0 vs
  0.71). The GNN path is best on generated documents (cross-chapter 0.925 vs 0.70).
- **Packs:** a constant ~370 tokens, i.e. 0.007% of a 10,000-page document (spec 5%).
- **Robustness:** recall is kept under all 5 perturbations (except typos, 0.95); hit@1
  drops 12–17%.
- **Fixed:** Phase 8 conflict detection is now indexed (~37× faster, identical results).

*Carried forward:*
- **Human preference ties at 50%;** raters prefer shorter sufficient packs. Pack length (Phase 9
  token budget, merge threshold for lookups) is the lever. A vs-full-context rating needs
  Phase 10.
- **Cross-chapter recall on real contracts** (stage 1 drops one chapter). Candidates:
  - reference-aware Silver Paths
  - a keep rule that protects cited sections
  - real training contracts
- **The rule-based pipeline's index doesn't scale** past ~1,000 pages (O(n²) semantic
  edges, 6-minute build at 10,000).
- **The spec's speed claims need GPUs / multiple machines** to test.

## Phase 9 — Topology-routed prompt + evidence pack (spec §5 phase 7) ✅

- [x] **Topology analysis** of the kept evidence graph (`prompt.analyse`): depth,
      max siblings per section, clause citations between items. Routes: PATH (depth > 3),
      SIBLING (> 3), SYNTHESIS (> 2), otherwise QA, with the precedence of D12. One
      template per route, deterministic.
- [x] **Pack contents:**
  - exact clause text with source offsets and page
  - section path (chapter > section titles)
  - Phase 8 roles in words ("qualifies E1", "continues E2", "may conflict with E1",
    "defines a term used in E3", "cited by E1")
  - citations as `-> E#`, or a 25-word excerpt when the cited clause isn't in the pack
  - definitions and tables (markdown)
- [x] **Prompt rules:** answer only from the evidence, cite label and section, say "The
      document does not say" when the evidence is insufficient, report conflicts. The
      Phase 7 confidence level is passed in. Answer / Evidence / Confidence format.
- [x] **Token budget** (`prompt.apply_budget`):
  - per-route budgets chosen on the dev sets (QA / SYNTHESIS 250, SIBLING / PATH 300)
  - primaries by score, each with its own dependencies; dependencies drop with their
    primary
  - sentence-boundary truncation
  - `DocumentQA(budgets=...)` / `SpecQA(budgets=...)`; `{}` disables
- [x] **Tests** (`tests/test_prompt.py`): each route, budget priority / orphan drop /
      truncation, deterministic rendering (unit and end to end), citations map back to
      node ids and offsets

**Result:**
- **Real contracts:** packs are already 98–159 tokens; the budget never binds; recall is
  unchanged (rule 1.0, v3.0 0.986). Prompts are ~330–360 tokens in total.
- **Deeper generated documents:** all four routes occur; the budget saves ~10% of tokens
  for −1 point of recall.

*Carried forward:*
- **Whether routed templates beat a single generic prompt** needs LLM answers (Phase 10).
- **PATH never occurs on the real test contracts** (they are shallow).
- **Token counts are an estimate.** Pass the target LLM's tokenizer as `counter` once one
  is wired.

## Phase 10 — Final LLM inference (spec §5 phase 8) ✅

- [x] **Pluggable providers** (`answer.py`, `--llm`, `DocumentQA(llm=...)`):
  - **Claude:** Anthropic SDK; default `claude-opus-5-5`, explicit effort `medium`, JSON
    schema via `output_config.format`, server-side refusal fallback `fallbacks: "default"`
    (turn off with `fallback=False`)
  - **OpenAI:** Chat Completions + JSON-schema `response_format`; model named explicitly
  - **Gemini:** `gemini:<model>` via Google's OpenAI-compatible endpoint
    (`GEMINI_API_KEY`); paced for the free tier; a daily-quota 429 stops the run cleanly
    (`QuotaExhausted`)
  - **Ollama:** `ollama:<model>` (local). Built, but local models froze the 8 GB laptop;
    not used for readings
  - **Extractive:** offline fallback; the older `prompt -> str` callable still works
- [ ] **MCP sampling (client-side LLM):** deferred. It belongs with an MCP server
      interface (Phase 12); the `Provider` interface is the hook.
- [x] **Output** (`Answer`): direct answer, cited evidence (label, section, node id,
      offsets, page, text), references, confidence, flags, provider / model / fallback
      used, tokens, cost, latency
- [x] **Citation validation:** cited labels (list + inline `[E#]`) must exist in the
      pack; invalid / uncited answers are flagged, and `strict=True` abstains on them
- [x] **Abstention:** no model call when the pack is empty or Phase 7 confidence is
      insufficient (when abstention is on); a model reply marked insufficient abstains;
      refusals abstain
- [~] **Readings** (`answer_eval.py`):
  - built: answer F1 / number match / optional LLM judge, citation precision / recall,
    abstention accuracy, tokens and cost, dry-run estimates
  - `--resume <rows.jsonl>`: each finished row is saved, so quota-stopped runs continue
  - extractive baseline measured
  - **live, Gemini 3.1 Flash-Lite (free), rule-based:** answer F1 0.46–0.47, number match
    1.00, citation P / R 0.83 / 1.00, abstention 4/4 per set (see READINGS)
  - **live, v3.0 on Gemini:** answer F1 0.46–0.47, number match 1.00, citation P / R
    0.86 / 0.99, abstention 8/8, false abstention 0
  - **abstention off (model decides):** 8/8 unanswerable refused, 0 false abstentions
    (vs 5 with the Phase 7 pre-call gate)
  - Claude / OpenAI: optional, need keys (dry-run estimate ~$0.4–1.0 per system on
    Claude Opus 5.5)
- [x] Tests (`tests/test_answer.py`, 14, fake clients, no network)

*Optional follow-ups:*
- Optional paid reference: set `ANTHROPIC_API_KEY` (or `ant auth login`) and/or `OPENAI_API_KEY`, then run
  `python -m gnn_pruner.answer_eval --sets eval/contracts_v1.jsonl eval/contracts_scale_v1.jsonl --system rule --llm claude --judge claude`
  (and `--system v3.0`, `--llm openai:<model>`).
- The spec path has no retrieval-confidence signal, so it never abstains. Port the Phase 7
  confidence features to `SpecQA`.

## v4 — Chapter workers + central board (user design, `docs/ARCHITECTURE.md`)

Constraints for the whole track:
- **Hardware:** 8 GB RAM, CPU only.
- **LLMs:** every LLM is hosted (Gemini free tier for development; Claude / OpenAI when
  approved). Local LLMs froze the laptop.
- **Evidence:** summaries are for navigation only. Only word-for-word original text is
  ever evidence.

### v4-1 — Tree + enrichment 🟡 (built and measured; human fact spot check pending)
- [x] **Structure from the PDF outline** (`ingest/pdf_loader.py`):
  - entries are matched to the text, also across split lines
  - flat "Book / Chapter" outlines are re-levelled (`toc_levels`)
  - other unnumbered headings become text
  - fallback: headings / numbering, then font style
- [x] **Text cleaning:** ligatures, invisible characters, curly quotes, drop caps,
      one-em paragraph indents (header / footer removal and de-hyphenation already
      existed)
- [x] **Document tree** (`tree.py`):
  - root > (part) > chapter > section > subsection > passage, with placeholders
  - wrapper title headings unwrapped
  - schedules / annexes / exhibits / appendices lifted to chapters
  - virtual chapters (~25, at most 40 pages) for unstructured text
  - subsections > 2,000 tokens sliced
  - sentence ids and `quote(ids)` (cite by id)
- [x] **Node record:** number, title, path, pages, span, tokens, summary / key facts /
      text, sentences, defined terms, links (refers_to, uses_definition)
- [x] **Root metadata** (`metadata.py`):
  - fields:
    - books: title, author, year, publisher, ISBN (checksum)
    - contracts: parties, effective date, governing law
  - sources: labelled lines and patterns → PDF fields → LLM with evidence
  - kept only if the text is in the document
  - e-book release years rejected
- [x] **Bottom-up enrichment** (`enrich.py`):
  - typed key facts citing sentence ids, batched
  - checks: ids belong to the subsection (paragraph ids expanded, misfiled facts moved),
    numbers present in the cited sentences
  - section → chapter → part → root summaries
  - lists by fact type
  - retries, then offline fallback
  - resumable after a quota stop; `revalidate` re-checks logged drops offline
- [x] **`upload.py`:** CLI, cache by file hash, `--re-enrich`, `--refresh-metadata`
- [x] **Readings** (`upload_eval.py`, READINGS "Build step 1"):
  - outline entries found: 100% (fonts only: 35% / 77%)
  - metadata: 8/8
  - facts kept: 88% / 97% / 100%
  - ~20 calls per 100 pages, $0
- [ ] **Human spot check of 36 key facts** (`docs/fact_check_sheet.md`)
- [x] Tests (`tests/test_tree.py`, 15, fake providers, no network)
- Not covered yet: a real contract PDF with parties / dates / law; scanned PDFs (OCR);
  entity links (v4-5)

### v4-2 — Long-document benchmark 🟡
- [x] **Documents** (`longbench.py`, `data/longbench/`):
  - contracts: data rooms of real CUAD contracts, 130 / 977 / 8,619 pages
  - books: libraries of NarrativeQA test-set novels, 119 / 1,001 / 9,988 pages
  - nested S ⊂ M ⊂ L
- [x] **Questions** (`eval/long/*.jsonl`):
  - CUAD clauses; CUAD-absent clauses (unanswerable)
  - NarrativeQA human questions
  - planted facts: single, distractors, two-hop, never-planted
- [x] **Systems** (`longbench_eval.py`):
  - rule pipeline, GNN pipeline
  - plain search k=5 / k=1 (equal budget)
  - whole document (`full`)
  - chapter workers
- [x] **Metrics:**
  - evidence hit (3-gram coverage)
  - planted-value match, abstention / false abstention
  - LLM judge (a different model), token F1
  - calls, tokens, latency, build time, memory
  - `--rescore` regrades saved answers
- [x] Readings S and M (retrieval) and S (answers): READINGS "Build step 2"
- [x] L retrieval: books, GNN 100% needles / 2-hop; contracts, pipelines 39–44% needles (wrong-contract picks); rule pipeline out of memory on the 9,988-page library
- [ ] Answers at M / L (free quota: ~500 calls per model per day; or Claude)
- [ ] Whole-document baseline on a strong long-context model (Claude, budget pending)
- [x] Tests (`tests/test_longbench.py`, 5)

### v4-3 — Central board + chapter workers 🟡
- [x] **Routing** (`workers/routing.py`): splits the question into the units it names
      (distinctive title words; a named book opens its chapters) and the content to search
- [x] **Board** (`workers/board.py`):
  - findings by sentence / passage id, unknown ids rejected and logged
  - exact quotes taken from the tree; merges by passage
  - definitions added
  - follow by id, title or free text (looked up document-wide)
  - call budget; evidence pack with paths and pages
- [x] **Worker** (`workers/worker.py`):
  - first view: outline + summaries + key facts + the chapter's best BM25 sentences for
    the content words
  - actions open / read / search / done, at most 4 steps, independent calls
  - prompt guards against instructions in the document
- [x] **Orchestrator** (`workers/orchestrator.py`):
  - named chapters, or every chapter when none is named
  - parallel workers, follow hops ≤ 2, definitions
  - no evidence → "does not say" without an answer call; else the Phase 10 answer step
- [x] Gemini pacing made thread-safe (parallel workers)
- [x] Tests (`tests/test_workers.py`, 8, scripted fake model)
- [x] **Readings, contracts S, run 1:**
  - CUAD judged correct 47% (whole document 33%, plain search 33%, rule 13%)
  - ~150 evidence tokens, 4.4 calls per question
- [ ] Run 2 (free-text follow, stricter posting): 8/42 done, resumes on quota reset
- [ ] Books (needs the gate or the whole-document route: an unnamed single-book question
      opens every chapter)

### v4-4 — Worker gate + comparison ⬜
- [ ] Gate:
  - a chapter opens if its local score (BM25 + vectors + GNN) is in the top k, **or**
    the table-of-contents screen (1 LLM call) marks it, **or** the question names it
- [ ] Readings:
  - system 5 (v4) vs systems 1–4: accuracy, citations, abstention
  - LLM calls per question
  - cost, latency
  - gate recall (share of answer chapters opened)

### v4-5 — Entity links ⬜
- [ ] Mentions of the same character / party / concept linked across chapters; used by
      the gate and by `follow`
- [ ] Readings: multi-chapter book questions with and without entity links

### v4-6 — Learning loop (= Phase 11) ⬜
- [ ] Accepted board posts become gate training labels; the gate retrains periodically
- [ ] User feedback feeds the same labels
- [ ] Readings: LLM calls per question and gate recall before / after retraining

### v4-7 — Demo app (local chat interface) ⬜ Planned
**Purpose:** a demo / project presentation, run on this laptop.
- **Scope:** single user, no login, no hosting. Gradio or Streamlit; Streamlit chosen for
  the multi-panel layout.
- **When:** starts only once the core engine is proven (see "Next up"); about 1–2 days.

```
┌──────────── Sidebar ────────────┬──────────── Chat ─────────────┬──── Source viewer ────┐
│ Upload PDF / Word / Markdown    │ question → streamed answer    │ cited PDF page, quote │
│   (progress; cached)            │ with numbered citations       │ highlighted           │
│ Document library                │ "How I found this" (gate,     │ Outline (tree)        │
│ Model: Gemini free / Claude /   │   calls, tokens, cost, time)  │ Facts & lists         │
│   offline                       │ feedback buttons              │                       │
└─────────────────────────────────┴───────────────────────────────┴───────────────────────┘
```

- [ ] **Service layer** (`service.py`): one API the UI calls (a later web API can reuse it)
  - library of processed documents (cached trees)
  - save upload; process with progress (tree → metadata → key facts / summaries; option
    to skip summaries for speed)
  - open a document (sentence index + gate, kept in memory)
  - ask: follow-up rewriting from the conversation (one call, only for follow-ups) →
    gate → workers → board → cited answer
  - offline mode (no model): gate-picked chapters, best sentences as cited extracts
  - citations: label, chapter / section path, page, exact quote, source file
  - trace: chapters opened and why, rounds, model calls, tokens, cost, time
  - feedback appended to `data/feedback.jsonl` (input for the learning loop, v4-6)
- [ ] **Interface** (`app/demo.py`):
  - **sidebar:** upload with a progress bar, document library, model choice (shows
    whether keys are set, never the keys), document info (title, author / parties,
    pages, chapters)
  - **chat:** history per document, streamed answers, "The document does not say"
    shown clearly
  - **citations:** clickable chips; the source viewer shows the PDF page with the quote
    highlighted (PyMuPDF) or the passage with the quote marked (other formats)
  - **"How I found this":** e.g. "opened 2 of 554 chapters: named in the question /
    best-matching sentence / table of contents"
  - **document tabs:** outline tree, key facts, lists (obligations, deadlines, amounts …)
  - **feedback:** 👍 / 👎 / "missed a clause" (with a note)
- [ ] **One-command start:** `./demo.sh` (keys read from the environment)
- [ ] **Tests:** service layer with a fake model (ask, offline mode, follow-up rewrite,
      citations, page highlight, feedback)
- [ ] **Walkthrough before the presentation:**
  - a contract and a book
  - a named-unit question, a cross-reference (two-hop), an unanswerable question, a
    follow-up
- **Known limit:** on the free Gemini tier an answer takes ~10–60 s (pacing) and the quota
  is ~500 calls per day. Claude would be faster.

**Later, only if it goes beyond a demo** (not planned now):
- FastAPI backend and background job queue
- React / Next.js frontend
- login and per-user isolation, database
- hosting on a 16–32 GB server
- monitoring
- CI on GitHub Actions

## Phase 11 — Training pipeline: curriculum + feedback (spec §5 training, §9 phase 3)

- [ ] One training command:
  1. Silver Path generation
  2. supervised pre-training
  3. PPO
  4. attention-head training
  5. joint fine-tune
- [ ] Curriculum: short → long documents, silver → feedback labels
- [ ] Feedback capture (thumbs, "missed the clause", corrected answer, correct section,
      highlights), linked to query, graph and answer. Converted to labels for retraining.
- [ ] Regression gate (spec §9): fixed 100-query adversarial suite, with ≤ 2% drop vs the
      previous model version
- [ ] LoRA/PEFT: only if an LLM is fine-tuned (spec §6)

## Phase 12 — Production hardening (spec §6, §9 phase 4)

- [ ] FastAPI service: ingest, query and feedback endpoints
- [ ] Explainability dashboard (Gradio/Streamlit): coarse/fine scores, attention boosts,
      shards, pruning reasons and paths
- [ ] Ray performance tuning; documentation
- [ ] Real-user question set (Phase 0 open item)
- [ ] Optional: Neo4j/Qdrant/Postgres backends behind the existing store interfaces
