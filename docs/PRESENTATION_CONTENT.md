# GNN PRUNER — B.Tech project presentation content

Material for the slides of the **GNN PRUNER** project (frozen at tag `gnn-pruner-final`).
Part A is a slide-by-slide outline. Part B holds the full data behind it. Part C lists
references, Part D a glossary, and Part E likely viva questions with answers.

All numbers come from `docs/READINGS.md` (measured on one MacBook, 8 GB RAM, CPU only).
**Check every reference against the original paper before the final slides.**

---

# Part A — Slide-by-slide outline (~22 slides)

### Slide 1 — Title
**GNN PRUNER: Structure-Aware Pruning of Very Long Documents for Grounded Question
Answering**
- Name, roll number, guide, department, institute, date
- One line: *"Answer questions over 10,000-page documents by sending the AI model only
  ~300 tokens of checked evidence."*

### Slide 2 — The problem
- Contracts, reports and books run to hundreds or thousands of pages.
- **Context limits:** large language models (LLMs) have a limited context window. A
  10,000-page document is ~5 million tokens, far beyond any model.
- **Even when it fits:**
  - cost grows with every token sent
  - models miss facts placed in the middle of long inputs ("Lost in the Middle", Liu et
    al., TACL 2024)
- **Answers need sources:** in legal work an answer without its source clause cannot be
  trusted.

### Slide 3 — Why plain retrieval (RAG) is not enough
- **Plain RAG** cuts the document into fixed chunks and sends the top-k most similar ones.
- **It ignores document structure:**
  - sections and chapters
  - cross-references ("subject to Section 4.2")
  - defined terms
- **Typical failures:**
  - **Missing dependencies:** the clause is retrieved, its definition or exception is not
  - **Cross-chapter answers:** the second half of the answer is in another chapter
  - **Bloated or incoherent context:** too many loosely related chunks

### Slide 4 — Research gap
| Existing approach | What it does | Gap we address |
|---|---|---|
| Long-context LLMs | read the whole input | do not fit 10,000 pages; cost grows linearly; mid-document recall drops |
| Plain chunk RAG (Lewis et al., 2020) | top-k similar chunks | ignores structure, cross-references and definitions |
| Tree / summary methods (RAPTOR, MemWalker, PageIndex) | navigate summaries | build summaries with an LLM; not a learned structural pruner; few guarantees on evidence size |
| Graph RAG (GraphRAG, HippoRAG) | entity graphs over a corpus | LLM-heavy indexing; does not use the document's own section / reference graph |
| **This project** | **a learned GNN pruner over the document's native structure graph** | constant-size evidence pack (~300–400 tokens) whatever the document size; dependency-aware (definitions, references, exceptions); verifiable citations; tested at 100 → 10,000 pages on one laptop |

### Slide 5 — Objectives
1. **Document graph:** convert documents (PDF / DOCX / HTML / Markdown) into a 4-layer
   graph (DAG): sections → sub-sections → paragraphs → sentences, with reference and
   definition edges.
2. **Learned pruning:** score and prune the graph with graph neural networks (GraphSAGE),
   trained without hand labels (Silver Paths + PPO).
3. **Scale:** add graph coarsening, cross-chapter attention, and distributed execution
   (Ray + METIS) for 10,000-page documents.
4. **Grounded answers:** build a small, dependency-complete, cited evidence pack; generate
   grounded answers with an LLM and check every citation.
5. **Measurement:** evaluate against the architecture specification's targets with our own
   measured readings.

### Slide 6 — System architecture (diagram)
```
Document (PDF/DOCX/HTML/MD)
  → Parse (PyMuPDF / Docling)
  → 4-layer DAG + edges (hierarchy, sequence, cross-reference, definition)  [Parquet shards]
  → Query planner (7 strategies)
  → Hybrid retrieval (BM25 + phrase + dense + structural, weighted RRF)
  → Stage 1: CoarseGNN on paragraph super-nodes (coarsening)    ┐ distributed
  → Cross-edge attention (local window 512 + global summary)    ┘ over Ray workers
  → Stage 2: FineGNN on sentences of kept paragraphs
  → Merge (p > 0.8 × mean) + dependency rules (definitions, references, qualifiers)
  → Topology-routed prompt (QA / PATH / SIBLING / SYNTHESIS) with a token budget
  → LLM answer with checked citations, or "The document does not say"
```

### Slide 7 — Step 1: Document graph (4-layer DAG)
| Layer | Content | Count in the training corpus (48 contracts, 985 pages) |
|---|---|---|
| L1 | sections / chapters | 1,709 |
| L2 | sub-sections | 8,739 |
| L3 | paragraphs / clauses / tables | 17,469 |
| L4 | sentences / sub-clauses | 31,267 (8,260 virtual) |
| **Total, one 1,000-page document** | | **51,304 nodes** |

- **Edges:** hierarchical, sequential, reference (cross-references, definitions).
- **Scale:** parse to graph in 1.2 s at 1,000 pages, 12.6 s at 10,000 pages (510,486
  nodes). Always a valid DAG.
- **Storage:** Parquet shards, memory-mapped; shard load 1.5–14 ms.

### Slide 8 — Step 2: Training without hand labels (Silver Paths)
- **Silver Paths:** BM25 + dense retrieval pick likely answer paths automatically.
  - **Coverage-weighted fusion** (our improvement) raised paraphrase recall from 0.54 to
    0.96.
- **Label quality:** leaf recall 0.94 (train) / 0.96 (validation), against known answers.
- **Corpus:** 48 generated contracts, 985 pages, 1,340 known-answer questions.

### Slide 9 — Step 3: CoarseGNN (GraphSAGE)
- **Model:** typed GraphSAGE (one convolution per edge type), 2 layers, 128 hidden, skip
  connection.
- **Inputs:** paragraph embedding, question interaction, BM25 and structure features.
- **Training:** a listwise loss (τ = 0.1), then PPO for the keep / prune policy.
  - supervised: 45 s; PPO: 85 s (CPU)
  - inference: 2–3.5 ms per question
- **Result:** best ranker in domain (MRR 0.794 vs 0.776 rule-based, 0.758 dense, 0.633
  BM25).
- **PPO:** keeps **2% of paragraphs while keeping 90% of answers** on long documents
  (spec target < 5%).

### Slide 10 — Step 4: Graph coarsening + FineGNN
- **How it works:** stage 1 ranks paragraph super-nodes; stage 2 ranks sentences only
  inside kept paragraphs.
- **Effect:** **~90% fewer nodes processed** (63 vs 620 on long test documents) at about
  −1 point MRR.
- **Evidence pack** = 2.8% of the document's words on long documents.
- **Scaling (CPU)**, with a 200-paragraph cap:

  | Pages | Latency |
  |---|---|
  | 100 | 18 ms |
  | 1,000 | 72 ms |
  | 10,000 | 1.34 s |

  The no-coarsening baseline could not run at 10,000 pages on 8 GB.

### Slide 11 — Step 5: Cross-edge attention (cross-chapter answers)
- **Mechanism:** local window (512 nodes) + global summary (top-100 nodes).
- **In domain:** cross-chapter complete@10 **0.85 → 0.92**, stage-1 0.95 → 1.00.
- **On real test contracts:** within noise. The rule pipeline's explicit reference
  following still wins there (honest finding).

### Slide 12 — Step 6: Distributed execution (Ray + METIS)
- **Design:** tree-cut and METIS partitioning; boundary (halo) replication; aggregator
  protocol.
- **Correctness:** distributed scores equal single-process scores (difference ≤ 7×10⁻⁶)
  with exact halos.
- **Finding:** the spec's 3-node halo is approximate (top-50 agreement 0.92–0.96) but kept
  every answer.
- **On one 8-core laptop:** speed-up only 1.3×. The gain is memory per worker; real
  speed-ups need several machines.

### Slide 13 — Step 7: Evidence pack, prompt routing, LLM answer
- **Routes:**
  - SYNTHESIS: > 2 clause citations
  - SIBLING: > 3 items in one section
  - PATH: depth > 3
  - QA: otherwise
- **Token budgets:**

  | Route | Tokens |
  |---|---|
  | QA | 250 |
  | SYNTHESIS | 250 |
  | SIBLING | 300 |
  | PATH | 300 |

- **Answer:** the LLM answers only from the pack in JSON (answer, cited labels,
  confidence, insufficient). Every citation label is checked.
- **Providers:** Gemini (free tier, used for readings), Claude, OpenAI, an offline
  extractive mode.

### Slide 14 — Results 1: retrieval quality (paragraph ranking, MRR / hit@1)
| Set | BM25 | Dense | Rule pipeline | GNN |
|---|---|---|---|---|
| Generated validation (280 q) | 0.633 / 0.56 | 0.758 / 0.65 | 0.776 / 0.70 | **0.794 / 0.71** |
| Dev multi-passage | 0.636 / 0.45 | 0.849 / 0.73 | 0.864 / 0.73 | **0.955 / 0.91** |
| Real test contracts | 0.795 / 0.72 | 0.859 / 0.75 | **0.965 / 0.94** | 0.935 / 0.89 |
| Real long test contracts | 0.725 / 0.61 | 0.786 / 0.69 | **0.918 / 0.86** | 0.852 / 0.78 |

The GNN is best in its training domain and second on real contracts (domain shift:
trained on generated contracts).

### Slide 15 — Results 2: speed and token reduction vs the spec
| | Spec reference | Measured (laptop CPU) |
|---|---|---|
| Latency at 10,000 pages | 80 ms (4× A100 cluster) | **4.7 s** (two-stage); rule pipeline 167 s → **~35× faster** |
| Tokens sent to LLM, 10,000 pages | 5% of document | **~377 tokens = 0.007%** (~13,500× fewer than full context) |
| Recall@10 (real test, small) | 95% | 0.958 (GNN) / 1.000 (rule) |
| Structural coherence | 91% | 0.98–0.99 |
| Cross-chapter recall | 94% | 0.925 (generated); 0.71 (real test, GNN) / 1.0 (rule) |
| Adversarial robustness | 84% | 0.83–0.88 hit@1 kept under perturbation |
| Nodes kept | < 5% | 2–3% (PPO / merge) |

### Slide 16 — Results 3: grounded answers with an LLM (Gemini 3.1 Flash-Lite, free, 80 test questions)
| | No LLM (extractive) | **With LLM** |
|---|---|---|
| Numbers in the answer correct | 55–60% | **100%** |
| Required clauses cited | 71–79% | **100%** |
| Answer F1 vs reference | 0.35–0.38 | 0.46–0.47 |
| Unanswerable questions refused | 1–4 of 4 | **8 / 8** |

- **Finding:** the model's own "insufficient evidence" judgement beat our score-based
  pre-check: 0 wrong refusals vs 5.
- **Cost:** $0 on the free tier.

### Slide 17 — Results 4: human evaluation
- **Setup:** blind A/B of evidence packs (GNN vs rule pipeline), 20 questions.
- **Result:** 50% preference (6 / 6 / 8 ties). The preferred pack was sufficient in 20 / 20.
- **Insight:** raters preferred the **shorter** sufficient pack in 11 of 12 decided pairs,
  so the token budget is the lever.

### Slide 18 — Extension: real books and documents (upload pipeline)
- **Real PDFs:** two public-domain books, 144 and 215 pages.

  | | Chapters found (vs the PDF outline) |
  |---|---|
  | Fonts only | 35% / 77% |
  | **With the PDF outline (our method)** | **100% / 100%** |

- **Metadata** (title, author; year / ISBN correctly left empty): **8 / 8**.
- **Key facts** citing sentence ids, checked automatically: 88% / 97% / 100% kept.
- **Cost:** ~20 model calls per 100 pages ($0 on the free tier).

### Slide 19 — Extension: long-document benchmark (100 → 10,000 pages)
- **Real contracts (CUAD):** 510 contracts, nested data rooms of 130 / 977 / 8,619 pages.
- **Real novels (NarrativeQA):** libraries of 119 / 1,001 / 9,988 pages.
- **Planted facts at equal evidence budget (~200–300 tokens):**

  | | Plain search | **Our pipeline** |
  |---|---|---|
  | Books, single facts | 39–67% | **100%** |
  | Books, two-hop facts | 0% | **100%** |
  | Contracts, single facts (≤ 1,000 pages) | 17% | 67–83% |

- **At ~10,000 pages (books):** the GNN pipeline finds **100%** of single and two-hop
  facts from ~285 tokens.
- **Chapter workers + relevance gate** (extension): on real contract clauses, 47% correct
  vs 33% when a model reads the whole 130-page document, sending ~150 tokens instead of
  ~87,000. The gate opens ~1 of 554 contracts with 95% accuracy.

### Slide 20 — Limitations (honest)
- **The GNN is trained on generated contracts:** weaker on real contracts (domain shift).
- **Distribution does not speed up one laptop** (needs a cluster); spec latencies assume
  GPUs.
- **Two-hop facts** fail for the worker extension; **story-level book questions** are weak
  for all systems.
- **Memory:** the rule pipeline cannot index a 9,988-page library on 8 GB.
- **Evaluation limits:** small sample sizes (15–42 questions per setting); free small
  models; a strong long-context baseline is not yet run.
- **No OCR** for scanned PDFs; English only.

### Slide 21 — Future work
- **"Complete or Silent"** (next project, planned): certified answers.
  - **Closure certificates:** every exception, override and amendment that could change
    the answer was read.
  - **Absence certificates:** self-calibration with planted "canary" facts gives a
    measured miss risk for "not in the document".
- **Learning loop:** retrain the GNN on real labels produced by the system.
- **Demo app:** chat interface with a citation viewer (the PDF page with the quote
  highlighted).

### Slide 22 — Conclusion
- **Built end to end:** parsing → 4-layer graph → GNN pruning (coarsening, attention,
  distribution) → cited LLM answers.
- **Measured honestly** against the specification:
  - **constant ~300–400-token evidence packs at any size**
  - ~35× faster than the rule pipeline at 10,000 pages
  - 100% correct numbers and citations with an LLM
- **Clear findings:** structure helps; learned models need real-domain labels; reference
  following matters for cross-chapter answers.

---

# Part B — Full data

## B1. Project at a glance
| Item | Value |
|---|---|
| Language / stack | Python 3.11 (uv), PyTorch, PyTorch Geometric, Ray, pymetis, sentence-transformers (MiniLM), FAISS, scikit-learn, PyMuPDF, Docling, pyarrow (Parquet), SQLite, pytest |
| Source code | ~14,100 lines (53 modules) |
| Tests | ~218 test functions (234 passing in the last full run, 1 skipped); no network needed |
| LLM providers | Gemini (free tier; used for readings), Claude, OpenAI, Ollama (local; dropped: froze the 8 GB laptop), extractive |
| Hardware | MacBook, 8 GB RAM, 8 CPU cores, no GPU |
| Phases | Foundation 0–8 → spec build G1–G6 → 9 (prompt) → 10 (LLM) → extension steps 1–4 |
| Repository | github.com/Usermayalive/GNN-PRUNER (tag `gnn-pruner-final`) |

## B2. Phases and what each delivered
| Phase | Delivered |
|---|---|
| 0 | Use case: commercial contracts; gold set `contracts_v1` (40 questions: 36 answerable, 4 unanswerable); question taxonomy (definition, fact, cross-reference, comparison, summary, contradiction, unanswerable) |
| 1 | Loaders: PDF (PyMuPDF, OCR fallback), DOCX, HTML, Markdown; same results across formats |
| 2 | Document graph: chapter / section / paragraph / clause / sentence nodes; hierarchical, sequential, cross-reference, definition edges; JSON + SQLite store |
| 3 | Embeddings (MiniLM, mpnet), FAISS / numpy index; dense recall 1.0 vs 0.79 TF-IDF |
| 4 | Query planner, 7 strategies; dev 1.0, held-out 0.739 |
| 5 | Hybrid retrieval: BM25, phrase, dense, structural, plan, graph sources; weighted reciprocal-rank fusion |
| 6 | Best-first bounded graph expansion |
| 7 | 12-feature linear scorer; calibrated evidence-confidence signal (abstention) |
| 8 | Adaptive pruning (floor + gap) + dependency rules (definitions, references, qualifiers, numeric conflicts) |
| G1 | Generated corpus (48 contracts, 985 pages, 1,340 questions); 4-layer Parquet export; Silver Paths; Docling option |
| G2 | CoarseGNN (typed GraphSAGE) supervised + PPO |
| G3 | Graph coarsening + FineGNN (two-stage) |
| G4 | Cross-edge attention (local window + global summary) |
| G5 | Distribution: tree-cut / METIS partitioning, halo replication, Ray actors, aggregator |
| G6 | Spec tables 8–10 measured; adversarial suite; human preference study |
| 9 | Topology-routed prompts (QA / PATH / SIBLING / SYNTHESIS), token budgets, citation map |
| 10 | LLM answers with checked citations and abstention; answer-quality readings |
| Ext. 1 | Upload: PDF outline structure, document tree, metadata, key facts and summaries (LLM) |
| Ext. 2 | Long-document benchmark (CUAD + NarrativeQA, 100 → 10,000 pages, planted facts) |
| Ext. 3–4 | Chapter workers + central board (sentence-id citations) + relevance gate |

## B3. Datasets
| Dataset | Source | Size | Used for |
|---|---|---|---|
| `contracts_v1` | 2 hand-written contracts (SaaS agreement ~1 page, MSA ~½ page) | 40 questions (36 answerable, 4 unanswerable) | **Test set** (never tuned on) |
| `contracts_scale_v1` | the same contracts expanded to ~10 pages each | 40 questions | **Test set**, long documents |
| `calibration_v1` | real contracts | 36 | dev: weights and calibration |
| `dev_scale_v1` | real, long | 20 | dev |
| `dev_pruning_v1` / `_scale_v1` | real, multi-passage answers | 11 / 11 | dev: pruning thresholds |
| `dev_crosschapter_v1` / `_scale_v1` | real, answers in 2 chapters | 10 / 10 | dev: cross-chapter (G4) |
| `planner_v1` / `planner_holdout_v1` | question-type labels | 46 / 23 | planner dev / held-out |
| `corpus_v1` | **generated** contracts (templated) | 48 docs, 985 pages, 503k words, 1,340 questions (1,060 train / 280 val) | GNN training (Silver Paths) |
| Scale documents | generated | 100 / 1,000 / 10,000 pages, 12 questions each | speed and scale (G3 / G5 / G6) |
| `adversarial_v1` | 20 generated validation questions × 5 perturbations | 100 | robustness |
| Human preference | blind A/B of evidence packs | 20 pairs | human evaluation |
| Real books (PDF) | Project Gutenberg: *Adventures of Sherlock Holmes* (144 pp), *A Tale of Two Cities* (215 pp) | 2 books | upload / structure readings |
| **CUAD v1** | The Atticus Project (CC BY 4.0) | 510 real contracts, ~8,900 pages, 41 lawyer-labelled clause types | long-document benchmark (data rooms 130 / 977 / 8,619 pages; 42 / 109 / 176 questions) |
| **NarrativeQA** (test set) | DeepMind (Apache 2.0) + Gutenberg texts | 52 novels used; human questions with 2 answers | long-document benchmark (libraries 119 / 1,001 / 9,988 pages; 42 / 94 / 200 questions) |
| Planted facts ("needles") | generated inside the benchmark | single, two-hop, distractor, never-planted | exact evidence for citation scoring |

## B4. Key result tables (copy from `docs/READINGS.md`)
- **G2 paragraph ranking:** Slide 14.
- **G2 ablations (MRR, corpus val / contracts_v1 / contracts_scale_v1):**

  | Variant | MRR |
  |---|---|
  | full (typed + skip) | 0.794 / 0.935 / 0.852 |
  | no edges | 0.780 / 0.919 / 0.835 |
  | untyped GraphSAGE | 0.722 / 0.898 / 0.833 |
  | **no query input** | **0.091 / 0.213 / 0.111** (the question input is essential) |

- **G3 scale (two-stage, coarse + fine latency):**

  | Pages | Spec rule | With a 200-paragraph cap |
  |---|---|---|
  | 100 | 12 ms | 18 ms |
  | 1,000 | 198 ms | 72 ms |
  | 10,000 | 4.07 s | 1.34 s |

- **G4 cross-chapter complete@10 (corpus val):** 0.85 → **0.917**.
  - Ablations: local only 0.900, global only 0.875, fine stage only 0.850.
- **G5 parity:** max logit difference ≤ 7×10⁻⁶ (exact halo); spec halo 3.6–4.7, top-50
  agreement 0.92–0.96; best speed-up 1.3× (coarse stage 7.6 → 5.8 s at 10,000 pages).
- **G6 latency p50:**

  | Pages | G3 | v3.0 | Rule pipeline |
  |---|---|---|---|
  | 100 | 23 ms | 53 ms | 60 ms |
  | 1,000 | 165 ms | 463 ms | 1,517 ms |
  | 10,000 | 4.7 s | 9.9 s | 167 s |

  Questions per hour at 10,000 pages: G3 ~760 vs rule ~22.
- **G6 tokens:** packs 300–400 tokens at every size (0.72% at 100 pages, 0.08% at 1,000,
  0.007% at 10,000).
- **G6 adversarial hit@1 kept** (v3.0 / rule):

  | Perturbation | v3.0 | Rule |
  |---|---|---|
  | distractor | 0.90 | 0.95 |
  | wrong number | 0.80 | 1.00 |
  | party swap | 0.75 | 0.80 |
  | typo | 0.70 | 0.75 |
  | paraphrase | 0.80 | 0.80 |

- **Phase 10 (Gemini):**

  | System | Answer F1 | Numbers | Citation P / R | Unanswerable refused | False refusals |
  |---|---|---|---|---|---|
  | rule | 0.46–0.47 | 1.00 | 0.83 / 1.00 | 8/8 | 5–8% (pre-check on) / 0% (off) |
  | v3.0 | 0.46–0.47 | 1.00 | 0.86 / 0.99 | 8/8 | 0% |

- **Extension, long-document benchmark (evidence found at ~200–300 tokens):**

  | Size | Plain search k=1 | Rule pipeline | GNN pipeline |
  |---|---|---|---|
  | contracts 130 pp (planted facts) | 17% | 67% | 83% |
  | contracts 977 pp (planted / two-hop) | 17% / 0% | 83% / 25% | 83% / 100% |
  | contracts 8,619 pp (planted / two-hop) | 11% / 0% | 39% / 67% | 44% / 100% |
  | books 119 / 1,001 pp (planted) | 67% / 42% | 100% / 100% | 100% / 100% |
  | books 9,988 pp (planted / two-hop) | 39% / 0% | out of memory | 100% / 100% |

## B5. Design decisions worth mentioning (where the spec was incomplete)
- **D4:** the GNN gets the question as input. Without it MRR collapses to 0.09–0.21.
- **D8:** virtual sentence nodes, so every paragraph has sentence children (uniform depth).
- **D12:** prompt-route precedence SYNTHESIS > SIBLING > PATH > QA.
- **Coverage-weighted fusion** for Silver Paths (paraphrase recall 0.54 → 0.96).
- **Indexed conflict detection:** identical results, ~37× faster at 10,000 pages.

---

# Part C — References

**Methods used in the project:**
1. W. L. Hamilton, R. Ying, J. Leskovec. *Inductive Representation Learning on Large
   Graphs* (GraphSAGE). NeurIPS 2017.
2. G. Karypis, V. Kumar. *A Fast and High Quality Multilevel Scheme for Partitioning
   Irregular Graphs* (METIS). SIAM Journal on Scientific Computing, 1998.
3. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. *Proximal Policy
   Optimization Algorithms* (PPO). arXiv:1707.06347, 2017.
4. P. Moritz et al. *Ray: A Distributed Framework for Emerging AI Applications.* OSDI 2018.
5. S. Robertson, H. Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond.*
   Foundations and Trends in Information Retrieval, 2009.
6. G. V. Cormack, C. L. A. Clarke, S. Büttcher. *Reciprocal Rank Fusion Outperforms
   Condorcet and Individual Rank Learning Methods.* SIGIR 2009.
7. N. Reimers, I. Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese
   BERT-Networks.* EMNLP-IJCNLP 2019.
8. W. Wang et al. *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression
   of Pre-Trained Transformers.* NeurIPS 2020. (Model used: `all-MiniLM-L6-v2`.)
9. J. Johnson, M. Douze, H. Jégou. *Billion-Scale Similarity Search with GPUs* (FAISS).
   IEEE Transactions on Big Data, 2021.
10. Z. Cao, T. Qin, T.-Y. Liu, M.-F. Tsai, H. Li. *Learning to Rank: From Pairwise Approach
    to Listwise Approach* (ListNet; listwise loss). ICML 2007.
11. A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
12. I. Beltagy, M. E. Peters, A. Cohan. *Longformer: The Long-Document Transformer*
    (sliding-window + global attention). arXiv:2004.05150, 2020.
13. M. Fey, J. E. Lenssen. *Fast Graph Representation Learning with PyTorch Geometric.*
    ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds.
14. C. Auer et al. *Docling Technical Report.* arXiv 2024.
15. P. Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
    NeurIPS 2020.
16. N. F. Liu et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024.

**Datasets:**

17. D. Hendrycks, C. Burns, A. Chen, S. Ball. *CUAD: An Expert-Annotated NLP Dataset for
    Legal Contract Review.* NeurIPS 2021 Datasets and Benchmarks Track.
18. T. Kočiský et al. *The NarrativeQA Reading Comprehension Challenge.* TACL 2018.
19. Project Gutenberg (public-domain books), gutenberg.org.

**Related work (positioning):**

20. P. Sarthi et al. *RAPTOR: Recursive Abstractive Processing for Tree-Organized
    Retrieval.* ICLR 2024.
21. H. Chen et al. *Walking Down the Memory Maze: Beyond Context Limit through Interactive
    Reading* (MemWalker). arXiv 2023.
22. PageIndex (VectifyAI): reasoning-based, vectorless retrieval over a document tree;
    open-source project, 2025.
23. D. Edge et al. *From Local to Global: A Graph RAG Approach to Query-Focused
    Summarization* (GraphRAG). arXiv 2024.
24. B. J. Gutiérrez et al. *HippoRAG: Neurobiologically Inspired Long-Term Memory for
    Large Language Models.* NeurIPS 2024.
25. Y. Zhang et al. *Chain of Agents: Large Language Models Collaborating on Long-Context
    Tasks.* NeurIPS 2024.
26. C.-P. Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language
    Models?* COLM 2024.
27. G. Kamradt. *Needle in a Haystack: pressure testing LLMs* (GitHub), 2023.

---

# Part D — Glossary (for the audience)
| Term | Meaning |
|---|---|
| **LLM** | large language model (e.g. Gemini, Claude) |
| **Context window** | the most text a model can read at once |
| **RAG** | retrieval-augmented generation: find relevant text, then let the model answer from it |
| **GNN** | graph neural network: learns from nodes and their connections |
| **GraphSAGE** | a GNN that builds each node's representation from its neighbours |
| **DAG** | directed acyclic graph (here: document → sections → paragraphs → sentences) |
| **Coarsening** | first rank whole paragraphs, then only the sentences inside the best ones |
| **PPO** | a reinforcement-learning method; here it learns which parts to keep or drop |
| **Silver Paths** | automatic (not hand-made) training labels from BM25 + dense search |
| **BM25** | classic keyword-ranking formula |
| **Dense retrieval / embeddings** | ranking by meaning using vector representations (MiniLM) |
| **RRF** | reciprocal-rank fusion: combining several rankings into one |
| **METIS** | a graph-partitioning algorithm (splitting the graph across workers) |
| **Ray** | a framework for running work in parallel across processes or machines |
| **MRR / hit@1 / Recall@10** | ranking metrics: position of the first correct item / correct item ranked first / correct item in the top 10 |
| **Evidence pack** | the small set of cited passages sent to the model |
| **Abstention** | answering "The document does not say" instead of guessing |

---

# Part E — Likely viva questions, with short answers

1. **Why a GNN and not plain RAG?**
   Documents have structure: sections, cross-references, definitions. A GNN scores a
   passage using its neighbours, so a clause linked to the question's definition or
   reference ranks higher. Measured: best ranking in domain (MRR 0.794 vs 0.758 dense,
   0.633 BM25).
2. **Why does the GNN lose to your rule pipeline on real contracts?**
   Domain shift. It was trained on generated contracts; a model trained on gold labels does
   no better on the real test set (0.908), so it is the data, not the label noise. Training
   on real labels is future work.
3. **How do you train without hand labels?**
   Silver Paths: BM25 + dense retrieval pick likely answer paths, weighted by keyword
   coverage. Leaf recall 0.94–0.96 against known answers.
4. **What does coarsening buy?**
   About 90% fewer nodes in the fine stage for about −1 point MRR. It is what makes 10,000
   pages run on 8 GB at all.
5. **Did distribution make it faster?**
   Only 1.3× on one laptop, because a single process already uses all cores. It is correct
   (exact parity); the real gain needs several machines. Reported honestly.
6. **How do you stop the model inventing answers?**
   It answers only from the pack, every citation label is checked, and it may say "The
   document does not say". With Gemini: 100% correct numbers and citations; all
   unanswerable questions refused.
7. **Why are your latencies far above the spec's 80 ms?**
   The spec assumes 4 A100 GPUs and a cluster; we run on a laptop CPU. Even so, 10,000
   pages take 4.7 s per question, ~35× faster than our rule pipeline.
8. **What is novel?**
   A learned pruner over the document's own structure graph, with a constant ~300–400-token
   cited evidence pack at any size, measured from 100 to 10,000 pages on commodity
   hardware, plus a nested long-document benchmark built from real contracts and books.
9. **Biggest limitation?**
   Domain shift of the learned model; small evaluation samples; no strong long-context
   baseline yet. All listed on the limitations slide.
10. **What next?**
    "Complete or Silent": certificates that nothing which could change the answer was left
    unread, and measured risk for "not in the document".
