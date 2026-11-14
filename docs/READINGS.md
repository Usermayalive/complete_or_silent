# Readings log

Measured values for the v3.0 build, one section per phase. Spec values are shown as
references, not targets. Unless stated otherwise:
- hardware: one Mac, CPU only
- dense embedder: `all-MiniLM-L6-v2`
- page: ~500 words (decision D3)

---

## G1 — Corpus, four-layer graph, Silver Paths (2026-10-17)

### Training corpus (`data/corpus_v1`)

Built with `python -m gnn_pruner.corpus --out data/corpus_v1 --docs 48 --embedder st:all-MiniLM-L6-v2`.

| Reading | Value |
|---|---|
| Documents | 48 generated contracts, split by document: 38 train / 10 val |
| Length | 2–120 pages each (log-uniform), 985 pages, 503k words |
| Four-layer nodes | L1 1,709 · L2 8,739 · L3 17,469 · L4 31,267 (8,260 of them virtual, D8) |
| Questions | 1,340 (1,060 train / 280 val). Every question has a known answer sentence. Types: lexical, paraphrase (~30%), definition, cross-chapter. |
| Build time | 58 s (generate, parse, layer, embed, write Parquet) |
| Size on disk | 93 MB (Parquet with embeddings) |
| Leakage | No generated sentence appears in any `examples/` document (including the scale set). No generated question equals an eval question (`tests/test_g1.py`). |

### Scaling of the document pipeline (generate → parse → four layers)

| Document | Words | Nodes (L1–L4) | Generate | Parse to graph | Four layers | Valid DAG |
|---|---|---|---|---|---|---|
| 1,000 pages | 0.50M | 51,304 | 0.1 s | 1.2 s | 0.2 s | yes |
| 10,000 pages | 5.0M | 510,486 | 1.2 s | 12.6 s | 2.3 s | yes |

The spec's Table 8 lists ~500,000 nodes for 10,000 pages; we measure 510k at 500 words
per page.

### Parquet shard loading (memory-mapped)

| Shard | Nodes | Graph load | Embeddings load | Spec reference |
|---|---|---|---|---|
| 2 pages | 138 | 1.5 ms | 0.9 ms | |
| 21 pages | 1,354 | 3.7 ms | 3.2 ms | |
| 109 pages | 5,904 | 13.7 ms | 11.7 ms | < 10 ms shard loading |

Most of the graph-load time is building Python objects from the table. The G5 shards
are sub-document partitions, so they will be smaller.

### Silver Path label quality

Labels are compared with the known answers:
- `leaf_recall`: share of answer snippets inside one of the 10 silver leaves
- `path_recall`: the answer's whole L1→L3 path is in the Silver Positive Set
- `MRR`: of the first answer leaf in the fused ranking

Generated corpus, **hybrid** mode (the mode written to `silver_{split}.jsonl`):

| Split | Group | n | leaf recall | path recall | MRR | hit@1 |
|---|---|---|---|---|---|---|
| train | all | 1,060 | 0.939 | 0.947 | 0.748 | 0.626 |
| train | lexical | 506 | 1.000 | 1.000 | 0.914 | 0.828 |
| train | paraphrase | 310 | 0.877 | 0.900 | 0.482 | 0.294 |
| train | definition | 96 | 1.000 | 1.000 | 0.957 | 0.927 |
| train | cross-chapter | 148 | 0.818 | 0.831 | 0.597 | 0.439 |
| val | all | 280 | 0.964 | 0.966 | 0.778 | 0.650 |
| val | paraphrase | 78 | 0.910 | 0.910 | 0.535 | 0.321 |
| val | cross-chapter | 40 | 0.925 | 0.938 | 0.625 | 0.425 |

Modes compared (train, all questions): BM25 only 0.673 leaf recall (paraphrases 0.048);
fallback 0.939; hybrid 0.939, with hybrid's MRR slightly higher.

Dev sets (real contracts; labels only measured, never written or trained on):

| Dev set | BM25 path recall | Hybrid path recall | Hybrid MRR |
|---|---|---|---|
| calibration_v1 (20) | 0.90 | **1.00** | 0.942 |
| dev_scale_v1 (20, long documents) | 0.90 | **1.00** | 0.942 |
| dev_pruning_v1 (11 multi-passage) | 0.909 | **1.00** | 0.803 |
| dev_pruning_scale_v1 (11 multi-passage, long) | 0.682 | **0.909** | 0.613 |

What changed the readings:
- **Coverage-weighted fusion.** With equal-weight RRF, BM25's near-random order on
  paraphrases pushed the dense hits out of the top 10. Weighting BM25 by its term
  coverage `c` of the question (and dense by `1 - c`) raised paraphrase leaf recall on
  the trial corpus from 0.54 to 0.96 (val 0.27 → 0.91).
- **Embedding choice.** On paraphrases, dense retrieval alone gets 0.90 with MiniLM and
  0.88 with mpnet. Prefixing headings to the leaf text adds about 0.02. MiniLM stays.
- **Cross-chapter questions are the weakest group.** Their second answer sits in another
  chapter, and the spec's Silver Path walks up the hierarchy only, not across
  reference edges. This is what G4's cross-edge attention addresses.
- **Precision is low by construction.** There are 10 silver leaves for 1–2 answer
  sentences, so precision is ~0.13 on the corpus and 0.15–0.25 on dev. The graded
  labels tell the GNN which silver leaves are strongest.

### Docling parser (D7) vs native loaders

`contracts_v1`, MiniLM, end to end:

| Format | Native recall / hit@1 / pack | Docling recall / hit@1 / pack |
|---|---|---|
| md | 1.0 / 0.944 / 4.92 | 1.0 / 0.944 / 4.92 |
| pdf | 1.0 / 0.944 / 4.92 | 1.0 / 0.944 / 4.92 |
| docx | 1.0 / 0.944 / 4.92 | 1.0 / 0.944 / 4.92 |
| html | 1.0 / 0.944 / 4.92 | 1.0 / 0.944 / 4.92 |

Docling gives identical end-to-end results in all four formats. Its first PDF run
downloads the layout models (`docling-layout-heron`), which takes a while on a slow
network; after that they are cached.

---

## G2 — CoarseGNN: supervised + PPO (2026-10-17)

### Model and training

- **Model:** typed GraphSAGE (one SAGEConv per edge type), 2 layers, 128 hidden units,
  skip connection. Input: paragraph embedding, embedding × query, cosine, BM25,
  coverage, structure (D4).
- **Training data:** 1,060 train questions / 280 validation questions.
- **Supervised stage:** 45 s, 5 epochs, best at epoch 1. Settings: learning rate 3e-4,
  τ = 0.1.
- **PPO stage:** 85 s, 40 iterations.
- **Artefacts:** `models/coarse_gnn_v2/{supervised,ppo}` (config, data hashes, history).
- **Inference:** 2–3.5 ms per question on ~330 coarse nodes (CPU).

What moved the readings (corpus validation MRR):

| Change | MRR |
|---|---|
| First model (learning rate 1e-3, targets proportional to grade) | 0.766 |
| Sharper listwise targets (τ = 0.1) + learning rate 3e-4 | 0.787 ± 0.01 (3 seeds; best single run 0.801) |
| Skip connection | 0.790 ± 0.004 (3 seeds) |
| Wider (256), deeper (3 layers), dropout 0.3 | 0.778–0.794, no gain |

Run-to-run spread is about ±0.01 on the same settings. Differences smaller than that
are noise.

### Paragraph ranking (MRR / hit@1; every L3 paragraph of the document is ranked)

| Set | BM25 | Dense | Rule-based pipeline | **GNN supervised** | GNN after PPO | Gold-label GNN* |
|---|---|---|---|---|---|---|
| corpus val (280, generated) | 0.633 / 0.564 | 0.758 / 0.650 | 0.776 / 0.700 | **0.794 / 0.707** | 0.761 / 0.636 | 0.932 / 0.893 |
| calibration_v1 (dev) | 0.859 / 0.80 | 0.950 / 0.90 | **0.975 / 0.95** | 0.917 / 0.85 | 0.900 / 0.80 | 0.975 / 0.95 |
| dev_scale_v1 (dev, long) | 0.857 / 0.80 | 0.904 / 0.85 | **0.950 / 0.90** | 0.942 / 0.90 | 0.900 / 0.80 | 0.975 / 0.95 |
| dev_pruning_v1 (dev, multi) | 0.636 / 0.45 | 0.849 / 0.73 | 0.864 / 0.73 | **0.955 / 0.91** | 0.909 / 0.82 | 0.909 / 0.82 |
| dev_pruning_scale_v1 (dev) | 0.570 / 0.36 | 0.762 / 0.64 | 0.825 / 0.73 | **0.849 / 0.73** | 0.803 / 0.64 | 0.796 / 0.64 |
| **contracts_v1 (test)** | 0.795 / 0.72 | 0.859 / 0.75 | **0.965 / 0.94** | 0.935 / 0.89 | 0.949 / 0.92 | 0.908 / 0.86 |
| **contracts_scale_v1 (test)** | 0.725 / 0.61 | 0.786 / 0.69 | **0.918 / 0.86** | 0.852 / 0.78 | 0.837 / 0.78 | 0.823 / 0.78 |

\* Trained on the generated corpus's known answers instead of Silver Paths: an upper
bound on what silver noise costs, not the spec method.

Recall@10 is 0.95–1.0 for the GNN on every set (rule-based pipeline 0.92–1.0).

### Ablations (MRR)

| Variant | corpus val | contracts_v1 | contracts_scale_v1 |
|---|---|---|---|
| Typed + skip (final) | 0.794 | 0.935 | 0.852 |
| Typed, no skip | 0.799 | 0.921 | 0.881 |
| No edges (per-node MLP) | 0.780 | 0.919 | 0.835 |
| Untyped (spec's single GraphSAGE; earlier settings) | 0.722 | 0.898 | 0.833 |
| No query input (spec formulas as written, D4) | **0.091** | **0.213** | **0.111** |

### PPO keep/prune policy (deterministic top-down, keep where p > 0.5)

| Set | Supervised: kept / answers kept | **PPO: kept / answers kept** |
|---|---|---|
| corpus val | 4.6% / 0.839 | **8.3% / 0.961** |
| contracts_v1 (28 paragraphs per document) | 10.5% / 0.792 | **14.6% / 0.931** |
| contracts_scale_v1 (~330 paragraphs) | 0.8% / 0.764 | **2.1% / 0.903** |

On long documents the PPO policy keeps about 2% of paragraphs (spec reference: < 5% of
nodes) and keeps 90% of the answers.

### End to end (`--scorer gnn` in the Phase 5–8 pipeline; pruning settings unchanged)

| Set | Scorer | Evidence recall | hit@1 | Primary precision | Pack words |
|---|---|---|---|---|---|
| contracts_v1 | linear | 1.000 | 0.944 | 0.629 | 86.9 |
| contracts_v1 | gnn | 0.944 | 0.889 | 0.719 | 60.0 |
| contracts_scale_v1 | linear | 1.000 | 0.833 | 0.501 | 102.0 |
| contracts_scale_v1 | gnn | 0.889 | 0.778 | 0.592 | 71.3 |

### Findings

1. **In domain, the GNN is the best ranker.** On generated validation documents it
   beats dense retrieval and the rule-based pipeline, and on the multi-passage dev sets
   it leads clearly (0.955 vs 0.864).
2. **On the real test contracts it is second to the rule-based pipeline**: 0.935 vs
   0.965, and 0.852 vs 0.918 on long documents. It is still far ahead of BM25 and dense
   alone.
   - The gold-label model does no better on test (0.908), so the gap is **domain
     shift**: training uses templated synthetic contracts. It is not silver noise.
   - The rule-based pipeline also has planner signals (definitions, cited sections,
     comparison sides) that the GNN does not see.
3. **Silver noise costs ~0.14 MRR in domain** (0.79 vs 0.93 with gold labels). Better
   labels matter more than model size.
4. **Edges help a little; edge types help more.** No edges is −0.01 to −0.02 MRR; the
   spec's untyped GraphSAGE is −0.07 on corpus val.
5. **The question input (D4) is essential.** Without it, MRR is 0.09–0.21.
6. **PPO trades ranking for keeping.** It raises the answers kept at a small kept share
   (90% of answers with 2% of paragraphs on long documents), and lowers MRR by 0.01–0.03.
   It is used as the keep/prune policy (G3/G5); the supervised model stays the ranker.
7. **End to end, the GNN gives smaller, more precise packs but loses some recall** (0.94
   / 0.89). The Phase 8 thresholds were tuned on linear scores and have not been re-tuned
   for GNN probabilities.

---

## G3 — Graph coarsening + FineGNN (2026-10-17)

### Setup

- **Stage 1:** paragraph super-nodes are the mean of their sentence embeddings (spec
  §3.2). CoarseGNN (G2 recipe) is retrained on them. Keep rule (D2): `p > θ · mean`, capped
  at the top 20%, minimum 5. θ = 1 was chosen on corpus val (θ 0.5–4 and cap 10–20% give
  MRR within noise; θ = 1, cap 20% keeps the most answers).
- **Stage 2:** `FineGNN` as in spec §3.4 (context projection, trainable alpha = 0.3,
  2-layer GraphSAGE, output projection), with input = G2 features + parent coarse
  probability (D4). Trained with the coarse model frozen (~1 epoch is best), then jointly
  with it. alpha stays near its initial 0.30–0.32 (it is trainable and receives
  gradient).
- **Stage 3:** merge keeps sentences with fine p > 0.8 × mean.
- **Baseline "no coarsening"** (spec v2.0 column): the same FineGNN over every node of the
  document.
- **Models:** `models/two_stage_v1/{staged,joint}`, `models/g3_flat`,
  `models/g3_nocontext`, `models/g3_direct`.

### Sentence-level ranking and stages (two-stage = `two_stage_v1/joint`)

Columns: MRR, hit@1, answer paragraph kept by stage 1, answers in the merged set,
share of sentences merged, nodes in the fine graph.

| Set | Method | MRR | hit@1 | Stage-1 recall | Merged recall | Merged share | Fine nodes |
|---|---|---|---|---|---|---|---|
| corpus val (795 nodes/doc) | two-stage | 0.778 | 0.686 | 0.986 | 0.977 | 6.4% | 92 |
| | no coarsening | 0.791 | 0.704 | 1.0 | 1.0 | 18.3% | 795 |
| | dense (sentences) | 0.789 | 0.686 | | | | |
| dev_scale_v1 (615 nodes/doc) | two-stage | **0.950** | 0.90 | 1.0 | 1.0 | 2.6% | 65 |
| | no coarsening | 0.917 | 0.85 | 1.0 | 1.0 | 16.4% | 619 |
| dev_pruning_scale_v1 | two-stage | 0.712 | 0.55 | 1.0 | 0.909 | 2.7% | 57 |
| | no coarsening | 0.742 | 0.55 | 1.0 | 1.0 | 17.0% | 617 |
| **contracts_v1 (test)** | two-stage | 0.887 | 0.806 | 0.958 | 0.889 | 13.0% | 15 |
| | no coarsening | 0.891 | 0.833 | 1.0 | 0.958 | 19.7% | 56 |
| | dense / BM25 | 0.881 / 0.775 | | | | | |
| **contracts_scale_v1 (test)** | two-stage | 0.835 | 0.750 | 0.958 | 0.931 | **2.5%** | **63** |
| | no coarsening | 0.851 | 0.778 | 1.0 | 1.0 | 17.0% | 620 |
| | dense / BM25 | 0.826 / 0.730 | | | | | |

Ablations (MRR; corpus val / contracts_v1 / contracts_scale_v1):
- no parent context (alpha path off): 0.750 / 0.897 / 0.862
- direct paragraph embeddings instead of mean pooling: 0.782 / 0.885 / 0.839
- staged only (no joint fine-tune): 0.776 / 0.887 / 0.837

### Spec path end to end (`gnn/spec_qa.py`: stages → merge → Phase 8 dependencies → pack)

| Set | Evidence recall | hit@1 | Primary precision | Pack words | Pack / document words | Latency (coarse + fine) |
|---|---|---|---|---|---|---|
| calibration_v1 | 1.000 | 0.950 | 0.592 | 58 | 17% | 3.1 ms |
| dev_pruning_v1 | 0.909 | 0.818 | 0.606 | 74 | 23% | 3.3 ms |
| dev_pruning_scale_v1 | 0.909 | 0.727 | 0.408 | 123 | 2.8% | 5.2 ms |
| **contracts_v1 (test)** | **0.986** | 0.833 | 0.520 | 69 | 20% | 3.1 ms |
| **contracts_scale_v1 (test)** | **0.958** | 0.778 | 0.286 | 124 | **2.8%** | 5.1 ms |

For comparison, the rule-based pipeline (Phases 4–8) gets 1.0 / 0.944 (small) and
1.0 / 0.833 (scale) recall / hit@1, with 87 / 102 pack words.

### Scale benchmark (`gnn/bench.py`, generated documents with known answers, 8 questions each)

| Pages | Words | Nodes | Paragraphs | Embed (offline) | Two-stage: coarse + fine | Fine nodes | Stage-1 recall | Merged recall / share | No coarsening |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 50k | 5,409 | 1,596 | 5.7 s | 8 + 4 = **12 ms** | 426 | 1.0 | 1.0 / 2.4% | 39 ms, 5,409 nodes, merged share 17.5% |
| 1,000 | 500k | 51,459 | 15,048 | 54 s | 112 + 85 = **198 ms** | 5,795 | 1.0 | 1.0 / 4.4% | 574 ms, 51,459 nodes, merged share 21.9% |
| 10,000 | 5.0M | 510,910 | 149,475 | 547 s | 1,946 + 2,129 = **4.07 s** | 47,931 | 1.0 | 1.0 / 3.1% | not run: needs ~5 GB of activations on an 8 GB machine |

Spec reference: 80 ms on 10,000 pages (4× A100, distributed); 90% fewer nodes processed.

With an absolute cap on stage 1 (`max_keep = 200` paragraphs, an option added in G3):

| Pages | Coarse + fine | Fine nodes | Stage-1 / merged recall | Merged share |
|---|---|---|---|---|
| 100 | 12 + 5 = 18 ms | 420 | 1.0 / 1.0 | 2.4% |
| 1,000 | 65 + 7 = **72 ms** | 806 | 1.0 / 1.0 | 0.8% |
| 10,000 | 1,230 + 115 = **1.34 s** | 960 | 1.0 / 1.0 | 0.12% |

### Findings

1. **Coarsening keeps quality close to single-pass and cuts the work by ~90%.**
   - Fine nodes: 92 vs 795 on corpus val; 63 vs 620 on the long test documents.
   - MRR is 0.4–1.6 points below no coarsening on most sets, and above it on dev_scale.
   - The spec's reference is +2–3%; we measure about −1 point. The loss comes from the 1–2
     test questions whose answer paragraph stage 1 drops (stage-1 recall 0.958).
2. **The merged output is small.** On long documents 2.5% of sentences are merged (spec:
   0.05–5%) with 93% of answers (no coarsening: 17% of sentences). The evidence pack is
   2.8% of the document's words.
3. **End to end, the spec path reaches evidence recall 0.986 / 0.958 on the test sets**
   with no planner, vs 1.0 / 1.0 for the rule-based pipeline. hit@1 is lower (0.83 vs
   0.94 small; 0.78 vs 0.83 scale), and coarse + fine latency is 3–5 ms.
4. **Scaling.**
   - With the spec rule (top 20%), the fine stage grows with the document: 4.1 s at 10,000
     pages on CPU.
   - With an absolute cap of 200 paragraphs, the fine stage is constant (~1,000 nodes,
     5–115 ms) at no recall cost, and 10,000 pages take 1.34 s.
   - What remains is the coarse pass over 239k nodes (1.2 s), the part G5 distributes.
   - The no-coarsening baseline could not run at 10,000 pages on this 8 GB machine.
     Coarsening is what makes that size feasible here.
5. **Spec details that don't show a clear effect at this size.**
   - Parent-context injection (alpha): with vs without is mixed (corpus val +0.028,
     test −0.01 / −0.027), within seed noise, so its benefit is unconfirmed.
   - alpha barely moves from 0.3.
   - Mean pooling vs direct paragraph embeddings: same within noise.
   - Joint fine-tuning adds ≤ 0.002.

---

## G4 — Cross-edge attention (2026-10-18)

### Setup

- **Module:** `CrossEdgeAttention` as in spec §4.4. `self.alpha` is defined (D6).
- **Local attention:** a 512-node sliding window over document order, computed in
  blocks (exact band attention).
- **Global summary:** mean of the hidden states of the top-100 nodes.
- **Score** (§4.2 plus a learned correction):
  `final = local + head(attended) + β·sigmoid(cos(attended, GS))`. The correction head
  starts at zero.
- **Training:** on top of `two_stage_v1/joint`, attention trained with the GNN frozen,
  then joint (curriculum). Attention is used at both stages unless stated. 3 seeds.
- **Models:** `models/g4/*`.
- **Cross-chapter sets:**
  - corpus val: 40 tagged questions
  - `dev_crosschapter_v1` (+ `_scale_v1`): 10 new real-contract questions, all spanning 2
    chapters
  - test: the `contracts_v1` / `contracts_scale_v1` questions whose answers sit in 2 or
    more chapters, selected automatically (7 each)
- **Metrics:** "complete" = every answer snippet found.

### Cross-chapter readings

Columns: paragraph complete@10 / stage-1 complete / merged complete. G4 is the mean of
3 seeds.

| Set (n) | G3 (no attention) | **G4 cross-edge attention** | Rule-based pipeline (explicit reference following) |
|---|---|---|---|
| corpus val (40) | 0.850 / 0.950 / 0.950 | **0.917 / 1.000** / 0.892 | 0.700 / – / – |
| dev cross-chapter (10) | 1.000 / 0.800 / 0.700 | 1.000 / 0.800 / 0.700 | 1.000 |
| dev cross-chapter, long (10) | 1.000 / 1.000 / 0.600 | 0.900 / 1.000 / 0.667 | 0.800 |
| **test contracts_v1 (7)** | 1.000 / 0.714 / 0.714 | 1.000 / 0.714 / 0.667 | 1.000 |
| **test contracts_scale_v1 (7)** | 0.714 / 0.714 / 0.714 | 0.714 / 0.762 / 0.714 | **1.000** |

Spec reference: cross-chapter recall 78% → 94% (+16 points).

All questions, sentence MRR (G3 → G4 mean of 3 seeds):
- corpus val: 0.778 → 0.791
- contracts_v1: 0.887 → 0.894
- contracts_scale_v1: 0.835 → 0.837
- dev cross-chapter long: 0.858 → 0.895

### Spec path end to end, complete evidence (every answer in the pack)

| Set | G3 | G4 (seed 0) | Pack words G3 → G4 |
|---|---|---|---|
| dev cross-chapter | 0.90 | **1.00** | 86 → 89 |
| dev cross-chapter, long | 0.80 | **0.90** | 120 → 135 |
| contracts_v1 (test) | 0.972 | 0.972 | 69 → 77 |
| contracts_scale_v1 (test) | 0.944 | **0.972** (evidence recall 0.958 → 0.986) | 124 → 152 |

### Ablations (corpus val: cross-chapter complete@10, all-question MRR)

| Variant | Cross-chapter complete@10 | All-question MRR |
|---|---|---|
| Full (both stages, 3 seeds) | 0.917 | 0.791 |
| Local window only | 0.900 | 0.765 |
| Global summary only | 0.875 | 0.775 |
| Coarse stage only | 0.900 | 0.776 |
| Fine stage only (the spec's placement) | 0.850 | 0.783 |

### Cost

| Pages | Coarse stage without → with attention | Fine nodes without → with | Total |
|---|---|---|---|
| 1,000 | 72 → 376 ms | 5,795 → 2,371 | 126 → 417 ms |
| 10,000 | 3.3 → 7.1 s | 47,931 → 16,010 | 8.06 → 8.13 s |

Spec reference: ~30 ms overhead (GPU). On CPU, windowed multi-head attention over
240k nodes costs about 4 s. Its boost changes the θ·mean keep rule's cut, so the fine
stage gets smaller, and the total at 10,000 pages ends up the same.

### Findings

1. **In domain, attention helps cross-chapter retrieval.** On corpus val, paragraph
   complete@10 rises from 0.85 to 0.92 and stage-1 complete from 0.95 to 1.00 (all 3
   seeds). All-question MRR rises by +0.013. It is a smaller gain than the spec's +16
   points.
2. **On the real contracts it is within noise.** The cross-chapter sets have 7–10
   questions; seeds disagree by ±1 question. End to end, the spec path finds complete
   evidence more often (scale test 0.944 → 0.972; dev 0.90 → 1.00), partly because it
   keeps more (packs 15–25% larger).
3. **The rule-based pipeline still wins cross-chapter on the real test contracts**
   (1.0 vs 0.71 on long documents). Its explicit reference following does what learned
   attention does not yet do on unseen documents. On generated documents the GNN path
   wins (0.92 vs 0.70).
4. **Both systems matter.** Local-only or global-only each lose part of the gain.
   Attention at the fine stage only (where the spec places it) helps least: by then the
   paragraphs are already chosen.
5. **The GlobalSummary is the only cross-shard input** (tested), as G5 needs.

---

## G5 — Distribution: partitioning, boundary replication, Ray (2026-10-20)

### Setup

- **Model:** `models/g4/cea_s0/joint` (two-stage + cross-edge attention).
- **Documents:** generated, 1,000 pages (24k coarse nodes) and 10,000 pages (239k coarse
  nodes); 8 known-answer questions each.
- **Workers:** Ray actors on this Mac (8 cores, 8 GB), threads per worker = 8 / workers.
- **Partitioning:**
  - tree-cut: whole chapters, contiguous, balanced
  - METIS (`pymetis`)
- **Halo modes:**
  - spec: the last/first N = 3 nodes at each shard edge
  - exact: the ±256-position attention window plus the 2-hop GNN neighbourhood
- **Protocol:**
  1. stats reduce
  2. GNN pass + local top-K
  3. GlobalSummary broadcast + attention
  4. aggregator: keep rule, FineGNN, merge (> 0.8 × mean), stitching
- **Parity:** max |distributed − single-process| over all coarse logits; top-50 paragraph
  agreement.

### 10,000 pages (tree-cut)

Columns: replicated (halo) nodes, max logit difference, top-50 agreement, coarse stage,
total per question, stage-1 / merged recall.

| Halo | Workers | Replicated | Max logit diff | Top-50 agree | Coarse | Total | Stage-1 / merged recall |
|---|---|---|---|---|---|---|---|
| exact | 1 | 0% | 6e-6 | 1.00 | 7.2 s | 9.4 s | 1.0 / 1.0 |
| exact | 2 | 43% | 7e-6 | 1.00 | 9.2 s | 12.2 s | 1.0 / 1.0 |
| exact | 4 | 96% | 7e-6 | 1.00 | 15.7 s | 18.7 s | 1.0 / 1.0 |
| spec (N = 3) | 1 | 0% | 6e-6 | 1.00 | 7.6 s | 9.8 s | 1.0 / 1.0 |
| spec (N = 3) | 2 | 0% | 3.64 | 0.94 | 6.7 s | 9.2 s | 1.0 / 1.0 |
| spec (N = 3) | 4 | 0% | 4.69 | 0.92 | **5.8 s** | **7.9 s** | 1.0 / 1.0 |

### 1,000 pages

| Partition | Halo | Workers | Replicated | Max logit diff | Top-50 agree | Coarse | Total |
|---|---|---|---|---|---|---|---|
| tree | exact | 1 / 2 / 4 | 0 / 44 / 101% | ≤ 7e-6 | 1.00 | 494 / 515 / 854 ms | 592 / 586 / 967 ms |
| tree | spec | 1 / 2 / 4 | 0 / 0 / 0.1% | 4e-6 / 3.93 / 3.93 | 1.00 / 0.96 / 0.95 | 391 / 361 / **307 ms** | 445 / 430 / **373 ms** |
| METIS | exact | 2 / 4 | 76 / 158% | ≤ 7e-6 | 1.00 | 607 / 859 ms | 669 / 953 ms |

Other costs:
- **Shard loading** (Parquet, memory-mapped, plus model): 1.4–2.3 s per worker at 1,000
  pages; 2.8–4.5 s at 10,000.
- **Writing shards:** 7.5–12 s at 10,000 pages.
- **Ray startup:** 4–21 s (first start).
- **Aggregator round trips:** the stats reduce takes 3–21 ms at 1,000 pages and 0.1–1.0 s
  at 10,000; the summary takes ≤ 11 ms.
- **Aggregator work** at 10,000 pages: fine stage 1.7–2.2 s, merge and stitching
  0.3–0.9 s.

Spec reference: 80 ms on 10,000 pages and 300 ms on 100,000 pages with 100 workers
(4× A100).

### Findings

1. **Distribution is correct.** In exact mode, distributed scores equal single-process
   scores (≤ 7e-6) for every configuration tested: 1–4 workers, tree-cut and METIS, at
   1,000 and 10,000 pages. This also holds in the tests, including with real Ray actors.
   The only cross-shard data is three scalars (stats) and the GlobalSummary vector.
2. **The spec's N = 3 boundary replication is not exact.** Logits drift by up to 3.6–4.7
   (top-50 paragraph agreement 0.92–0.96). It is still faithful enough here: the answers
   were kept by stage 1 and merged for all 8 questions at both sizes. A 2-layer GNN reads
   2 hops, and references and chapter headings cross shard edges, so 3 sequential nodes
   cannot cover them.
3. **Exact parity is expensive** for this graph shape. A chapter heading is linked to all
   of its paragraphs, so a 2-hop halo pulls in whole neighbouring chapters: +43% nodes
   with 2 workers, +96–158% with 4.
4. **On one 8-core machine, more workers buy little.** A single process already uses all
   cores (PyTorch threads).
   - Best case: spec halo with 4 workers, 1.3× faster on the coarse stage (10,000 pages
     7.6 → 5.8 s; 1,000 pages 391 → 307 ms).
   - Exact halos are slower with more workers.
   - The spec's gains assume separate machines and GPUs. Here the measured benefit is
     memory per worker (each holds ¼ of the document) and a working protocol.
5. **At 10,000 pages the aggregator's own work is about 2–3 s** (fine stage, merge,
   stitching). Distributing the coarse pass alone cannot reach the spec's 80 ms; the fine
   stage would need the `max_keep` cap (G3: fine ~115 ms) and GPU attention.

---

## G6 — The spec's tables, measured (2026-10-21)

### Setup

- **Systems:**
  - **v3.0:** spec path (coarsening + FineGNN + cross-edge attention + merge + Phase 8
    dependencies; `models/g4/cea_s0/joint`)
  - **v3.0+cap:** the same with `max_keep = 200`
  - **G3:** two-stage, no attention
  - **v2.0:** the spec's "no coarsening, no attention" column (FineGNN over every node,
    `models/g3_flat`)
  - **rule:** the Phase 4–8 pipeline
- **Hardware:** one Mac, CPU, 8 GB, single process.
- **Scale documents:** generated 100 / 1,000 / 10,000-page contracts with 12 known-answer
  questions each (`data/bench/`, seed 21). At 10,000 pages each system runs in its own
  process: together they exceeded 8 GB and thrashed.
- **Tokens:** counted with the MiniLM WordPiece tokenizer.
- Code: `python -m gnn_pruner.gnn.g6 scale | quality | adversarial | sheet`.

### Spec Table 8: latency per question, p50 (p99)

| Document | Nodes (measured) | v3.0 | v3.0+cap | G3 | "v2.0" (no coarsening) | Rule-based | Spec v3.0 / v2.0 |
|---|---|---|---|---|---|---|---|
| 100 pages | 5,409 | 53 ms (143) | 51 ms (56) | **23 ms** (27) | 49 ms (73) | 60 ms (97) | 25 / 50 ms |
| 1,000 pages | 51,459 | 463 ms (1,430) | 431 ms (450) | **165 ms** (390) | 419 ms (1,282) | 1,517 ms (1,966) | 40 / 250 ms |
| 10,000 pages | 510,910 | 9.9 s (17.1) | 6.7 s (20.1) | **4.7 s** (14.9) | not run: > 8 GB | 167 s (221); index build 6 min | 80 ms / ~2.5 s |

Answer recall on these generated documents (evidence recall / complete / hit@1):

| | v3.0 | v3.0+cap | G3 | v2.0 | Rule-based |
|---|---|---|---|---|---|
| 100 pages | 1.0 / 1.0 / 0.67 | 1.0 / 1.0 / 0.67 | 1.0 / 1.0 / 0.67 | 1.0 / 1.0 / 0.67 | 0.75 / 0.75 / 0.58 |
| 1,000 pages | 0.88 / 0.83 / 0.58 | 0.88 / 0.83 / 0.67 | 0.96 / 0.92 / 0.75 | 0.88 / 0.83 / 0.58 | 0.71 / 0.67 / 0.67 |
| 10,000 pages | 0.83 / 0.83 / 0.67 | 0.92 / 0.92 / 0.67 | 0.92 / 0.92 / 0.50 | – | 0.92 / 0.92 / 0.75 |

Questions per hour on one Mac at 10,000 pages (from p50): v3.0 ~360, v3.0+cap ~540,
G3 ~760, rule-based ~22.

### Spec Table 9: accuracy

| Metric (set) | v3.0 | G3 | "v2.0" | Rule-based | Spec v2.0 → v3.0 |
|---|---|---|---|---|---|
| Node-level Recall@10, paragraphs (test, small) | 0.958 | 0.958 | **1.000** | **1.000** | 92% → 95% |
| Recall@10 (test, long) | 0.958 | 0.931 | 0.944 | **1.000** | |
| Recall@10 (generated val, 280 q) | **0.955** | 0.950 | 0.948 | 0.916 | |
| Structural coherence (test small / long / generated val) | 0.99 / 0.98 / 0.99 | 1.0 / 0.96 / 0.99 | 0.99 / 1.0 / 0.99 | 0.99 / 1.0 / 0.98 | 87% → 91% |
| Cross-chapter complete@10 (test small, 7 q / long, 7 q) | 0.71 / 0.71 | 0.71 / 0.71 | **1.0** / 0.86 | **1.0 / 1.0** | 78% → 94% |
| Cross-chapter complete@10 (dev 10 q / dev long 10 q) | 0.90 / 0.90 | 0.80 / 0.90 | **1.0 / 1.0** | **1.0** / 0.80 | |
| Cross-chapter complete@10 (generated val, 40 q) | **0.925** | 0.900 | **0.925** | 0.700 | |
| Adversarial robustness: hit@1 kept under perturbation (100 q) | 0.83 (0.95 → 0.79) | **0.88** (0.85 → 0.75) | 0.82 (0.90 → 0.74) | 0.86 (1.00 → 0.86) | 76% → 84% |
| Adversarial: Recall@10 under perturbation | 0.99 (base 1.0) | 0.99 | 0.99 | 0.99 | |
| Human preference, blind A/B v3.0 vs rule-based, 20 contracts_v1 questions | **50%** (6 v3.0 / 6 rule / 8 same; each ≥ as good 70%) | | | 50% | 62% → 71% (vs full context) |

Definitions (D11):
- **Structural coherence:** for every primary evidence item, the share of what it needs
  (definitions of terms it uses, clauses it cites) that is also in the pack. Section
  paths are always present.
- **Adversarial suite** (`eval/adversarial_v1.jsonl`): 20 generated-validation questions
  × 5 perturbations: distractor topic appended, wrong number asked, parties swapped,
  typos, topic and verb paraphrased. The 20 originals are in
  `eval/adversarial_base_v1.jsonl`.

Adversarial hit@1 by perturbation (v3.0 / rule-based):

| Perturbation | v3.0 | Rule-based |
|---|---|---|
| distractor | 0.90 | 0.95 |
| wrong number | 0.80 | 1.00 |
| party swap | 0.75 | 0.80 |
| typo | 0.70 | 0.75 |
| paraphrase | 0.80 | 0.80 |

Typos are the only perturbation that costs recall (0.95).

### Spec Table 10: tokens sent to the LLM, and cost per query

| Document | Document tokens | v3.0 pack tokens | Share | Rule-based pack | Share | Spec v2.0 → v3.0 |
|---|---|---|---|---|---|---|
| test contracts (small, ~550 tokens) | | ~100 | 22% | ~115 | 26% | 10% → 5% |
| test contracts (long, ~5,900 tokens) | | ~200 | 3.5% | ~135 | 2.3% | |
| 100 pages | ~51.6k | 372 | **0.72%** | 266 | 0.51% | |
| 1,000 pages | ~0.5M | 394 | **0.08%** | 324 | 0.06% | |
| 10,000 pages | 5.09M | 377 | **0.007%** | 300 | 0.006% | |

Packs stay at ~300–400 tokens whatever the document size, so the share falls as the
document grows. On long documents both systems are far below the spec's 5%.

LLM input cost per query = `pack_tokens × price_per_input_token` (plus the answer's
output tokens), against `document_tokens × price` for full context. At 10,000 pages that
is ~377 vs ~5.09M input tokens, ~13,500× fewer (full context would not fit any model's
window anyway). The per-token price is yours to set; no prices are assumed here.
Compute cost per query on this Mac is just the latency above: one machine, no cluster
fee.

### Findings

1. **Speed.**
   - Fastest everywhere: the two-stage model without attention (G3).
   - At 10,000 pages, G3 is ~35× faster than the rule-based pipeline (4.7 s vs 167 s),
     v3.0+cap ~25×.
   - The spec's 80 ms assumes GPUs and a cluster. On one CPU we measure 4.7–9.9 s;
     cross-edge attention costs ~2× (v3.0 vs G3).
2. **Quality on the real test contracts.** The rule-based pipeline is still best (Recall@10
   1.0, cross-chapter 1.0). The learned systems are close on everything except
   cross-chapter questions, where stage 1 drops one of the two chapters (0.71).
3. **On generated documents (the GNN's training domain)**, v3.0 is best on Recall@10 and
   cross-chapter recall (0.925 vs 0.70 rule-based).
4. **Coarsening costs some cross-chapter recall.** "v2.0" (no coarsening) beats v3.0 on
   cross-chapter questions on small documents (1.0 vs 0.71), but it cannot run at 10,000
   pages on 8 GB.
5. **Robustness.** All systems keep recall under every perturbation except typos (0.95).
   The top-ranked passage suffers more: typos and party swaps hurt most, and the
   rule-based pipeline is slightly more stable.
6. **Token reduction far exceeds the spec's target on long documents:** packs are a
   constant ~370 tokens.
7. **Found and fixed along the way:**
   - **Phase 8 conflict detection scanned the whole document per evidence item.** That
     made 10,000-page questions ~55 s (v3.0) and ~3 min (rule-based). It is now indexed:
     identical results, ~37× faster.
   - **The rule-based pipeline's index build doesn't scale:** 6 minutes and 2.7 GB at
     10,000 pages, and its semantic-edge step is O(n²), so it was switched off above
     5,000 paragraphs.

### Human preference (rated 2026-10-21; `docs/human_preference_ratings.json`)

Blind A/B on 20 test questions (`contracts_v1`), v3.0 vs rule-based evidence packs.

- **Result:** v3.0 preferred 6, rule-based 6, same 8, so **50% preference: a tie**.
  The preferred pack was sufficient on its own in **20/20**.
- **Pattern:** in 11 of the 12 decided pairs the rater chose the **shorter** pack that
  still answered. The exception was the liability summary (Q2), where the fuller pack won.
  - v3.0 lost on definition and lookup questions (it added neighbouring clauses).
  - Rule-based lost on paraphrased questions (its packs grew to ~10 items).
- **Comparability:** the spec's 62% → 71% compares pruned context against full-context
  answers. That needs LLM answers (Phase 10), so the two numbers are not directly
  comparable. The ratings point at **pack length** as the lever: the Phase 9 / 10 token
  budget, and the merge threshold for lookup questions.

---

## Phase 9 — Topology-routed prompt + evidence pack (2026-10-21)

### Setup

- **Code:** `prompt.py` (routing, templates, budget), `evidence.py` (pack, citations,
  citation map).
- **Routes** (spec thresholds; precedence is D12):
  - SYNTHESIS: clause citations between kept items > 2
  - SIBLING: items in one section > 3
  - PATH: depth > 3
  - QA: otherwise
- **Budgets** (evidence tokens per route, chosen on the dev sets): QA 250, SYNTHESIS 250,
  SIBLING 300, PATH 300.
- **Tokens:** word and punctuation pieces.
- **Prompt overhead:** rules + template + format ≈ 215–230 tokens.

### Budget sweep on the dev sets (82 questions per system; evidence recall / evidence tokens)

| Budget | Rule-based | v3.0 |
|---|---|---|
| none | 0.963 / 107 | 0.988 / 135 |
| 100 | 0.909 / 70 | 0.951 / 85 |
| 150 | 0.951 / 95 | 0.970 / 113 |
| 200 | 0.963 / 105 | 0.982 / 127 |
| 250+ | 0.963 / 107 | 0.988 / 134 |

Per route:
- **SYNTHESIS:** both systems lose nothing at 200.
- **QA:** v3.0 loses 1% at 200 and nothing at 250.
- **SIBLING:** rule-based loses answers below 250 (n = 3).
- **PATH:** never occurred on the dev or test contracts. They nest at most chapter >
  subsection > paragraph.

### Readings with the chosen budgets

| Set | System | Routes (QA / SYNTH / SIB / PATH) | Evidence recall no budget → budget | Evidence tokens | Prompt tokens |
|---|---|---|---|---|---|
| test (72 q: contracts_v1 + scale) | rule-based | 55 / 16 / 1 / 0 | 1.000 → 1.000 | 112 → 112 | 330 |
| | v3.0 | 52 / 17 / 3 / 0 | 0.986 → 0.986 | 136 → 135 | 362 |
| generated val (280 q) | rule-based | 178 / 8 / 19 / 75 | 0.886 → 0.877 | 208 → 198 | 460 |
| | v3.0 | 114 / 1 / 71 / 94 | 0.952 → 0.941 | 269 → 243 | 540 |

### Findings

1. **On the real contracts, Phase 8 packs are already short** (98–159 tokens). The budget
   never binds and costs nothing.
2. **On longer, deeper documents (generated val) all four routes occur.** The budget
   trims ~10% of evidence tokens (v3.0 269 → 243) for about −1 point of answer recall.
   The human ratings preferred shorter sufficient packs, so the budgets stay.
3. **SYNTHESIS fires for clause chains**, e.g. the availability → Service Credits →
   99.9% qualifier, with the conflicting "one hour" support clause flagged. PATH fires for
   sentences in sub-sections (`9.2 Price Changes`). SIBLING fires for several provisions
   of one section.
4. **Every pack is deterministic and every label maps back to its node.** The source
   offsets reproduce the item text exactly (tests).
5. **Not measured yet: whether routed templates produce better answers than one generic
   prompt.** That needs an LLM (Phase 10 eval).

---

## Phase 10 — Answer generation (2026-10-21)

### Setup

- **Code:** `answer.py` (providers, grounded `Answer`, citation validation, abstention)
  and `answer_eval.py` (readings).
- **Providers:**
  - **Claude** (`claude`, `claude:<model>`): Anthropic SDK; default model
    `claude-opus-5-5`, effort `medium`, JSON-schema output, server-side refusal fallback
    on.
  - **OpenAI** (`openai:<model>`): Chat Completions with a JSON-schema
    `response_format`; you name the model.
  - **Extractive:** offline, no LLM.
- **Run:**
  - live: `python -m gnn_pruner.answer_eval --sets eval/contracts_v1.jsonl --system rule|v3.0 --llm claude [--judge claude]`
  - estimate first: `--dry-run`

### Offline baseline: extractive answers (top evidence item, cited) on the test sets

Columns: answer F1 vs reference, number match, citation precision / recall, abstention
on unanswerable, false abstention.

| System | Set | Answer F1 | Number match | Citation P / R | Abstained (unanswerable) | False abstention |
|---|---|---|---|---|---|---|
| rule-based, abstention on | contracts_v1 | 0.383 | 0.60 | 0.94 / 0.79 | **4/4** | 5.6% (2/36) |
| rule-based, abstention on | contracts_scale_v1 | 0.348 | 0.55 | 0.85 / 0.71 | 1/4 | 8.3% |
| rule-based, abstention off | contracts_v1 | 0.387 | 0.59 | 0.94 / 0.81 | 0/4 | 0% |
| v3.0 (spec path) | contracts_v1 | 0.354 | 0.59 | 0.83 / 0.72 | 0/4 | 0% |
| v3.0 (spec path) | contracts_scale_v1 | 0.342 | 0.50 | 0.75 / 0.65 | 0/4 | 0% |

- **Extractive answers are verbatim clauses,** so token F1 against hand-written reference
  answers stays low (~0.35). This is the floor an LLM should beat.
- **Citations are highly precise** (the top item usually holds the answer) but cover only
  one item, so recall is 0.65–0.81 on multi-clause questions.
- **Abstention comes from the Phase 7 retrieval-confidence signal,** so it behaves as
  measured there. The spec path has no such signal yet, so it never abstains.

### Live LLM readings: Gemini 3.1 Flash-Lite, free tier (2026-10-22)

- **Provider:** `gemini:gemini-3.1-flash-lite` through Google's OpenAI-compatible
  endpoint; JSON-schema output; cost $0.
- **Why this model:** the free tier allows only 20 requests per day for
  `gemini-flash-latest` (gemini-3.8-flash), which stopped the first run. Flash-Lite
  finished all 80.
- **Why not local:** local models (Ollama, phi3.5 / gemma2:9b) froze the 8 GB laptop.
- **Rows:** saved per question in `eval/runs/p10_gemini-3.1-flash-lite_{rule,rule_noabstain,v3}.jsonl`
  (`--resume` reuses them).
- **Run cost:** 240 model calls, $0.

| System | Set | Answer F1 | Number match | Citation P / R | Abstained (unanswerable) | False abstention | Prompt tokens | Latency (median) |
|---|---|---|---|---|---|---|---|---|
| rule-based, abstention on | contracts_v1 | **0.463** | **1.00** | 0.83 / **1.00** | 4/4 | 5.6% (2/36) | 368 | 12.6 s* |
| rule-based, abstention on | contracts_scale_v1 | **0.470** | **1.00** | 0.83 / **1.00** | 4/4 | 8.3% (3/36) | 412 | |
| rule-based, **abstention off** (model decides) | contracts_v1 | 0.445 | **1.00** | 0.83 / **1.00** | **4/4** | **0%** | 469 | |
| rule-based, **abstention off** (model decides) | contracts_scale_v1 | 0.467 | **1.00** | 0.83 / **1.00** | **4/4** | **0%** | 473 | |
| v3.0 (spec path) | contracts_v1 | 0.462 | **1.00** | **0.86** / 0.99 | **4/4** | **0%** | 413 | |
| v3.0 (spec path) | contracts_scale_v1 | 0.467 | **1.00** | **0.86** / 0.99 | **4/4** | **0%** | 555 | |

\* Mostly free-tier pacing: calls are spaced 6.5 s apart and retried on rate limits (2.7 s
fastest, 123 s worst after back-off). Not a property of the pipeline.

Compared with the extractive baseline:

1. **Every number in every reference answer is now stated correctly** (number match
   0.55–0.60 → 1.00).
2. **Every gold clause is cited** (citation recall 0.71–0.79 → 1.00), with no invalid
   or missing citations (0 flagged).
3. **Answer F1 rises 0.35–0.38 → 0.46–0.47.** Token F1 understates this: the model's
   answers are full sentences that carry conditions the short references leave out.
   `--judge` would grade meaning instead.
4. **Citation precision 0.83 (down from 0.85–0.94).** This is mostly the metric, not
   wrong citations. On multi-clause questions (liability caps, Misconduct, Service
   Credits) the model also cites the definitions and conditions that the dependency
   rules added to the pack, and those are not in the gold snippets.
5. **The model judges "not in the document" better than our pre-call signal.**
   - **With the Phase 7 signal on:** the signal refuses before the model is called. It
     wrongly refuses 5 answerable paraphrased questions (para-02 / 03 / 05), each of
     which the model answers correctly with full citation recall when allowed to.
   - **With the signal off, or on the spec path (no signal):** the model itself refuses
     all 8 unanswerable questions (`insufficient: true`) and no answerable ones.
   - **Design consequence:** the not-found decision should use the model's judgement,
     with retrieval confidence as a second input, not a gate in front of it. Recorded for
     the central board (REQUIREMENTS v4-3).
   - **Caveat:** 8 unanswerable questions is a small sample. The long-document benchmark
     needs many more.
6. **Both pipelines give the same answer quality with a model.** v3.0 cites slightly
   more precisely (0.86 vs 0.83) and misses one gold clause on msa-06 (recall 0.99), with
   prompts 10–30% longer on the 10-page set.
7. **Same answers on the 10-page versions as on the 1-page ones:** the pruned pack keeps
   the prompt at ~410 tokens either way.

### Dry-run estimate for paid models

No Claude or OpenAI key is set. The dry-run cost estimate for the 80 test questions:

| System | Input tokens (mean / total) | Estimated cost, Claude Opus 5.5 ($4 / $20 per 1M) |
|---|---|---|
| rule-based | 428 / 34.3k | ~$0.38 at 150 output tokens per answer; more with thinking (billed as output), likely $0.4–1.0 |
| v3.0 | 380–498 / 35.1k | ~$0.38; likely $0.4–1.0 |

The `--judge claude` option roughly doubles the cost. OpenAI costs depend on the model you
choose; pass its prices to see them in the readings.

---

## Build step 1 (v4-1) — Upload: document tree, metadata, key facts (2026-10-22)

### Setup

- **Code:**
  - `ingest/pdf_loader.py`: outline headings, text cleaning, drop caps, paragraph indents
  - `tree.py`, `metadata.py`, `enrich.py`
  - `upload.py` (CLI, cache in `data/trees/`)
  - `upload_eval.py` (readings)
- **Documents:**
  - two real public-domain books from Project Gutenberg, printed to PDF by Chrome with
    bookmarks (`data/books/`):
    - *The Adventures of Sherlock Holmes*: 144 pages, 12 stories
    - *A Tale of Two Cities*: 215 pages, 3 books, 45 chapters
  - the 10-page test contract (`examples/scale/saas_agreement.md`)
- **Model:** `gemini-3.5-flash-lite`, free tier, $0. The `gemini-3.1-flash-lite` daily quota
  ran out mid-way; `gemini-2.5-flash-lite` is retired for new users.
- **Batching:** 6,000 tokens / at most 15 items per call; at most 8 facts per subsection.
- **Metadata key:** `data/books/key.json`. The answer key marks "year" and "ISBN" as *not in
  the text*: the Gutenberg files state neither.
- **Raw results:** `eval/runs/step1_upload_readings.json`. Spot-check sheet:
  `docs/fact_check_sheet.md`.

### Structure (the PDF outline is the reference)

| Book | Outline entries | Fonts only: found / false headings | With outline: found / false |
|---|---|---|---|
| Sherlock Holmes | 20 | 35% / 30 | **100% / 0** |
| Two Cities | 52 | 77% / 46 | **100% / 0** |

- **Fonts alone fail on real book PDFs:**
  - drop caps ("I", "A", "W") became top-level sections
  - headings split over two lines became two headings
  - title-page lines became headings
- **With the outline, every entry is matched to the text.** This includes split headings
  ("I." + "A SCANDAL IN BOHEMIA"). "Book the First" contains its chapters even though the
  outline lists both at the same level.
- **Paragraph fix:** paragraphs indented exactly one em had been merged into page-long
  blocks (212 → 879 paragraphs on Sherlock).

### The trees

| Document | Pages | Parts / chapters / subsections | Passages / sentence ids | Tokens | Build time |
|---|---|---|---|---|---|
| Sherlock Holmes | 144 | 0 / 16 / 136 | 879 / 5,768 | 132k | 11.7 s |
| Two Cities | 215 | 3 / 49 / 164 | 1,172 / 6,967 | 174k | 15.6 s |
| SaaS contract (10 pages) | — | 0 / 45 / 46 | 291 / 301 | 5k | <0.1 s |

Long chapters without sections are sliced into ~1,200-token subsections (Sherlock's stories:
9–13 each).

### Metadata (answer key: title, author, year, ISBN)

| Book | Correct | Notes |
|---|---|---|
| Sherlock Holmes | 4/4 | title and author from labelled lines; year and ISBN correctly left empty |
| Two Cities | 4/4 | same |

- **A first run caught a wrong year.** The model proposed "1999" from the e-book's release
  line. That value was "in the text", so the evidence rule alone let it through. Fixed with
  an instruction plus a rule rejecting years taken from release / update lines (tested).
- **Publisher:** "Project Gutenberg" (stated in the licence text) is kept for Sherlock; it is
  not in the key.

### Key facts

| Document | Proposed | Kept | Dropped: bad sentence ids | Dropped: number not in source |
|---|---|---|---|---|
| Sherlock Holmes | 582 | 511 (**87.8%**) | 56 | 15 |
| Two Cities | 592 | 575 (**97.1%**) | 3 | 14 |
| SaaS contract | 293 | 293 (**100%**) | 0 | 0 |

**How the checks were made fair** (measured, in order):
1. **First run (Sherlock, `gemini-3.1-flash-lite`): 49% kept.**
   - The model cited paragraph ids where sentence ids were asked for. These are now
     accepted and expanded to their sentences.
   - The model pasted ids into the fact text ("… at no cost. [p2.s1]"), and their digits
     failed the number check. Ids are now stripped first.
2. **First 42 subsections after the fixes: 96% kept.**
3. **Facts filed under the wrong item of a batch** move to the subsection that owns their
   sentences (3 per book on re-check).

**What is still dropped:**
- **Sherlock:** 50 facts cite sentence numbers that do not exist (e.g. `p275.s22` in a
  shorter paragraph), and 17 cite paragraphs outside their item. These are unverifiable,
  so dropping them is correct. Sherlock's long dialogue paragraphs make the model lose
  count; Two Cities has 3 such facts.
- **Number mismatches** are mostly wording: "1775" vs "one thousand seven hundred and
  seventy-five". These cannot be confirmed automatically, so they are dropped.

**Quality:**
- Summaries are coherent at every level (root, parts, chapters). For example, Part 1:
  "Covers Book the First … Introduces Mr. Jarvis Lorry, Miss Manette …".
- Some kept facts overreach their sentences. For example, Sherlock #12: "Rucastle and Miss
  Hunter are parties to the employment arrangement".
- Fact accuracy beyond the automatic checks needs the human spot check
  (`docs/fact_check_sheet.md`, 36 facts). **Pending.**

### Lists (typed facts, for lookup questions)

- **SaaS contract:** 130 obligations, 56 conditions, 45 rights, 29 deadlines, 25 amounts,
  4 definitions.
- **Books:** people, places, events and dates, e.g. Two Cities: 113 events, 73 people.

### Cost (free tier; tokens are input + output)

| Document | Model calls | Tokens | Time | Calls per 100 pages |
|---|---|---|---|---|
| Sherlock Holmes | 31 | 259k | 5.6 min | 21.5 |
| Two Cities | 40 | 323k | 6.8 min | 18.6 |
| SaaS contract | 11 | 38k | 1.9 min | — |

- **Time** is mostly the free-tier pacing (6.5 s between calls).
- **At paid Claude prices** (Haiku 4.5, batch: $0.50 / $2.50 per 1M), Two Cities would cost
  about $0.25. That is in line with the $1–3 per 1,000 pages estimate.

### Bugs found and fixed by these readings

1. **Run reported "complete" without being complete:** 2 unparseable replies left 30 of 46
   contract subsections without facts.
   - **Cause:** Gemini 3.x counts reasoning against the output cap, which was 4,000.
   - **Fix:**
     - the cap is now 16,000
     - unanswered items are retried in the same run, then fall back to offline facts
     - "complete" is now checked
2. **The paragraph-indent and drop-cap issues** above.

### Known limits

- **Hyphenated words across line breaks lose their hyphen** ("boarding-school" →
  "boardingschool").
- **Not yet tested:**
  - a real contract PDF with parties, dates and governing law (the test contracts state
    none)
  - scanned PDFs (no OCR)

---

## Build step 2 (v4-2) — Long-document benchmark (2026-10-23)

### The benchmark (`longbench.py`; documents in `data/longbench/`, questions in `eval/long/`)

| Document | Pages (3,000 chars) | Units | Questions |
|---|---|---|---|
| contracts S / M / L | 130 / 977 / **8,619** | 5 / 47 / 510 real CUAD contracts | 42 / 109 / 176 |
| books S / M / L | 119 / 1,001 / **9,988** | 1 / 6 / ~52 Gutenberg novels (NarrativeQA test set) | 42 / 94 / 200 |

- **Nested:** S is inside M is inside L, so every question is asked at every size it fits.
- **Question kinds:**
  - **CUAD clauses:** the lawyers' spans are the gold evidence; clauses CUAD marks absent
    become unanswerable questions
  - **NarrativeQA:** two human answers, no spans
  - **Planted facts ("needles"):**
    - single facts with reworded questions, plus a look-alike distractor elsewhere
    - two-hop facts (contracts: a rate card in another contract; books: distant chapters)
    - questions about facts never planted
- **Licences:** CUAD CC BY 4.0, NarrativeQA Apache 2.0, Gutenberg texts public domain.

**Scoring:**
- **Evidence hit:** word-3-gram coverage of the gold span ≥ 0.5; a two-hop question needs
  both parts.
- **Answers:**
  - planted facts: all content words of the planted value
  - unanswerable questions: the model abstains
  - CUAD and NarrativeQA: an LLM judge (`gemini-3.1-flash-lite`, a different model from the
    one that answers)
- **Free:** all answers from `gemini-3.5-flash-lite`, $0.

### Retrieval only (no model; does the pack contain the gold?)

| Size | Family | Plain search k=5 (~1,200 tok) | Plain search k=1 (~200 tok) | Rule pipeline (~270 tok) | GNN pipeline (~270 tok) |
|---|---|---|---|---|---|
| S | contracts: CUAD / needle / 2-hop | 60% / 67% / 0% | 20% / 17% / 0% | 13% / 67% / 0% | 7% / **83%** / 0% |
| S | books: needle / 2-hop | 67% / 0% | 67% / 0% | **100% / 100%** | **100% / 100%** |
| M | contracts: CUAD / needle / 2-hop | 44% / 42% / 0% | 13% / 17% / 0% | 13% / **83%** / 25% | 16% / **83% / 100%** |
| M | books: needle / 2-hop | 83% / 25% | 42% / 0% | **100% / 100%** | **100% / 100%** |
| L | contracts (8,619 pages): CUAD / needle / 2-hop | 37% / 39% / 0% | 14% / 11% / 0% | 14% / 39% / 67% | 10% / 44% / **100%** |
| L | books (9,988 pages): needle / 2-hop | 61% / 17% | 39% / 0% | out of memory | **100% / 100%** |

1. **At the same evidence budget (~200–300 tokens), our pipeline finds planted facts far
   more often than plain search:**
   - contracts: 67–83% vs 17%
   - books: 100% vs 42–67%
   - two-hop facts: up to 100% vs 0–25%

   Plain search catches up only by sending ~5× more text.
2. **Real contract clauses (CUAD) are the weak spot for every system at a small budget**
   (7–20%).
   - **Diagnosis:** questions name the contract by its title ("…the Content License
     Agreement of WebmdHealthCorp…"). Our pipeline reaches the right contract (4 of 6 pack
     items from it) but the wrong clause: the title words dominate the search.
   - **Response:** this motivated routing in step 3.
3. **Cost at 1,000 pages:** build 1–1.5 min, peak memory ≤ 1 GB. Retrieval latency: rule
   ~0.1–0.15 s, GNN ~0.3 s, plain search ~0.01 s.
4. **At ~10,000 pages (L):**
   - **Books: the GNN pipeline still finds every planted fact and every two-hop fact**
     from ~285 tokens. Plain search drops to 39% (equal budget) / 61% (5× budget).
   - **Contracts (510 contracts of similar form):** the pipelines fall from 83% (M) to
     39–44% on planted facts. They pick the right *kind* of clause from the wrong
     contract; the distractors are the same clause type in other contracts. Two-hop facts
     hold up: GNN 100%, rule 67%. The chapter-worker design targets this: routing names
     the right contract 95% of the time at this size (step 4 below).
   - **Cost on this 8 GB laptop:**

     | System | Build | Memory | Per question |
     |---|---|---|---|
     | rule (contracts) | 36 min | 3.5 GB | 1.2 s |
     | GNN (contracts) | 38 min | 3.1 GB | 6.8 s |
     | GNN (books) | 17 min | 3.5 GB | 9.6 s |
     | plain search | 5–21 min | ~1.1 GB | 0.012 s |

     **The rule pipeline cannot index the 9,988-page book library in 8 GB:** it was
     killed twice, once running alone.
   - **Run notes:**
     - The first L contract runs used a document whose structure was broken by a stray
       "# # #" line in one CUAD contract. That line became a top-level heading: 5
       chapters instead of ~550. Fixed and rerun; S and M documents are byte-identical.
     - The laptop slept overnight (10:39 pm–10:44 am) and froze one run. Long runs now
       use `caffeinate`.

### Answers at ~100 pages (S): pipeline vs plain search vs the whole document

| Contracts, 130 pages | Whole document (~87k tok) | Plain search k=5 | Rule pipeline | **Chapter workers** (step 3) |
|---|---|---|---|---|
| CUAD answers correct (judge score) | 33% (0.40) | 33% (0.47) | 13% (0.13) | **47% (0.60)** |
| CUAD wrongly refused | 53% | 40% | 87% | **13%** |
| CUAD absent: correctly refused | 87% | 67% | **93%** | 67% |
| Planted facts correct | 50% | 50% | 50% | **67%** |
| Two-hop correct | 0% | 0% | 0% | 0% |
| Never-planted: correctly refused | 75% | 75% | **100%** | 50% |
| Evidence sent / model calls per question | 87k tokens / 1 | 1.4k / 1 | 0.3k / 1 | **0.15k / 4.4** |

| Books, 119 pages | Whole document (~81k tok) | Plain search k=5 | Rule pipeline |
|---|---|---|---|
| NarrativeQA correct (judge score) | **37% (0.42)** | 20% (0.22) | 3% (0.03) |
| NarrativeQA wrongly refused | 33% | 63% | 83% |
| Planted facts correct | 33% | 50% | **100%** |
| Never-planted: correctly refused | 100% | 100% | 100% |

1. **Sending the whole 120-page document did not make a small model reliable.**
   `gemini-3.5-flash-lite` missed or wrongly refused half the planted facts. One answer
   was simply wrong: "underneath one of the thwarts" for the key in the violet tin.
   - **Caveat:** this is a small, fast model. A stronger long-context model is the real
     baseline (Claude run, pending budget).
2. **Book questions are mostly about the whole story** ("Who is Norman's father?"). A
   ~270-token pack refuses 83% of them. These need the whole-document route (R2, step 4);
   no system is good yet (best 37%).
3. **Planted facts in a book: our pipeline answers all of them correctly** from ~270
   tokens; the whole document gets 33%.
4. **Two-hop facts were not answered by any system.** The pipelines often retrieve the
   first part (the "copper scarf"), but the second hop needs a new search. This is fixed
   for the workers below.

### Step 3: chapter workers + central board (first measurement, contracts S)

`workers/`: routing, board, worker, orchestrator (see REQUIREMENTS v4-3).

**Run 1:** 42 questions, 4.4 calls and ~150 evidence tokens per question.
- **Routing named a contract in 83% of questions;** 2.2 chapters searched on average.
- **Best system on CUAD clauses:**
  - evidence found 67% (rule pipeline 13%)
  - judged correct 47% (whole document 33%)
  - wrongly refused 13% (whole document 53%)

**Weaknesses found in run 1:**
- **Two-hop across contracts: 0%.** A worker could follow only chapter ids or titles, not
  "rate card RC-8985".
  - **Fixed:** follow now takes free text, which the board looks up across the whole
    document (tested).
- **Absent clauses / never-planted facts are refused less often** (67% / 50%). Workers
  posted on-topic sentences that do not state the answer.
  - **Fixed:** the worker prompt now requires sentences that state the answer.

**Run 2 (after both fixes): 8 of 42 so far, then the free daily quota ran out.**
- The 8 CUAD questions: evidence found 87.5%, judge score 0.75, no wrong refusals.
- Too few to conclude. The run resumes on the same model when the quota resets.

---

## Build step 4 (v4-4) — Relevance gate in front of the workers (2026-10-23, in progress)

**Code:** `workers/gate.py`, plus scope / evidence limits / widening / whole-document notes in
`workers/routing.py` and `workers/orchestrator.py`; tests in `tests/test_workers.py`.

**Gate:**
- **Signals:**
  - lexical: each chapter's *best* sentence (BM25 on the question's content words)
  - dense: MiniLM best passage, optional
  - named units from routing
  - TOC screen: one model call over chapter titles and summaries
- **Rounds:**
  - **round 1:** the named unit, or else the union of every signal's top k (any signal
    can open a chapter)
  - **round 2:** the next k
  - **round 3:** the question rewritten as document-style keywords; that round's workers
    search with the rewritten words
- **Scope** sets k and the evidence size:

  | Scope | k | Evidence |
  |---|---|---|
  | single | 3 | 600 tokens |
  | multi | 4 | 1,000 tokens |
  | global | 8 | 2,000 tokens, plus workers' partial answers as notes (not evidence) |

### Gate readings (no model; TOC screen off)

For each question with gold evidence: the chapter that holds the answer (looked up inside
the unit the question is about), and whether the gate opens it.

| Document | Chapters | Questions (with gold) | Routing correct | Answer chapter in round 1 | By round 2 | Opened: round 1 / by round 2 |
|---|---|---|---|---|---|---|
| contracts S (130 pages) | 8 | 23 | 100% | 91% (CUAD and single needles 100%) | **100%** | 1.5 / 4.5 |
| contracts M (977 pages) | 55 | 71 | 100% | 94% (CUAD and single needles 100%) | **98.6%** | 1.3 / 4.3 |
| books S (119 pages, 1 book) | 20 | 8 | — (book not a unit) | 87.5% (single needles 100%, two-hop 50%)* | 87.5% | 3 / 6 |
| books M (1,001 pages) | 155 | 16 | 100% | 81% | 87.5% | 2.8 / 5.8 |
| contracts L (8,619 pages) | 554 | 119 | **95%** | 95%* | 97.5%* | 1.2 / 4.2 |
| books L (9,988 pages) | 1,669 | 24 | 100% | 83% (single needles 100%) | 87.5% | 2.9 / 5.9 |

\* **Corrected 2026-11-12.** First recorded as 100% / 100%: in a one-book library the
title-page chapter carries the book's name, the "unit" lookup restricted the gold search
to that placeholder, found nothing, and an empty gold set counted as "found". The lookup
now ignores non-body chapters and falls back to the whole document; questions whose gold
text cannot be located are no longer counted. Other rows were unaffected (their units
are real contracts or parts).

\* About 6% of L CUAD spans could not be located inside their contract (the span crosses a
paragraph break). Those questions count as found, so the conservative round-1 figure is
the routing rate, 95%.

**The gate scales:**
- In the 8,619-page room, the gate opens ~1.2 of 554 chapters in round 1 and reaches the
  answer's contract 95% of the time.
- In the 9,988-page library: ~3 of 1,669 chapters, and every single planted fact's chapter
  is opened in round 1.

That is the step the pipelines above are missing at L.

- **Two-hop questions are not in round 1 by design:** the second part is in another chapter.
  - contracts: 75–100% by round 2
  - books M: 50%; the rest needs a worker's follow or search

**Bugs the gate readings found (all fixed, with tests):**
1. **One-book library routed every question to its title-page chapter.** That chapter is
   titled with the book's name. Placeholder chapters are no longer routable.
2. **Contracts split into slices ("… (1/4)") were never routed.** The slice numbers became
   "distinctive" title words. Slice markers are now ignored.
3. **The rewrite round opened the right chapter, but its worker searched with the old
   words.**
4. **Measurement bug:** boilerplate clauses ("governed by the laws of New York") were
   attributed to the wrong contract. The answer chapter is now looked up inside the
   question's own unit.

**Still to measure (needs model calls):**
- the TOC screen
- gated vs ungated workers
- the whole-document route on NarrativeQA
- the five-system comparison

---

# Complete or Silent — P1 readings (2026-11-12)

Code: `src/gnn_pruner/certify/` (qualifiers, reverse, canaries, plan); tests in
`tests/test_certify.py` (9) and `tests/test_workers.py` (+3). Full suite: 223 passed.

## U2 / U3: qualifier graph and reverse index on the real 130-page data room

| | Links | Resolved | qualifies | overrides | amends | precedence | → section | → term | → "foregoing" |
|---|---|---|---|---|---|---|---|---|---|
| first version | 302 | 255 | 215 | 72 | 14 | 1 | 37 | 197 | 22 |
| after the term filter (U2.6) | 204 | 76 | 152 | 44 | 7 | 1 | 37 | 18 | 22 |

- Build time 0.06 s. Section links checked by hand on a sample look right ("Subject to
  Section 8.2" → 8.2; "Section 7.2.2(a)" → 7.2.2; "Article 10" → 10.x).
- **Bug found and fixed:** party names ("BLI", "Ginkgo") and "Agreement" are defined
  terms too and were linked everywhere (197 term links, 34 to "agreement"). Terms used in
  more than 5% of a contract's sentences, and a generic list, are now dropped: 18 term
  links remain, all real ("receiving party", "headstart period", "confidential information").
- "Resolved" fell because unresolved cues are kept on purpose: "shall not apply …" without
  a section number is exactly what the whole-document sweep (Q4) must catch.
- Link precision on a hand-checked sample of 100 (U2.7): not yet measured.

## U4: canary calibration — stated vs measured miss rate (early RQ1 probe)

Lexical gate only, no routing, whole document; 300 mutation canaries per document;
real questions = every benchmark question with gold evidence (CUAD clauses, planted
single and two-hop facts). "Miss" = the answer's chapter is not in the top k.

**1. Naive canaries lie.** Queried with a random half of their own words, canaries rank
first almost always, so the calibration promised k = 1–2 chapters at 98%:

| Room | α | k | canary miss | **real miss** |
|---|---|---|---|---|
| 130 pages (8 chapters) | 0.98 | 2 | 0.0% | **60.9%** |

Rank distributions explain it (p50 / p75 / p90 / p98 of the answer chapter's rank):

| 977-page room, 55 chapters | p50 | p75 | p90 | p98 |
|---|---|---|---|---|
| real questions | 16 | 30 | 41 | 47 |
| canary, random 50% of its words | 1 | 1 | 1 | 2 |
| canary, its 10% most common words | 12 | 23 | 38 | 47 |

Real questions share few rare words with their answer (query–answer word overlap: median
0.33). **Design decision:** a canary is queried with its lowest-IDF 10% of content words
(`QUERY_SHARE = 0.1`). Chosen on the 130-page room; it reproduced the real rank shape on
the 977-page room it was not tuned on.

**2. With common-word canaries (current code; canary planted in its own unit, its source
sentence blanked in the hidden copy):**

| Room | α | k (of N) | canary miss (upper) | **real miss** (upper, n) | by kind |
|---|---|---|---|---|---|
| 130 pp, N = 8 | 0.90 | 6 | 5.3% (8.0) | **8.7%** (24.9, 23) | CUAD 2/15, needles 0/8 |
| | 0.95 | 7 | 1.7% (3.9) | 4.3% (19.0) | CUAD 1/15 |
| | 0.98 | 7 | 1.7% (3.9) | 4.3% (19.0) | CUAD 1/15 |
| 977 pp, N = 55 | 0.90 | 30 | 9.3% (12.5) | **25.4%** (35.2, 71) | CUAD 18/55, needles 0/16 |
| | 0.95 | 38 | 4.7% (7.2) | 16.9% (25.9) | CUAD 12/55 |
| | 0.98 | 46 | 0.7% (2.2) | 4.2% (10.6) | CUAD 3/55 |

- **Planted facts: 0 misses** at every α on both rooms (24 questions).
- **Real CUAD questions are missed more often than stated** on the larger room. They ask
  by clause *type* ("Does X have a non-compete clause?"), and those words occur in many
  contracts, so a lexical rank cannot single out the right contract. In the workers these
  questions name their contract (basis "named": all its chapters are opened; coverage =
  routing, measured 95–100%), so the room table governs only unnamed questions.
- **Blanking the source sentence** in the hidden copy changed nothing (identical numbers);
  kept because it is the right simulation of a real answer, which has no twin.
- **Cost of certainty without routing** is high: 46 of 55 chapters at 98% on the 977-page
  room. The unit table (a named book or contract ranked within itself) asks for 2–4
  chapters. These are the first points of the RQ4 cost curve.
- Sample sizes are small (23 and 71 real questions); upper bounds are wide. The P3 test
  needs the trap benchmark (≥ 300 questions per size).

**119-page library (one book, 20 chapters; 8 questions with gold, all planted):**

| α | k (of 20) | canary miss | real miss |
|---|---|---|---|
| 0.90 | 15 | 7.3% | 0 / 8 |
| 0.95 | 16 | 4.7% | 0 / 8 |
| 0.98 | 19 | 1.0% | 0 / 8 |

- A single book counts as one unit, so its unit table equals its room table (a question
  that names the book opens the whole book's ranking).
- Only the "clause" class had enough canaries (296 of 300): novels hold few numbers.
  **Design decision:** a class needs ≥ 30 canaries (`MIN_CLASS_N`) for its own k;
  otherwise the "all" table is used. Before this rule the fact and qualifier classes
  fell to "open everything" (k = 20) from a handful of ranks.
- **Measurement bug found on the way:** see the corrected books-S gate row above.

**What this means for the paper (RQ1, honest):** the absence certificate holds for
planted facts and is optimistic for clause-type questions when no unit is named. Either
the canary class for "clause-type" questions must be queried the way such questions are
asked (the clause-type words, not the sentence's own words), or the certificate for
unnamed clause-type questions must be stated with the measured gap. Open for P3.

## Q1 / Q2: question class and calibrated reading plan

- `classify_question`: fact / clause / list / existence by patterns (tests cover the four).
- `reading_plan`: basis named (open the whole unit) / unit (unit table) / room (room table)
  / uncalibrated; `ask(..., calibration=, alpha=)` uses the calibrated k for the gate and
  records `certified`, the highest α whose k chapters were all opened. Unit-tested with
  the scripted model; not yet run on the benchmark with a real model.

## Q4: closure expansion on the real 130-page data room (no model)

Seeds: the gold clause of each of the 15 answerable CUAD questions. Forward links
(definitions, references), reverse index, sweep candidates (unverified without a model).

| Version | Items / q | Forward | Reverse | Swept | Rounds | Fixpoint | Time / q |
|---|---|---|---|---|---|---|---|
| first | 14.3 | 12.5 | 1.8 | 84.9 | 2.8 | 4 / 15 | 6.9 ms |
| **current** | **2.7** | 2.4 | 0.3 | 9.0 | 2.0 | **15 / 15** | 0.5 ms |

- **Bug found and fixed (design):** the tree's `uses_definition` links cover every defined
  term, including party names and "Agreement", and every accepted definition cascaded its
  own links: 11 of 15 closures hit the 24-item budget on definitions alone. Now only the
  answer, qualifiers, overrides and amendments cascade; a definition joins only if it
  defines a kept term (the qualifier graph's filter); the sweep has one budget of 12
  candidates per closure. Items per question: 0–9.
- A term counts as name-like when it occurs in more than max(5% of the contract's
  sentences, 10 sentences); the pure share rule dropped "Fees" from a 12-sentence test room.
- Reverse links are rare on real CUAD clauses (0.3 per question): most CUAD gold spans are
  not the target of a "notwithstanding Section X" elsewhere. The trap benchmark (P2)
  plants exactly those; this reading says the structural part finds them when present
  (unit tests: far override, amendment through its title, subject-to, precedence) and adds
  little noise when absent.
- The sweep needs the qualifier worker: 9 candidates per question on average, one model
  call. Without a model the closure reports them as unverified, and a certificate built on
  it must say so. Not yet run with a real model.

## Q5–Q7 and the end-to-end path (built, scripted-model tests only)

- **Q5 precedence** (`certify/precedence.py`): later amendment supersedes its target
  (kept, marked); an explicit "schedules prevail" clause ranks schedule items over the
  body; an amendment dated earlier than what it amends is not applied and is flagged as a
  conflict; two amendments governing one clause are a conflict.
- **Q6 certificates** (`certify/certificate.py`): closure / absence / uncertified, with
  the share of chapters and tokens read; reasons for "uncertified": budget, unjudged sweep
  candidates, conflicts, unsupported answer sentences, fewer chapters than the
  calibration asks, no calibration.
- **Q7 verifier** (`certify/verify.py`): one independent call per answer; a label outside
  the pack counts as unsupported.
- **`ask(..., certify=True)`**: found → closure → precedence → answer (pack widened to the
  multi budget, roles note in the prompt) → verifier → closure certificate; not found →
  absence certificate. On the BETA example with a scripted model the certificate is
  COMPLETE with the override, amendment, carve-out, definition and precedence clause in
  the pack and the 12-month cap marked superseded.
- **Bug found on the way:** the two-hop bridge (F10) had already pulled in the override
  sentence by its phrase "shall not exceed", so the closure skipped it and the pack
  labelled it a bridge. A bridge now takes the closure's structural relation.
- **Not yet measured with a real model:** the qualifier worker's verdicts, the verifier,
  and the certificate rates on the benchmark.

## F10 / F11 re-measured: chapter workers on contracts S (42 questions, gemini-3.5-flash-lite, gated)

Run `long_contracts_S_workers_..._f10f11` vs the earlier gated run, same questions and model.

| Kind | n | Calls / q (new / old) | Evidence coverage (new / old) | Correct (new / old) | Refused (new / old) |
|---|---|---|---|---|---|
| CUAD clause | 15 | 5.9 / 5.7 | 0.64 / 0.69 | 9 / 10 | 3 / 3 (same 3 wrong refusals) |
| CUAD absent | 15 | **8.5 / 10.0** | — | — | 10 / 10 |
| planted single | 6 | 3.0 / 3.0 | 0.79 / 0.79 | 5 / 5 | 1 / 1 |
| planted two-hop | 2 | 8.5 / 8.0 | **0.83 / 0.17** | **2 / 0** | 0 / 0 |
| never planted | 4 | 6.8 / 7.0 | — | — | 3 / 3 |
| **all** | 42 | **6.6 / 7.1** | | | |

- **F10:** the bridge step (no model call) puts the second part of a two-part fact into
  the pack by its rare phrase ("rate card RC-8985"); both two-hop questions are now
  judged correct. n = 2: the trap benchmark (P2) is where this gets a real sample.
- **F11:** widening rounds open only chapters with word evidence (≥ 50% of the best
  chapter's score) and give their workers 2 steps: 15% fewer calls on absent questions,
  6% overall, with no change in what is refused or found. The remaining cost of a
  "not found" (8.5 calls) is the widening itself, which the absence certificate turns
  into a stated guarantee instead of a hope.
- Unchanged weaknesses: 5 of 15 absent CUAD questions still get an answer from an
  on-topic but non-stating clause (the same 5 as before); the judge prompt and the
  worker's "must state the answer" rule are the next levers.

## P2: the trap benchmark (`certify/traps.py`, built 2026-11-14)

Built on the same CUAD rooms (seed 7) as its own family `traps`; original rooms unchanged.

| Room | Documents | Pages | Questions | Per kind |
|---|---|---|---|---|
| traps S | 33 | 139 | 60 | 12 × far exception, override chain, unlinked, precedence, existence |
| traps M | 215 | 1,034 | 360 | 72 × each |

Each trap plants a numbered base clause (a named company's liability cap) plus a
look-alike cap in another contract. Gold per case: required sentences, superseded
sentences, the expected certificate (closure / absence). Templated text (several
wordings); mutating real clauses is still to do.

### Structural trap recall (no model): is every required provision reached?

| Kind | traps S (n) | traps M (n) | Reached by |
|---|---|---|---|
| far exception, same contract | 4 / 4 | 24 / 24 | reverse index |
| far exception, in a schedule | 4 / 4 | 24 / 24 | reverse index |
| far exception, side letter (another document) | 4 / 4 | 24 / 24 | reverse index, contract named in the text |
| override chain (amendment of an amendment) | 12 / 12 | 71 / 72 | reverse index; old caps marked superseded in all |
| unlinked carve-out (no section number) | 12 / 12 offered | 72 / 72 offered | sweep → qualifier worker |
| precedence (schedule restates the cap) | 12 / 12 offered | 72 / 72 offered | sweep (restatement) → qualifier worker |

Closure time: 30 ms / question (S), 222 ms (M); graph + reverse index 1.0 s / 17.5 s.

**Bugs the traps found (all fixed, with tests):**
1. *Amendment of an amendment* resolved "Section 42 of the MSA, as amended by Amendment
   No. 1" into Amendment No. 1 (its title shares more words with the sentence). Contract
   names now never resolve to an amending document; amendments bind to the contract.
2. *Two amendments of one clause* were reported as a conflict. The later one (by
   amendment number, else by date) now governs and the earlier one is superseded.
3. *"Nothing in the foregoing limitations shall limit …"* was not a qualifier cue.
4. *A schedule restating a cap* has no cue word at all, so the cue sweep missed it. The
   sweep now also offers restatements: the contract's highest-BM25 sentences for the
   seed's words. The qualifier worker gained the verdict "restates"; a restatement
   governs only through an explicit precedence clause.

## First real-model certificates (gemini-3.1-flash-lite, contracts S, 8 questions)

| Question | Calls | Certificate | Closure items | Sweep judged → kept |
|---|---|---|---|---|
| 4 CUAD clauses | 4–19 | 3 COMPLETE, 1 UNCERTIFIED | 0–16 | 43 → 1 |
| 2 absent CUAD clauses | 8, 23 | 2 NOT FOUND at 98% (named contract opened whole) | — | — |
| 2 planted facts | 4, 20 | 2 COMPLETE | 0–4 | 24 → 0 |

- The one "uncertified": the verifier found the answer's date was not in the quoted
  text — the certificate did its job.
- The qualifier worker rejected 66 of 67 sweep candidates on real CUAD text: real
  contracts in this room have few carve-outs of the clauses asked about.
- Wording bug fixed: an absence certificate for a question naming its contract printed
  "0 canaries"; it now says the named unit was opened whole and the risk is routing.

## RQ2 first reading: certified vs plain workers on the trap benchmark (traps S)

30 trap questions (6 per kind, balanced by `--limit 30`), the 139-page trap room, the same
model for both systems (gemini-3.5-flash-lite), gated workers; "certified" =
`--system workers_certify` (calibration, closure, precedence, verifier, certificates).
Scorer: every amount of the reference in the answer (plus "fraud" for carve-outs);
existence: yes / no as expected.

| Kind | Required provisions in evidence (plain → certified) | Correct (plain → certified) | Calls / q | Certificate right |
|---|---|---|---|---|
| far exception | 0.00 → **1.00** | 0 / 6 → **6 / 6** | 4.5 → 6.8 | 6 / 6 |
| override chain | 0.67 → **1.00** | 4 / 6 → **6 / 6** | 4.3 → 7.0 | 6 / 6 |
| precedence | 0.17 → **1.00** | 1 / 6 → **6 / 6** | 5.3 → 8.0 | 6 / 6 |
| unlinked carve-out | 0.00 → **1.00** | 0 / 6 → **6 / 6** | 5.0 → 7.2 | 6 / 6 |
| existence (3 yes, 3 no) | — | 6 / 6 → 6 / 6 | 7.8 → 10.2 | 6 / 6 (3 closure, 3 absence) |
| **all** | | **11 / 30 → 30 / 30** | **5.4 → 7.8** | **30 / 30** |

- Plain workers find the base clause and stop: the cap they report is wrong or incomplete
  in 19 of 30 cases. The certified system reaches every required provision and states
  the result with the carve-outs and the superseded amounts.
- Cost of certainty here: +2.4 calls per question (+44%).
- **Caveats (must be in the paper):** n = 30, one small room, one model, templated trap
  text (mutation of real clauses still to do), and the traps were designed together with
  the system. The M room (360 traps, 1,034 pages), a second model, and the external
  baselines (B1 long-context, B3 Chain-of-Agents) are the next runs.

**Benchmark bugs found by this run (fixed; documents byte-identical after the fix):**
1. Every "no" existence question asked about the same contract, and "yes" traps had
   planted a change-of-control clause into it: the system's "yes" was right, the gold
   was wrong. "No" questions now pick a (contract, clause type) pair where the contract
   truly lacks the clause (no CUAD label, nothing planted, none of the key words).
2. The automatic scorer demanded the reference's exact wording ("no limit for fraud");
   correct certified answers scored 0. It now checks the amounts and the carve-out.

## U4.7 + RQ1: calibration at 10,000 pages, and type canaries (2026-11-14, no model)

**Fact canaries on the 8,619-page room** (554 chapters, 300 canaries, 119 real questions;
18 min, 772 MB peak):

| α | k | canary miss | real miss (upper) | by kind |
|---|---|---|---|---|
| 0.90 | 268 | 9.7% | 23.5% (30.8) | CUAD 27/95, planted 1/24 |
| 0.95 | 305 | 4.7% | 17.6% (24.4) | CUAD 21/95, planted 0/24 |
| 0.98 | 393 | 1.7% | 7.6% (12.8) | CUAD 9/95, planted 0/24 |

Same pattern as at 977 pages: planted facts are covered, clause-type questions are not.

**Type canaries** (new, `canaries.type_canaries`, `cuad_type_library`): a real CUAD clause
of category X from a contract outside the room is planted in a random chapter and looked
up with the words of the category-X question (unit name removed). Lexical gate, no
routing, whole room; real = the benchmark's CUAD questions.

| Room | Ranks p50/p75/p90/p98: type canaries vs real CUAD | α 0.90 real miss (stated 10%) | α 0.95 (5%) | α 0.98 (2%) |
|---|---|---|---|---|
| 130 pp (8 ch, n = 15) | 4/6/7/8 vs 4/5/6/7 | 6.7% | 0% | 0% |
| 977 pp (55 ch, n = 55) | 21/32/43/51 vs 16/35/43/47 | **10.9%** | **1.8%** | **1.8%** |
| 8,619 pp (554 ch, n = 89) | 196/324/411/492 vs 155/271/388/506 | **5.6%** (ub 11.4) | **3.4%** (ub 8.5) | **2.2%** (ub 6.9) |

- **Stated and measured miss rates now agree within the confidence intervals at every
  size.** With fact canaries the 977-page room missed 25% / 17% / 4%; with type canaries
  11% / 1.8% / 1.8%. This is the first positive RQ1 evidence for clause-type questions.
- **Caveat (leak at L):** the 8,619-page room holds every CUAD contract, so its "outside"
  library fell back to contracts that are in the room (canaries with twins). The 130 and
  977-page numbers are clean. Paper protocol: hold out a fixed set of contracts from all
  rooms as the type library.
- **Cost:** a room-wide certificate for an unnamed clause-type question needs most of the
  room (493 of 554 chapters at 98%): lexical ranking cannot single out one contract by
  clause type. When the question names its contract (all CUAD questions do), the named
  unit is opened whole and the certificate costs one contract.
- Type canaries are now part of `calibrate(..., type_library=)`; clause and existence
  questions use the "type" table when present (`CANARY_CLASS` preference chains); the
  certified benchmark system builds the library for contract rooms. Calibration cache
  versioned (`.calibration.v2.json`).
