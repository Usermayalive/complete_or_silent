# Phase 0: Use Case Definition

## Domain

**Commercial contracts** (services agreements, SaaS subscriptions, NDAs), in English,
with numbered sections and a definitions clause.

Why contracts first:
- Strong explicit structure (numbered sections, a definitions clause), so the graph is meaningful.
- Meaning is distributed: a clause often depends on a definition and on another
  clause it cites ("Except as set out in Section 4.2"). This is exactly where
  flat chunk retrieval fails and graph expansion and pruning pay off.
- Answers are verifiable against exact clause text, so evaluation is cheap.

## Users

Lawyers, contract managers, procurement, and sales operations staff reviewing a single
contract. They need the answer *and* the clauses it rests on. A wrong answer that
reads cleanly is worse than "not stated in the document".

## In scope (v1)

| Question type | Example | Planner label |
|---|---|---|
| Definition lookup | "What does Misconduct mean?" | `definition_lookup` |
| Fact lookup | "When must invoices be paid?" | `fact_lookup` |
| Consequence / cross-reference | "What happens if the Supplier commits misconduct?" | `cross_reference_reasoning` |
| Clause comparison | "Compare sections 4.1 and 4.2" | `clause_comparison` |
| Section summary | "Summarize the liability terms" | `summary` |
| Contradiction detection | "Is there a conflict in the incident response time?" | `contradiction_detection` |
| Unanswerable | "What law governs this agreement?" | must abstain |

- One document per query.
- Input formats: Markdown, plain text, PDF with a text layer, DOCX, and HTML. Scanned PDFs need Tesseract installed.

## Out of scope (v1)

- Questions across several contracts, and portfolio search.
- Legal advice or judgement ("is this clause enforceable?"). The system reports what
  the text says, not what it means legally.
- Handwriting and non-English contracts.
- Redlining, drafting, and clause generation.

## Evaluation

- Dataset: [`eval/contracts_v1.jsonl`](../eval/contracts_v1.jsonl) has 35 questions over 2 contracts
  ([sample_contract.md](../examples/sample_contract.md), [saas_agreement.md](../examples/saas_agreement.md)).
  It includes 4 unanswerable questions, 1 planted contradiction, and 8 paraphrase questions (`para-*`) that share almost no words with their answer clause.
- Each case has a question, a gold query type, gold evidence (verbatim snippets that must
  appear in the evidence pack), and a reference answer.
- Run: `PYTHONPATH=src python -m gnn_pruner.evaluation eval/contracts_v1.jsonl --verbose`

| Metric | Definition | v1 target | Baseline (2026-10-08) | After Phase 2 |
|---|---|---|---|---|
| planner_accuracy | Predicted query type equals the gold type | ≥ 0.90 | 1.00 | 1.00 |
| evidence_recall | Share of gold snippets that appear in the evidence pack | ≥ 0.95 | 0.977 | 1.00 |
| full_recall_rate | Share of questions where every gold snippet was retrieved | ≥ 0.90 | 0.955 | 1.00 |
| hit_at_1 | First evidence item contains a gold snippet | ≥ 0.80 | 0.864 | 0.913 |
| evidence_precision | Share of evidence items that contain a gold snippet | ≥ 0.50 | 0.406 | 0.446 |
| mean_pack_size | Evidence items per answerable question | ≤ 6 | 4.27 | 4.09 |
| abstention_rate | Unanswerable questions that return an empty pack | ≥ 0.75 | 0.25 | 0.25 |

### Phase 3: embedder comparison (35 questions, min-max scaling, no path prefix)

| Metric | TF-IDF | all-MiniLM-L6-v2 | all-mpnet-base-v2 |
|---|---|---|---|
| evidence_recall | 0.79 | **1.00** | 0.984 |
| hit_at_1 | 0.742 | **0.871** | **0.871** |
| embedding_recall_at_5 (vector search alone) | 0.71 | **0.952** | **0.952** |
| evidence_precision | **0.341** | 0.289 | 0.289 |
| mean_pack_size | **3.29** | 6.13 | 6.03 |
| abstention_rate | **0.25** | 0.00 | 0.00 |
| recall on the 8 paraphrase questions | 0.25 | **1.00** | **1.00** |

Findings:
- TF-IDF fails on paraphrases ("Can they charge us more next year?"). It finds only 2 of 8, so a dense model is required for real users. MiniLM matches mpnet and is about 5x smaller. It is the default whenever `sentence-transformers` is installed (`--embedder auto` in the CLIs).
- Dense models produce larger, less precise packs and never abstain. Every passage has a moderate cosine similarity, so more of them pass the relative pruning cutoff. This is a scoring and pruning calibration problem (Phases 7–8), not a retrieval one.
- Min-max scaling of similarities within each document raised the top-result accuracy of both dense models (MiniLM 0.71 → 0.87). With TF-IDF it lost one keyword question (msa-06). It is now the default.
- Prefixing each node with its section path improved TF-IDF's vector-search recall (0.71 → 0.74), but reduced it for both dense models (0.95 → 0.90 / 0.92). It is off by default (`contextual=True` to enable).

### Phase 4: query planner

The planner is measured on its own with two sets. The development set
([eval/planner_v1.jsonl](../eval/planner_v1.jsonl), 46 questions) was written first. It was
used to design the rules and supplies the example questions for the semantic fallback. The
held-out set ([eval/planner_holdout_v1.jsonl](../eval/planner_holdout_v1.jsonl), 23 questions)
was written afterwards and scored without changing any rules.

Run: `python -m gnn_pruner.evaluation eval/planner_holdout_v1.jsonl --planner --embedder auto --verbose`

| Metric | Old rules, dev set | Dev set | Held-out (TF-IDF: rules only) | Held-out (MiniLM: rules + semantic fallback) |
|---|---|---|---|---|
| type_accuracy | 0.717 | 1.00 | 0.565 | **0.739** |
| focus_accuracy (definition term) | 0.375 | 1.00 | 0.25 | **1.00** |
| section_recall | n/a | 1.00 | 0.90 | 0.80 |
| sides_accuracy (comparison) | n/a | 1.00 | 0.25 | 0.25 |

- The 1.00 scores on the dev set only show that the rules fit the questions they were written
  for. The held-out column is the honest number.
- The rules are precise but brittle. They fail on unseen wording ("what would follow from",
  "rundown", "disagree", "cheaper than"). When no rule fires, the semantic fallback compares
  the question with the dev set's example questions, but only with a dense embedder.
- The fallback's margins were chosen on the Phase 0 set, which played no part in building the
  examples. Its fact questions reach +0.246 toward other types, so non-definition types need a
  margin of 0.25. Definitions use 0.15 because they must also name a defined term. At 0.05,
  the fallback relabelled 10 of 31 fact questions in the Phase 0 set.
- Similarity to example questions separates comparison, summary and contradiction questions
  from fact questions poorly. Almost all of the fallback's gain is on definitions. Comparison
  side extraction on unseen wording (0.25) is the weakest part.
- End to end (35-question set, MiniLM), planning plus plural stemming moved hit_at_1 from
  0.871 to 0.935 and precision from 0.289 to 0.315. Recall stayed at 1.00.

### Phase 5: hybrid retrieval

Six ranked sources are fused with weighted reciprocal rank fusion (`retrieval.py`):

| Source | What it ranks | RRF weight |
|---|---|---|
| lexical | BM25 for the question and each planner sub-query | 1.0 |
| phrase | quoted phrases and shared word pairs | 1.0 |
| semantic | vector search for the question and each sub-query | 1.0 |
| structural | content of the 3 sections whose title or chapter path best matches | 0.5 |
| plan | the planner's resolved sections, clauses and definitions (also pinned) | 1.0 |
| graph | parent paragraphs and outgoing references of the top 10 fused nodes | 0.5 |

The top 5 of each direct source (lexical, phrase and semantic) always enter the candidate
set. The rest of the budget (`plan.max_candidates`, 30–50) is filled in fused order. Every
candidate records which sources found it.

Run: `python -m gnn_pruner.evaluation eval/contracts_scale_v1.jsonl --embedder auto --sources`

Each method's recall on its own vs the candidate set (MiniLM, 36 answerable questions):

| Method | Small set recall@10 | Scale set recall@5 | Scale set recall@10 |
|---|---|---|---|
| lexical | 0.806 | 0.792 | 0.806 |
| phrase | 0.514 | 0.514 | 0.514 |
| semantic | **1.00** | 0.917 | 0.944 |
| structural | 0.472 | 0.472 | 0.472 |
| plan | 0.514 | 0.514 | 0.514 |
| graph | 0.00 | 0.069 | 0.069 |
| fused (first k) | **1.00** | 0.917 | **0.986** |
| **candidate set** (31 nodes on average) | **1.00** | | **1.00** |

With TF-IDF, candidate recall on the scale set rose from 0.806 to 0.833, above the best
single method (lexical, 0.806). Downstream metrics (evidence recall, hit@1, precision) are
unchanged on both sets with both embedders.

- The acceptance rule is met: the candidate set recalls at least as much as every single
  method, and its size stays bounded (`tests/test_retrieval.py` checks this).
- **The fused order does not always beat the best method at small k.** On the scale set with
  TF-IDF, fused recall@5 is 0.764, while lexical alone gets 0.792. RRF ranks a passage that
  only one source finds below passages that several sources find. The quota keeps such
  passages in the candidate set. Ordering them is the scorer's job (Phase 7).
- Structural and graph sources add little recall on these sets. Their answers were already
  found by text match, and expansion (Phase 6) follows references anyway. They are kept at
  half weight because they are cheap, and they give provenance that a later learned ranker
  can use.
- Rejected: adding section-summary similarity to the structural vote. It duplicated the
  semantic source and dropped scale-set hit@1 from 0.833 to 0.778 with MiniLM.
- Fixed: the planner's pinned nodes used to be listed ahead of the fused order, which made
  the fused ranking look worse than it was (scale set msa-03: every method had the answer in
  its top 5, but the fused top 5 did not). Pins are now a ranked source inside the fusion.

### Phase 6: graph expansion

Expansion is best-first. Each neighbour's priority is its anchor's fused score × a link
prior × 0.5 per hop:

| Link | Prior |
|---|---|
| parent section | 1.0 |
| paragraph around a sentence | 0.95 |
| definition used | 0.9 |
| cited clause | 0.85 |
| prev/next with a sequencing cue | 0.7 |
| sibling (`expand_same_section` only) | 0.6 |
| prev/next with no cue | 0.35 |
| semantically similar passage | 0.3 |

A sequencing cue is a list intro ending in ":", or a next node that opens with "Such",
"These", "provided that", a lowercase word or "(b)". Sections don't count against the
budget. The plan's `max_expanded` sets the budget for content nodes. Every expanded node
records the path from the anchor that reached it.

A new metric, `context_recall`, scores expansion. For each gold passage, it takes the
nodes the passage depends on (its section, definitions of the terms it uses, clauses it
cites) and measures the share that reached the subgraph.

Run: `python -m gnn_pruner.evaluation eval/contracts_scale_v1.jsonl --embedder auto --expansion-budgets 0 5 10 20 40`

| Scale set, MiniLM | budget 0 | 5 | 10 | 20 | 40 | unbounded |
|---|---|---|---|---|---|---|
| context_recall | 0.923 | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 |
| content nodes in subgraph | 31.0 | 36.0 | 41.0 | 50.9 | 65.0 | 70.3 |

- Budget 5 already gives full context recall on both sets with both embedders. With
  TF-IDF the ceiling is 0.87, because candidates never retrieve some gold passages, so
  their dependencies can't be reached. The defaults (10, rising to 20 or 30 for
  strategies that need sections or 2-hop chains) leave headroom.
- Subgraph size on the scale set, counting sections, compared with the old unbounded
  expansion: MiniLM 88.7 → 62.1, TF-IDF 45.8 → 34.8. Small set: unchanged.
- End-to-end metrics are unchanged at every budget, including 0. Pruning adds
  definitions and cited clauses directly from the graph, and scoring ranks the subgraph
  rather than reading all of it. Expansion matters most to the scoring features (Phase 7)
  and to a future learned ranker.
- Only once does expansion add gold that retrieval missed: para-06, "How long must we keep
  their secrets?". The answer, "These obligations survive termination...", is now reached
  through a sequencing cue, not through a plain prev/next link.

### Phase 7: relevance scoring

`scoring.py` gives every subgraph node 12 features and a linear score. Each scored node
carries its features and each feature's signed contribution (`ScoredNode.contributions`,
CLI `--explain`). Weights live in the plan (strategy-specific overrides from the planner)
and can be overridden per run (`DocumentQA(weights=...)`, `--weights w.json`). A
different ranker can replace `LinearScorer` behind the `Scorer` interface.

New features: `centrality` (relevance that flows into a node from its subgraph
neighbours, by personalised PageRank minus the node's own restart share) and
`path_coherence` (the Phase 6 best-path priority).

**Dev / calibration set.** [`eval/calibration_v1.jsonl`](../eval/calibration_v1.jsonl)
has 36 new questions over the two short contracts: 20 answerable (including colloquial
paraphrases) and 16 unanswerable (force majeure, assignment, warranties, audit...). It was
written before any signal was looked at. Weights and the abstention calibration were
chosen on it only. `contracts_v1` and the scale set are test sets.

**Weights.** On the dev set, fusion weight 0.3 gained one hit@1 question with both TF-IDF
and MiniLM. Centrality was neutral (TF-IDF) or harmful (MiniLM, −1 to −2 questions).
Path coherence added nothing beyond fusion. So the default is `fusion=0.3`, and centrality
and path coherence are computed and shown with weight 0.

Run: `python -m gnn_pruner.evaluation eval/contracts_v1.jsonl --embedder auto --ranking`

Ranking of content nodes, Phase 7 scored order vs Phase 5 fused order (36 answerable;
a node is relevant if it contains a gold snippet):

| Set, embedder | Order | hit@1 | MRR | nDCG@10 | recall@5 |
|---|---|---|---|---|---|
| small, TF-IDF | fused | **0.806** | **0.819** | **0.814** | 0.819 |
| | scored | 0.750 | 0.796 | 0.807 | **0.847** |
| small, MiniLM | fused | 0.806 | 0.861 | 0.890 | 0.917 |
| | scored | **0.917** | **0.942** | **0.941** | **0.958** |
| small, mpnet | fused | 0.778 | 0.846 | 0.878 | 0.958 |
| | scored | **0.917** | **0.940** | **0.941** | **0.986** |
| scale, TF-IDF | fused | **0.722** | **0.755** | 0.753 | 0.764 |
| | scored | 0.694 | 0.747 | 0.753 | **0.792** |
| scale, MiniLM | fused | **0.861** | 0.894 | 0.893 | 0.889 |
| | scored | 0.833 | 0.893 | 0.894 | **0.944** |
| scale, mpnet | fused | **0.806** | **0.859** | 0.861 | **0.917** |
| | scored | 0.778 | 0.849 | 0.860 | 0.889 |

- With a dense embedder on the short contracts, scoring clearly improves on the fused order:
  hit@1 rises by 0.11 to 0.14.
- On the long documents, and with TF-IDF, the scored order is level with or slightly
  behind fused at hit@1 (one question). It is ahead at recall@5 in 4 of 6 rows. The
  carried-forward Phase 5 issue is fixed for the case it named: on the scale set with
  TF-IDF, scored recall@5 (0.792) now equals the best single method (lexical) and beats
  fused (0.764).
- The fusion weight moved MiniLM's scored hit@1 from 0.889 to 0.917 (small) and from 0.806
  to 0.833 (scale).
- End to end, with abstention off: hit@1 is unchanged everywhere. Scale-set recall with
  MiniLM fell from 1.00 to 0.986 (msa-06). Distractor paragraphs with a high fused score
  fill the 8-item pack before clause 4.2 gets in. This is a pack-cap and dependency
  problem for Phase 8. Packs grew slightly on the small set (MiniLM 5.86 → 6.0) and shrank
  on the scale set (6.86 → 6.58).

**Abstention signal.** Scores are min-max scaled per question, so they cannot say whether
the document holds an answer at all. `EvidenceConfidence` uses absolute signals instead:

- `margin`: the best raw cosine minus the document's median cosine
- `coverage`: the best node's share of the question's IDF-weighted terms
- `anchor`: the question names something that exists, such as a cited section, a defined
  term, a quoted phrase found verbatim, resolved comparison sides, or a resolved summary
  section. Anchored questions are never abstained on.

A logistic model per embedder (`src/gnn_pruner/calibration.json`, C=100) is fitted on the
calibration set. Its threshold is the lowest probability of any unanchored answerable
question there, so it keeps all 20. Refit with
`python -m gnn_pruner.evaluation eval/calibration_v1.jsonl --fit-calibration --embedder <spec>`.

| | TF-IDF | MiniLM | mpnet |
|---|---|---|---|
| confidence AUC, calibration set (fit) | 0.875 | 0.95 | 0.931 |
| confidence AUC, small set (test) | 0.868 | **0.965** | 0.958 |
| confidence AUC, scale set (test) | 0.826 | 0.854 | 0.875 |
| unanswerable abstained on, calibration set | 0/16 | 11/16 | 8/16 |

With hard abstention on (`--abstain`):

| Test set, embedder | abstention_rate | false_abstention_rate | evidence_recall | hit@1 |
|---|---|---|---|---|
| small, MiniLM | **1.00** (was 0.00) | 0.056 (para-03, para-05) | 0.944 (was 1.00) | 0.889 |
| small, mpnet | 0.50 | 0.056 | 0.944 | 0.889 |
| scale, MiniLM | 0.25 | 0.083 (para-02, -03, -05) | 0.903 | 0.778 |
| scale, mpnet | 0.00 | 0.083 | 0.875 | 0.778 |

- **Hard abstention is off by default.** It works on short documents, but it costs
  colloquial paraphrases ("Can we walk away from the deal early?" has a lower margin than
  "What law governs this agreement?"). On long documents it is a net loss: there, distractor
  clauses raise the margin of unanswerable questions (governing law: p = 0.19 → 0.61). The
  signal is always computed and reported (`result.confidence`, CLI line
  `evidence confidence ...`), so Phases 9–10 can tell the LLM that the evidence is weak.
- TF-IDF cannot abstain safely. Its paraphrase questions have zero margin and zero
  coverage, so the threshold that keeps every answerable question abstains on nothing.
  (TF-IDF's `abstention_rate` of 0.25 on the small set comes from empty retrievals, not
  from this signal.)
- Next steps for abstention: a margin that is robust to document length, a cross-encoder
  or NLI check on the top passages, or the Phase 10 LLM itself.

### Phase 8: adaptive pruning

`pruning.py` builds the pack in three steps. Every dropped node records why
(`PrunedSubgraph.drop_reasons`; CLI `--explain` prints a summary and the chain of
dependencies).

1. **Primary evidence, adaptive budget.** Content is taken in score order until either:
   - the score falls below 0.45 of the top score (floor), or
   - the score drops by more than `gap` from one distinct passage to the next.

   Past a gap, a passage still qualifies if its semantic and lexical match both reach
   0.6. This handles "How may Confidential Information be used?", where a boosted
   definition sits above the clause that answers. Widening a sentence to its paragraph,
   and text already covered, do not count as steps. `plan.max_evidence` is only a safety
   cap. Along the way, the pruner skips:
   - low-signal nodes (semantic and lexical both below 0.2)
   - nodes with no provenance (reached only by an adjacent/similar/sibling expansion link)
   - covered text
   - redundant siblings: more than 2 from one section, unless near the top or a strong
     match. Summary and comparison questions are exempt.
2. **Strategy guarantees.** Both comparison sides, the summary's target section (new),
   and the counterpart of each contradiction candidate.
3. **Dependencies**, depth first from the strongest evidence, at most 4:
   - the passage a kept one continues into ("These obligations survive...")
   - definitions of the terms it uses
   - clauses it cites
   - clauses that cite it with a qualifier phrase within 6 words of the citation ("The
     cap in Section 4.1 does not apply")
   - same-section qualifiers ("Claims under this indemnity are subject to Section 4.2")
   - statements elsewhere that say the same thing with a different number ("within four
     hours" / "within one hour"); this keeps both sides of a contradiction for any
     question type

   Qualifiers and cited clauses are followed further: indemnity → "subject to 4.2" → 4.2
   → 4.1.

**Dev sets** (thresholds chosen here only):
- `calibration_v1` (20 answerable)
- `dev_scale_v1` (the same 20 questions over the long documents)
- `dev_pruning_v1` and `dev_pruning_scale_v1`: 11 new multi-passage questions (lists,
  summaries, comparisons, qualifiers, a numeric conflict), on the short and the long
  documents

The multi-passage set mattered. On single-answer questions alone, the sweep favoured gap
0.1 (half the words), which lost 5 of 22 multi-passage questions with TF-IDF.

- **Gap:** 0.2 for dense embedders (0.15 had smaller packs and the same recall for
  MiniLM, but cost TF-IDF). 0.25 for TF-IDF, whose compressed similarity makes every step
  bigger (`pruning.EMBEDDER_DEFAULTS`, like the per-embedder calibration).
- **Rejected:** `gap_after=2` (doubled MiniLM packs) and a dependency budget of 3 (context
  recall 1.0 → 0.79 for TF-IDF).

**Test sets, old pruner → new** (36 answerable; `evidence_recall` / `pack_context_recall` /
`primary_precision` / `mean_pack_size` / `mean_pack_words`; hit@1 unchanged everywhere):

| Set, embedder | recall | context | primary precision | items | words |
|---|---|---|---|---|---|
| small, TF-IDF | 0.833 → **0.861** | 0.841 → 0.841 | 0.493 → **0.711** | 3.58 → 3.61 | 65.9 → 63.9 |
| scale, TF-IDF | 0.806 → **0.833** | 0.795 → 0.795 | 0.401 → **0.599** | 4.50 → 4.56 | 79.4 → 78.2 |
| small, MiniLM | 1.00 → 1.00 | 0.955 → 0.943 | 0.337 → **0.629** | 6.00 → **4.92** | 110.2 → **86.9** |
| scale, MiniLM | 0.986 → **1.00** | 0.932 → **0.943** | 0.277 → **0.501** | 6.58 → **5.94** | 114.1 → **102.0** |
| small, mpnet | 1.00 → 1.00 | 0.955 → 0.943 | 0.326 → **0.612** | 5.78 → **4.78** | 106.2 → **83.7** |
| scale, mpnet | 0.958 → **1.00** | 0.932 → **0.943** | 0.265 → **0.518** | 6.58 → **5.81** | 115.4 → **99.9** |

Run: `python -m gnn_pruner.evaluation eval/contracts_scale_v1.jsonl --embedder auto`

- **The Phase 3 bloat is reduced, not gone.** Dense packs are 10–23% smaller in words and
  now smaller than 6 items on average. Primary precision roughly doubled. All-items
  precision (MiniLM 0.389 small / 0.319 scale) still misses the ≥ 0.5 target, because it
  counts definitions and cited clauses as noise. On long documents, many questions still
  stop at the budget: templated distractor clauses score in a flat band with no gap
  (msa-03 has five "invoice disputes" clauses at 0.95–1.0 of the top score).
- **Phase 7 carry-over fixed:** scale msa-06 now gets clause 4.2 through the qualifier
  chain, instead of losing it to distractors. mpnet's scale para-06 is also fixed, by
  the continuation rule.
- **Honest caveat:** the continuation rule, qualifier proximity, and depth-first
  dependencies were added after the first test run exposed para-06 and msa-06. They are
  rule changes, re-checked on the dev sets, but they were not blind to the test set.
- **Remaining losses:**
  - Small msa-06: the pack has 4.2 and 4.1, but not Section 3, which 4.2 also cites,
    because the dependency budget runs out (context 0.955 → 0.943).
  - TF-IDF scale saas-15: a 0.23 gap after the top clause cuts the answer.
  - TF-IDF dev dpr-02: the old pruner answered it.

Notes on the baseline:
- **Abstention is the main gap.** The pipeline always returns its best-scoring text, so
  questions such as "governing law" and "free trial" get confident but irrelevant evidence.
  It needs an absolute relevance floor, not just a cutoff relative to the top score.
- **Precision is low by design.** Pruning adds definitions and cited clauses alongside the
  passage that answers the question. This metric counts those additions as noise, even
  though they are needed context.
- **Recall and planner accuracy are optimistic.** The questions were written by the
  system's author over a small set of two documents. Treat these as regression checks,
  not quality claims, until questions from real users are added.
- The baseline column covers 26 questions. Phase 2 added saas-15, which asks about a clause reference (Section 2(c)). The saas-09 miss is fixed: pruning now widens a kept sentence to its full paragraph.

## Exit criteria for Phase 0

- [x] Domain, users, scope, and non-goals written down
- [x] Question types mapped to planner labels
- [x] Gold dataset with answerable, unanswerable, and contradiction cases
- [x] Evaluation harness, with a test that checks dataset validity
- [x] Baseline recorded
- [ ] Add 50+ questions from real users over 5+ real contracts (needs input from the domain owner)
