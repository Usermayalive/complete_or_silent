# Complete or Silent: method

Every answer either comes with proof that **nothing that could change it was left
unread**, or the system says exactly what it could not certify. Plan and experiments:
`docs/PAPER_PLAN.md`.

✅ already built · 🆕 new for this work

## Flow

```
UPLOAD (once per document)
  U1 Tree + summaries + key facts ✅
  U2 Qualifier graph 🆕   who qualifies / overrides / amends what
  U3 Reverse index 🆕     for every clause: everything that points AT it
  U4 Canary calibration 🆕 fake facts planted in a hidden copy → measured miss rates
                          → per-document table "open k chapters for 98% recall"
QUESTION
  Q1 Understand → Q2 Calibrated reading plan → Q3 Workers + board ✅
     found:   Q4 Closure expansion 🆕 → Q5 Precedence 🆕 → Q6a Closure certificate 🆕
     nothing: Q6b Absence certificate 🆕
     neither: Q6c "Uncertified" + reason (honest third outcome)
  Q7 Answer from the pack + entailment check 🆕 → Q8 answer + citations + certificate
```

### Upload
**U2. Qualifier graph** (patterns, no model; optional model check for ambiguous cases).
Each match becomes a typed link to its target: a section number, a defined term, a
schedule, or a clause in another contract of a data room (by title).

| Phrase | Link type |
|---|---|
| "subject to", "except as provided in", "provided that", "unless" | qualifies |
| "notwithstanding", "in lieu of", "shall prevail over" | overrides |
| "is hereby amended / deleted / replaced", "supersedes" | amends |
| "order of precedence" clauses | precedence rule |

**U3. Reverse index.**
- For every clause / section / term / schedule: every sentence anywhere that points at it.
- Finds the exception 3,000 pages away by lookup, not by reading forward.

**U4. Canary calibration** (no model).
1. Mutate real sentences of each class (fact / clause / qualifier): names, numbers, dates
   changed, using the document's own vocabulary.
2. Plant them in a hidden copy of the index.
3. Record where the gate ranks the right chapter.
4. Split-conformal calibration gives, per class:
   - chapters to open for 90 / 95 / 98% recall
   - a score threshold for "any X in the room" questions
   - the reverse sweep's recall on qualifier canaries

Messy documents automatically get bigger budgets or weaker guarantees.

### Question
- **Q1. Understand:**
  - routing (named unit, content words) ✅
  - scope ✅
  - class: fact / clause / list / existence 🆕
  - target certainty α (default 98%) 🆕
- **Q2. Plan:** open k(α, class) chapters from the calibration table (named units always);
  for "any X" questions, every unit above the calibrated threshold.
- **Q3. Workers:** post sentence ids; the board verifies by id and quotes exactly ✅.
- **Q4. Closure expansion:**
  1. forward links: definitions, cross-references, schedules ✅
  2. reverse index lookup: who qualifies / overrides / amends this
  3. full-document text sweep for unlinked qualifiers (no model)
  4. qualifier worker (one call per batch): qualifies / overrides / irrelevant, ids only
  5. repeat to a fixpoint or the budget
- **Q5. Precedence:** explicit precedence clause > amendment date (later wins) > specific
  over general. Superseded text is kept and marked.
- **Q6. Certificates:**
  - **closure:** forward / reverse / sweep counts, precedence applied, COMPLETE or not
  - **absence:** chapters read, canaries, measured miss rate with its upper bound,
    certified level
  - **uncertified:** the reason (budget, no reliable calibration for this class)
- **Q7. Answer:** main clause → qualifications → overrides (superseded marked); citations
  checked ✅; each sentence checked for entailment by an independent verifier 🆕.

## Worked example (illustrative numbers)

**The data room:** BETA Logistics, 510 contracts, ~8,600 pages (~2.9M tokens, beyond a
1M context). Planted in it:

| Where | Page | Text |
|---|---|---|
| MSA BETA–ACME §12.3 | 4,410 | liability ≤ fees paid in the **12 months** before the claim |
| MSA §1 | 4,012 | "Fees" means all amounts invoiced |
| Schedule 3 | 2,817 | "**Notwithstanding Section 12.3**", data-protection liability ≤ **5,000,000 USD** |
| Schedule 5 | 2,903 | the limitations "**shall not apply to fraud**" (no section number: unlinked) |
| Amendment No. 2 (2022) | 7,402 | §12.3 "**deleted and replaced**": **24 months** |
| MSA §20 | 4,455 | Schedules prevail over the body |

### Q1: "What is ACME's maximum liability under the BETA Master Services Agreement?"
1. **Routing** names the contract. The worker posts §12.3 (12 months). A careless system
   stops here.
2. **Closure expansion:**
   - forward: the "Fees" definition
   - reverse index: Schedule 3 (overrides), Amendment 2 (amends), §9.1 (mention)
   - text sweep: Schedule 5 (fraud)
   - the qualifier worker classifies them; a fixpoint is reached
3. **Precedence:**
   - Amendment 2 supersedes 12 months (it is later)
   - Schedules prevail (§20), so the carve-out stands
4. **Answer:** 24 months of fees [Amendment 2, p. 7,402], replacing the 12-month cap
   [§12.3, superseded]. Except:
   - data-protection breaches ≤ 5M USD [Schedule 3, p. 2,817]
   - no cap for fraud [Schedule 5, p. 2,903]

   "Fees" defined [§1].
5. **Certificate:** COMPLETE. Read 0.6% of the text, ~9 model calls.

### Q2: "Does the BETA–ACME agreement have a change-of-control clause?"
- **Read:** the entire contract and both amendments pointing at it; nothing found.
- **Absence certificate:** a 1.5% measured miss rate on 200 clause canaries in this
  contract (upper bound 2.0%) → "Not found, certified at 98%".

### Q3: "Which contracts in the data room are triggered by a change of control of BETA?"
- **Screen:** a calibrated threshold screens all 510 contracts (no model); 37 are opened.
- **Found in 4,** with clauses and pages.
- **Absence certificate for the other 473** (miss risk ≤ 2%).
- **Cost:** 7% of the text read, ~41 calls.

## What other approaches likely do (to be measured)

| Approach | Likely answer |
|---|---|
| Frontier long-context model | cannot load 2.9M tokens; on smaller rooms often "12 months" |
| Plain RAG | "12 months", or 24 without the exceptions |
| Workers without closure (built today) | "12 months" |
| Complete or Silent | 24 months + both exceptions, with a certificate |

**One line:** other systems answer from the clause they find; this one proves it checked
everything that could change the answer, and says how sure it is when it says "not
there".

---

## How this relates to GNN PRUNER

This project grows out of GNN PRUNER (frozen at tag `gnn-pruner-final` in
`Usermayalive/GNN-PRUNER`). Same goal: shrink a huge document into a small,
structure-aware, checked set of evidence before a model answers. What changes is the
"brain":
- **Before:** a trained GNN scored nodes and pruned the graph.
- **Now:** a calibrated, mostly keyword-based gate picks chapters, model workers read them,
  **the graph proves completeness** (closure certificate), and **canaries measure the
  risk** (absence certificate).

| GNN PRUNER component | Built there | Role here | Weight |
|---|---|---|---|
| 4-layer document tree / DAG, sentence ids | yes | **Core:** the tree workers navigate; cite by id | high |
| Document graph (cross-references, definitions, Phase 6 dependency rules) | yes | **Core, expanded** into the qualifier graph + reverse index | high |
| Evidence pack, topology routing, answer step, citation checks (Phases 9–10) | yes | Reused for the final cited answer | medium |
| Gate, chapter workers, central board, routing (v4-3 / v4-4) | yes | Reused; the gate gets calibrated budgets | high |
| Long-document benchmark (nested 100 / 1,000 / 10,000 pages; planted facts) | yes | Reused; extended with traps | high |
| GNN (CoarseGNN, FineGNN, cross-edge attention) | yes | **Optional** gate signal, kept only if an ablation shows it helps | low |
| Graph coarsening (G3) | yes | The coarse-to-fine *idea* survives (chapter → section → passage); the learned coarsening is not used | low |
| Distributed system (Ray, METIS partitioning, halos) | yes | Not used at runtime on one laptop (workers are threads); chapter slicing comes from it; a cluster deployment could use Ray | none (future) |
| Silver Paths + PPO training | yes | Not used (they trained the GNN) | none |
| Phase 7 confidence signal | yes | Replaced by model judgement + certificates | none |

**Retrieval style:** mostly **vectorless**, close to PageIndex.
- **Gate:** BM25 on each chapter's best sentence, plus a table-of-contents screen in which
  the model reasons over titles and summaries.
- **Workers:** navigate the tree by reasoning and search inside a chapter with BM25.
- **Canaries, qualifier graph, reverse index:** no vectors.
- **Optional:** MiniLM vectors are an extra gate signal.

It is "RAG" only in the broad sense (find evidence, then write); plain top-k chunk RAG is
a baseline.

**Code reuse:** about 50–60%.

---

## Implementation design (planned; not started)

New package `gnn_pruner/certify/` (renamed with the project later). Everything below is a
plan; no code is written yet.

| Module | Holds | Key structures | Model calls |
|---|---|---|---|
| `qualifiers.py` (U2) | cue patterns, target extraction, target resolution | `QualifierLink(sentence, passage, chapter, kind ∈ {qualifies, overrides, amends, precedence, mentions}, cue, target, target_label, resolved)` | none (optional check for ambiguous cues) |
| `reverse.py` (U3) | reverse index | `pointing_at(node)` → links targeting the node, its ancestors below the chapter, or its descendants (a link to "Section 12" also concerns 12.3) | none |
| `canaries.py` (U4) | mutation canaries per class; hidden-copy index; gate ranks; split-conformal tables | `Calibration{class → {α → chapters k, score threshold}, canaries n, measured miss, upper bound}`, stored with the tree | none |
| `plan.py` (Q2) | reading plan from the calibration table, the question class and α; "any X in the room" uses the score threshold | `ReadingPlan(chapters, α, class, basis)` | none |
| `closure.py` (Q4) | forward closure, reverse sweep, full-document text sweep, qualifier worker, fixpoint | `Closure(items, forward, reverse, swept, verdicts, complete)` | one per batch of candidate qualifiers |
| `precedence.py` (Q5) | precedence clause > amendment date > specific over general | ordered items, superseded marks, unresolved conflicts flagged | none |
| `certificate.py` (Q6) | closure / absence / uncertified certificates, readable + JSON | `Certificate(kind, level, read, evidence, reasons)` | none |
| `verify.py` (Q7) | entailment check of every answer sentence against its quotes | per-sentence verdicts | one |
| `traps.py` (P2) | trap generator built on `longbench.py`: far exception, override chain, unlinked qualifier, precedence conflict, existence yes / no; gold for answer, required qualifiers, certificate correctness | trap questions in `eval/long/` | none |

**Design rules (from what GNN PRUNER measured):**
- **Resolve references inside the right contract.**
  - The existing document graph resolves "Section 12.3" document-wide (first match). In a
    data room of 510 contracts that points to the wrong contract.
  - Qualifier targets resolve within the sentence's own contract (its chapter, its slices
    "(k/n)", and the schedules lifted after it).
  - They cross into another contract only when the sentence names it (amendments: "…of
    the Master Services Agreement between BETA and ACME"), matched by distinctive title
    words as in routing.
- **Cite by id everywhere:** workers and the qualifier worker return sentence ids only.
- **Calibration and graph building use no model:** the expensive part is only where it
  must be (workers, qualifier worker, entailment, answer).
- **Three honest outcomes:** complete, absent (certified at α), or uncertified with a
  reason. A certificate is never phrased as certainty about one answer.

---

## Optional extension: a learning loop for the GNN

Every accepted board finding is a free label ("for this question, this passage held the
answer"). Retraining the GNN gate on those labels from real documents would address its
domain shift (it was trained on generated contracts). Showing the gate improve and
costs fall with use would be a secondary result.

Not on the critical path: only if time allows after P3.
