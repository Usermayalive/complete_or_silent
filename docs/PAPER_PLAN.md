# Paper plan: "Complete or Silent"

**Working title:** *Complete or Silent: Certified-Completeness Question Answering over
Documents Beyond Any Context Window*

**Target:** an A\* main track (ACL / EMNLP / NAACL via ACL Rolling Review, or SIGIR).
Fallbacks, in order: Findings of ACL/EMNLP → CIKM / ICAIL / JURIX → the NLLP workshop.
**Pace:** as fast as possible. Phases, not months; parallel where possible; each phase
ends with a go / no-go check.

---

## 1. The claim, in one paragraph

**The problem:** systems that answer from very long documents (contracts, data rooms,
books) answer from the passage they find and stop.
- They miss the exception, override or amendment that sits elsewhere.
- When they say "not found", they cannot tell absence from failure.

**What we propose:** **certified completeness.**
- **Closure certificate:** every positive answer comes with one. A structural qualifier
  graph, a reverse index and a full-document sweep show that every provision that could
  change the answer was read.
- **Absence certificate:** every negative answer comes with one. *Per-document
  self-calibration with mutation canaries* gives a measured miss risk for this document
  and question class, while reading well under 1% of the text.

**The evidence:** on real contracts (CUAD data rooms) and books (NarrativeQA) from 100
to 10,000 pages, plus a new trap benchmark (far exceptions, override chains, unlinked
qualifiers, existence questions):
- certificates hold at their stated level
- answers beat a frontier long-context model and strong retrieval baselines
- at a small fraction of the cost

## 2. Research questions → the experiment that answers each

| # | Question | Experiment | Decides |
|---|---|---|---|
| RQ1 | Do mutation canaries predict the real miss rate on *this* document? | Calibrate on canaries; measure recall on real CUAD clauses and planted facts at 100 / 1,000 / 10,000 pages | Whether absence certificates are valid. **The make-or-break.** |
| RQ2 | Does closure expansion find far exceptions and overrides that others miss? | Trap benchmark: our system vs long-context, RAG and agent baselines | The headline result |
| RQ3 | Are certificates honest? | False-certificate rate ("complete" / "absent" but wrong) vs the stated α | Reviewer trust |
| RQ4 | What does certainty cost? | Guarantee vs text read, AI calls, tokens, time; curves for α = 90 / 95 / 98% | The efficiency story |
| RQ5 | Which parts matter? | Ablations (section 6) | Mechanism credibility |

## 3. What already exists (reused)

All of the following are built:
- the document tree with sentence ids (cite by id)
- the relevance gate and chapter workers
- the central board (quote checks, merging, follow, definitions) and citation checks
- the nested benchmark: S ⊂ M ⊂ L contracts and books, planted facts, CUAD and NarrativeQA
- the evaluation runner (judge, rescore, gate readings)
- the Phase 6 dependency rules and cross-reference / definition links

About 60% of the system.

## 4. Phases

### P0 — Lock the novelty (literature review)  ·  ~2–3 days  ·  parallel with P1
- [ ] Systematic search, 2023–now:
  - conformal / certified retrieval and abstention
  - canaries / synthetic probes for calibration
  - long-document agents: Chain-of-Agents, LongAgent, MemWalker, ReadAgent, RAPTOR,
    PageIndex, DocAtlas, Corpus2Skill
  - legal exception / override reasoning: CLAUSE, Exception Chain Collapse, ContractEval,
    LegalBench-Deep
  - stale / override evidence: FinCacheServe, stale-document poisoning
- [ ] A related-work table: what each does / does not do vs our two certificates
- [ ] **Go / no-go:** no published system provides certificates (closure + absence) for
      documents beyond context. If one does, sharpen the claim around what remains new.
- **You:** read the 8–10 closest papers (needed to defend the work).

### P1 — Build the certificates  ·  ~7–10 days of engineering
- [ ] **U2 qualifier graph:**
  - pattern extraction of qualifies / overrides / amends / precedence
  - target resolution: section number, defined term, schedule, other contract by title
  - tests
- [ ] **U3 reverse index:** clause / section / term / schedule → every sentence pointing
      at it (data-room wide); tests
- [ ] **U4 canary calibration:**
  - mutation canaries per class (fact / clause / qualifier) planted in a hidden copy of
    the index
  - gate ranks → split-conformal reading budget k(α, class) and score thresholds for
    "any X in the room" questions
  - stored with the tree; tests
- [ ] **Q2 calibrated reading plan:** replaces the fixed k
- [ ] **Q4 closure expansion:**
  - forward closure, reverse sweep, full-document text sweep
  - qualifier worker (one call per batch)
  - fixpoint, budget
- [ ] **Q5 precedence:** explicit precedence clause > amendment date > specific over
      general; superseded items marked
- [ ] **Q6 certificates:** closure / absence / uncertified (honest third outcome);
      readable and JSON
- [ ] **Q7 entailment check:** an independent verifier model checks each answer
      sentence against its quotes
- [ ] **Also fix the known engine weaknesses:**
  - two-hop facts (0% for workers)
  - widening cost on unanswerable questions
- **Cost:** mostly no-model code; free-tier Gemini for development.

### P2 — Trap benchmark  ·  ~2–3 days  ·  parallel with P1
- [ ] **Extend `longbench.py` with planted traps in the existing S / M / L rooms:**
  - far exception: the qualifier 10 / 100 / 1,000+ pages away; same contract / schedule /
    another contract
  - override chain: amendment of an amendment
  - unlinked qualifier ("notwithstanding the foregoing limits", no section number)
  - precedence conflict with an order-of-precedence clause
  - existence yes / no ("is there any …?"), single contract and data-room wide
- [ ] Gold for each: the correct answer, every qualifier that must be mentioned, and
      whether a "complete" or "absent" certificate is correct
- [ ] Realism: trap text built by **mutating real CUAD clauses**, not templates (also
      reduces canary / test overlap: different seeds and texts from the calibration
      canaries)
- [ ] **Size:** ≥ 300 trap questions per size tier (S, M; L where the budget allows),
      plus the existing CUAD / NarrativeQA questions
- [ ] Data card; licences (CUAD CC BY 4.0, NarrativeQA Apache 2.0, Gutenberg public
      domain)

### P3 — Validity experiments (RQ1, RQ3, RQ4)  ·  ~3–4 days  ·  mostly free
**Checkpoint after P1 + P2. These experiments decide which outcome row the paper is in.**
- [ ] **Calibrate then test:**
  - canaries at α = 90 / 95 / 98%
  - measure the real miss rate on real CUAD clauses, planted facts and trap qualifiers
    (gate level: no model calls; worker level: a sample with model calls)
- [ ] **Certificate validity:** stated vs measured miss rate, false-certificate rate, with
      confidence intervals (≥ 300 questions per setting)
- [ ] **Where it breaks:** by document size, question class, structure quality (books vs
      contracts), canary type (mutation vs template)
- [ ] **Cost curves:** guarantee vs % of text read and model calls
- [ ] **Go / no-go:**
  - **pass** (measured ≈ stated, within CI) → full paper, A\* main
  - **partial** → position the closure certificate as the main result and absence as an
    analysis; aim main / Findings
  - **fail** → analysis paper ("when self-calibration works"), Findings / workshop

### P4 — Comparisons with strong baselines (RQ2)  ·  ~5–7 days  ·  needs the API budget
- [ ] **Systems:**
  1. a frontier long-context model with the whole document (Claude, 1M context, prompt
     caching; S and M; L does not fit, which is itself a result)
  2. strong RAG: hybrid BM25 + dense + a reranker
  3. a Chain-of-Agents-style baseline: sequential workers over chunks + a manager
     (re-implemented)
  4. ours without closure / certificates
  5. ours, full
- [ ] **Metrics:**
  - answer correctness (judge)
  - qualifier recall (every required exception mentioned)
  - override correctness
  - certificate validity
  - cost (tokens, calls, time, $)
- [ ] **Two answer models** (one closed, one open or second closed), to show the result
      does not depend on one model
- [ ] **Judge validation:** the AI judge vs human labels on a sample (from P5)
- **Budget estimate:** ~$200–400 in total, mostly the long-context baseline (prompt
  caching cuts repeat reads of one document to ~5% of the input price). A paid
  dry-run estimate is shown before every run.

### P5 — Human evaluation  ·  runs alongside P3 / P4
- [ ] 50–100 answers + certificates rated by 2 annotators (law students or a lawyer):
  - answer correct?
  - every relevant exception included?
  - certificate justified?
- [ ] Agreement (Cohen's κ); the AI judge's agreement with humans
- [ ] Your 36-fact spot-check sheet (`docs/fact_check_sheet.md`) is the pilot: do it first
- **You:** find 1–2 annotators (classmates in law, or paid).

### P6 — Ablations (RQ5)  ·  ~2 days  ·  mostly free (gate-level) + a small model-call sample
- [ ] Remove each part, one at a time:
  - reverse sweep / text sweep / forward closure
  - canaries (a fixed k instead)
  - mutation canaries (template canaries instead)
  - GNN gate (the lexical gate instead)
  - precedence step
- [ ] Each ablation reported on trap recall, certificate validity and cost

### P7 — Write and submit  ·  ~7–10 days (drafting starts during P3)
- [ ] **Structure** (8 pages + appendix):
  1. intro and motivation (the BETA example)
  2. related work
  3. problem: what "complete" and "absent" mean formally
  4. method
  5. benchmark
  6. experiments: RQ1–RQ5
  7. limitations: exchangeability assumption, structure quality, cost of the long-context
     baseline
  8. ethics: legal-advice disclaimer, data licences
- [ ] Figures:
  - the system diagram
  - certificate examples
  - guarantee-vs-cost curves
  - stated-vs-measured validity plot
  - the trap results table
- [ ] Artefacts: code + benchmark + data card on GitHub (anonymised for review)
- [ ] ARR requirements: the responsible-NLP checklist; **AI-assistance disclosure**
- [ ] Submit to the next ACL Rolling Review deadline after P3 passes (check current dates
      on the ARR site)
- [ ] arXiv preprint (after anonymity rules allow)

## 5. Critical path and parallelism

```
P0 lit review ──┐
P1 build ───────┼──► P3 validity ──► P4 baselines ──► P7 write ──► submit
P2 traps ───────┘        │  (go / no-go)    ▲
P5 human eval (starts with the fact sheet) ─┘
P6 ablations (after P3, alongside P4)
```

**Fastest realistic path:** **~5–7 weeks of focused work** from today to a submittable
paper. It depends on the API budget being available at P4 and on annotators at P5.

## 6. What only you can do (blocking items)

| Item | Needed by | Why |
|---|---|---|
| **A faculty mentor / co-author** | as early as possible (by P3) | Raises acceptance odds a lot; writes the recommendation |
| **API budget ~$200–400** (the approved $50 covers early tests only) | P4 | The frontier long-context baseline is the decisive comparison |
| **Annotators** (1–2 law students or a lawyer) + your fact sheet | P5 | Human validation of answers and certificates |
| **Read the closest 8–10 papers** | P0–P7 | You must defend the novelty and design |
| **The AI-assistance disclosure** wording | P7 | Required by ACL venues |

## 7. Risks and responses

| Risk | Response |
|---|---|
| Canary guarantees do not transfer to real clauses (RQ1 fails) | Lead with closure certificates + traps; publish the calibration analysis; Findings / workshop |
| A very similar paper appears | Sharpen to the uncovered half (closure for far exceptions in data rooms, or absence at 10,000 pages); speed matters |
| The long-context baseline wins on S / M | Expected on small documents; the paper's claim is beyond-context scale + certificates; report honestly |
| Budget or rate limits slow P4 | Prompt caching, batch discounts, smaller S / M samples with CIs |
| Reviewers doubt the AI judge | Human agreement numbers (P5); exact-match metrics for planted values |

## 8. Definition of done (submittable)

- [ ] RQ1–RQ5 each answered with ≥ 300 questions per main setting and confidence intervals
- [ ] Certificates' stated vs measured validity reported, including where they fail
- [ ] A frontier long-context baseline, strong RAG and an agent baseline compared on
      identical questions
- [ ] Human evaluation with agreement
- [ ] Ablations
- [ ] Code + benchmark released; data card; limitations; disclosure
