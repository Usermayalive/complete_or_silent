# Complete or Silent (A + B): the idea end to end

One file that explains the whole research idea: what it is, how it works, how it is
built, why it is new, and why it has a real chance at a top venue. Details live in
`docs/COMPLETE_OR_SILENT.md` (method) and `docs/PAPER_PLAN.md` (phases).

---

## 1. The idea in three sentences

Systems that answer questions over huge documents find one passage and answer from it.
They miss the exception, override or amendment written somewhere else, and when they say
"not found" nobody knows whether the answer is truly absent or the system simply failed
to look.

**Complete or Silent** makes every answer come with a certificate:

| | Name | It proves |
|---|---|---|
| **A** | **Closure certificate** (for "found" answers) | everything that could change this answer (exceptions, overrides, amendments, definitions) was read |
| **B** | **Absence certificate** (for "not found" answers) | if the answer were in the document, we would have found it with probability ≥ α (for example 98%), measured on *this* document |

If neither can be given, the system says so ("uncertified, because …"). That honest third
outcome is part of the design.

---

## 2. Why this problem matters

- **Real failure, real cost.** In contracts the clause you find is often not the final
  word: "Section 12.3 caps liability at 12 months of fees" can be overridden by a schedule
  ("notwithstanding Section 12.3 …"), replaced by an amendment, or carved out ("shall not
  apply to fraud") hundreds of pages away. An answer without them is wrong, but looks right.
- **"Not found" is the most dangerous answer.** In due diligence, "there is no
  change-of-control clause" is acted on. Today no system can say how likely it is to have
  missed one.
- **Documents beyond any context window.** A data room of 500 contracts (≈ 8,600 pages,
  ≈ 2.9M tokens) cannot be given to a model whole, so completeness cannot come from
  "just read everything".

---

## 3. How it works (end to end)

### 3.1 Once per document (upload)

```
U1  Document tree ✅        chapters → sections → passages, every sentence has an id
U2  Qualifier graph         who qualifies / overrides / amends what (patterns, no model)
U3  Reverse index           for every clause: every sentence that points AT it
U4  Canary calibration      fake facts planted in a hidden copy → measured miss rates
                            → "open k chapters for 98% recall" for this document
```

**U2: qualifier graph.** Patterns find qualifier cues and their targets:

| Cue | Link |
|---|---|
| "subject to", "except as provided in", "provided that", "shall not apply" | qualifies |
| "notwithstanding", "in lieu of", "shall prevail" | overrides |
| "is hereby amended", "deleted and replaced", "supersedes" | amends |
| "order of precedence", "in the event of any conflict" | precedence |

Targets: a section ("Section 12.3"), a schedule ("Schedule 3"), a defined term ("Fees"),
another contract named by its title, or "the foregoing". Targets resolve **inside the
sentence's own contract** (a data room may contain 500 different "Section 12.3"s); an
amendment's references resolve into the contract its title names.

**U3: reverse index.** For every clause, every sentence anywhere that points at it,
including links to its parent ("Section 12" also concerns 12.3). The far exception is
found by **lookup**, not by reading forward.

**U4: canary calibration (the core of B).**
1. Take real sentences of the document and mutate them (change names, numbers, dates,
   using the document's own vocabulary) to make **canaries**: fake facts that look like
   the document's own text.
2. Plant them in a hidden copy of the search index, in random chapters.
3. For each canary, ask the question that targets it and record **at which rank the
   gate places the chapter that holds it**.
4. **Split-conformal calibration:** with n canary ranks r₁ … rₙ of one question class,
   choose k = the ⌈(n+1)·α⌉-th smallest rank. If a real answer behaves like a canary
   (exchangeability), opening the top k chapters contains it with probability ≥ α.
5. Store per class (fact / clause / qualifier): k for α = 90 / 95 / 98%, a score threshold
   for "any X in the room?" questions, and the measured miss rate with its upper
   confidence bound (Clopper–Pearson).

Messy documents automatically get bigger k (more reading) or weaker guarantees. No model
calls are needed for calibration.

### 3.2 Per question

```
Q1 Understand     routing ✅, scope ✅, class (fact / clause / list / existence), target α
Q2 Plan           open k(α, class) chapters from this document's calibration table
Q3 Read           chapter workers post sentence ids; the board verifies and quotes ✅
      found ─► Q4 Closure expansion ─► Q5 Precedence ─► Q6a Closure certificate (A)
      none  ─► Q6b Absence certificate (B)
      else  ─► Q6c "Uncertified" + reason
Q7 Answer         main clause → qualifications → overrides; every sentence checked
                  against its quotes by an independent verifier model
```

**Q4: closure expansion (the core of A).** Starting from the found clause:
1. forward links: its definitions, cross-references, schedules ✅ (already built)
2. reverse index: every sentence that qualifies / overrides / amends it (U3)
3. text sweep of the whole document for unlinked qualifiers about the same subject
   ("the limitations above shall not apply to fraud"), no model
4. a qualifier worker (one model call per batch) labels each candidate:
   qualifies / overrides / irrelevant, by sentence id only
5. repeat for every newly accepted item until nothing new is added (fixpoint) or the
   budget runs out. **Fixpoint reached = COMPLETE.** Budget hit = "uncertified".

**Q5: precedence.** Explicit precedence clause > later amendment > specific over general.
Superseded text is kept and marked, never silently dropped.

**Q6: certificates** (readable and JSON):
- **Closure:** forward / reverse / sweep counts, precedence applied, COMPLETE or not
- **Absence:** chapters read, canaries used, measured miss rate and its upper bound,
  certified level (for example "not found, certified at 98%")
- **Uncertified:** the reason (budget, no reliable calibration for this class)

### 3.3 Worked example (illustrative)

Data room: BETA Logistics, 510 contracts, ~8,600 pages. Question: *"What is ACME's
maximum liability under the BETA MSA?"*

| Step | What happens |
|---|---|
| Read | the worker finds MSA §12.3: "liability ≤ fees of the last 12 months" (a normal system stops here) |
| Forward | the definition of "Fees" (§1) |
| Reverse | Schedule 3 "Notwithstanding Section 12.3 … ≤ 5M USD for data breaches" (p. 2,817); Amendment 2 "§12.3 deleted and replaced: 24 months" (p. 7,402) |
| Sweep | Schedule 5 "shall not apply to fraud" (no section number, so no link) |
| Precedence | the amendment is later → 24 months; §20 says schedules prevail → carve-outs stand |
| Answer | **24 months of fees** (12-month cap superseded), except data breaches ≤ 5M USD and no cap for fraud; all cited |
| Certificate | **COMPLETE**: 0.6% of the text read, ~9 model calls |

*"Is there a change-of-control clause in the BETA–ACME MSA?"* → the contract and both
amendments are read, nothing is found → **"Not found, certified at 98%"** (measured miss
rate 1.5% on 200 clause canaries in this contract; upper bound 2.0%).

---

## 4. Implementation

### 4.1 What is reused (built and measured in GNN PRUNER, ~60%)

- document tree with sentence ids, cite-by-id quoting
- relevance gate, chapter workers, central board (verification, merging, definitions)
- nested benchmark: CUAD data rooms (130 / 977 / 8,619 pages) and NarrativeQA libraries
  (119 / 1,001 / 9,988 pages) with planted facts
- evaluation runner (model judge, rescoring, gate readings), dependency rules

### 4.2 New modules: `src/gnn_pruner/certify/`

| Module | Step | Model calls | Status |
|---|---|---|---|
| `qualifiers.py` | U2 qualifier graph | none | **first version written, 5 tests pass** (see §8) |
| `reverse.py` | U3 reverse index | none | **first version written** |
| `canaries.py` | U4 mutation canaries + split-conformal tables | none | planned |
| `plan.py` | Q2 calibrated reading plan | none | planned |
| `closure.py` | Q4 forward + reverse + sweep + qualifier worker, fixpoint | 1 per batch | planned |
| `precedence.py` | Q5 ordering, superseded marks | none | planned |
| `certificate.py` | Q6 closure / absence / uncertified | none | planned |
| `verify.py` | Q7 entailment check per answer sentence | 1 | planned |
| `traps.py` | trap benchmark on top of `longbench.py` | none | planned |

**Design rules:** cite by id everywhere; calibration and graph building use no model (the
cost goes only where it must); three honest outcomes; a certificate is never phrased as
certainty about one answer.

### 4.3 The trap benchmark (new dataset contribution)

Planted inside the existing real CUAD and NarrativeQA rooms, built by **mutating real
clauses** (not templates):
- **far exception:** the qualifier 10 / 100 / 1,000+ pages away
- **override chain:** an amendment of an amendment
- **unlinked qualifier:** "notwithstanding the foregoing limits", no section number
- **precedence conflict:** with an order-of-precedence clause
- **existence yes / no:** "is there any …?", in one contract and room-wide

Gold for each: the correct answer, every qualifier that must be mentioned, and whether
"complete" or "absent" is the correct certificate. Target ≥ 300 questions per size.

---

## 5. Why it is new

| Existing line of work | What it does | What it does not do |
|---|---|---|
| Long-context models (+ "Lost in the Middle", RULER) | read everything that fits | cannot hold 2.9M tokens; no proof nothing was missed |
| RAG, GraphRAG, HippoRAG, RAPTOR | retrieve relevant passages | no completeness notion; "not found" is uncalibrated |
| Agents (Chain of Agents, MemWalker, PageIndex) | read / navigate in parts | stop at the first good passage; no certificate |
| Conformal prediction for LLMs / retrieval | coverage guarantees on held-out labelled data | needs labelled data from the same distribution; not per document, not beyond-context |
| Legal NLP (CUAD, ContractEval, exception reasoning) | clause finding, classification | no override / amendment closure over whole data rooms |

**The two new ideas:**
1. **Per-document self-calibration with mutation canaries.** The guarantee is measured on
   the very document being asked about, with no labelled data, using fake facts that
   look like the document's own text.
2. **Closure as a fixpoint over a structural qualifier graph + reverse index + sweep.**
   "Complete" becomes a checkable property, not a hope.

Plus a benchmark of traps that existing systems are expected to fail.

*To be confirmed in P0 (a systematic literature search) before claiming it in the paper.*

---

## 6. Why it has a higher chance at a top venue

**What A\* reviewers look for, and how this answers each:**

| Reviewer asks | This work |
|---|---|
| A real, clearly stated problem? | wrong-but-confident answers and unverifiable "not found" in huge documents |
| A new idea, not an engineering stack? | certificates as a new output type; per-document canary calibration |
| A formal claim that can be tested? | the stated α is checked against the measured miss rate (RQ1, RQ3) |
| Strong baselines? | a frontier long-context model, strong RAG, a Chain-of-Agents baseline, ours without certificates |
| A reusable contribution? | the trap benchmark + code |
| Honesty about limits? | "uncertified" outcome; exchangeability assumption stated; failures reported |

**Compared with the GNN PRUNER paper alone:** that paper is a solid system paper
(engineering + measurements). This one adds a *guarantee* that can be proven right or
wrong, which is what moves a paper from "workshop / Findings" to "main track".

**Honest odds depend on one experiment (P3):**

| P3 outcome | Paper | Likely venue |
|---|---|---|
| Canary-stated miss rate ≈ real miss rate (within CI) | full paper | ACL / EMNLP / NAACL main, SIGIR |
| Only closure works well | closure as the main result, absence as analysis | main track or Findings |
| Calibration fails | "when self-calibration works" analysis | Findings / workshop (NLLP) |

No acceptance is guaranteed; top venues accept roughly one in four or five submissions.
The plan is built so that every outcome still produces a publishable paper.

---

## 7. Experiments (what the paper must show)

| # | Question | How |
|---|---|---|
| RQ1 | Do canaries predict the real miss rate on this document? | calibrate on canaries, measure on real CUAD clauses and planted facts at 100 / 1,000 / 10,000 pages |
| RQ2 | Does closure find far exceptions others miss? | trap benchmark: ours vs long-context, RAG, agent baselines |
| RQ3 | Are certificates honest? | false-certificate rate vs α, with confidence intervals |
| RQ4 | What does certainty cost? | guarantee vs text read, calls, tokens, time (α = 90 / 95 / 98%) |
| RQ5 | Which parts matter? | ablations: no reverse index, no sweep, fixed k instead of canaries, template canaries, no precedence |

Plus a human check (2 annotators, 50–100 answers) and two different answer models.

---

## 8. Plan and status

| Phase | What | Cost |
|---|---|---|
| P0 | literature search, lock the novelty | free |
| P1 | build the certificates (§4.2), fix two-hop and widening cost | free tier |
| P2 | trap benchmark | free |
| P3 | validity experiments (**make-or-break**) | mostly free |
| P4 | strong baselines, incl. a frontier long-context model | ~$200–400 API |
| P5 | human evaluation | annotators |
| P6 | ablations | mostly free |
| P7 | write and submit (ARR) | — |

**Status (paused at your request):** a first `certify/qualifiers.py` (U2) and
`certify/reverse.py` (U3) exist with `tests/test_certify.py` (5 tests pass on a BETA-style
example: schedules grouped with their contract, "Section 12.3" resolved inside the right
contract, amendments resolved through their title, unlinked "shall not apply" kept). On the
real 130-page CUAD room it runs in 0.06 s and finds 302 links; section links look correct
(for example "Subject to Section 8.2" → 8.2). **Known issue found:** party names and
"Agreement" are picked up as defined terms ("BLI", "Ginkgo"), which creates noisy term
links; this needs a filter before U3 can be trusted. Nothing is committed yet; it can be
kept or discarded when work resumes.

**What you need to do:** read the 8–10 closest papers (P0), find 1–2 annotators (P5),
approve the API budget (P4), and give the new GitHub repo URL.
