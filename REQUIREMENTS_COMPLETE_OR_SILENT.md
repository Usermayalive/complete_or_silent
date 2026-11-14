# Requirements: Complete or Silent

Requirements, metrics, experiments, timeline and status for the research paper
**"Complete or Silent: Certified-Completeness Question Answering over Documents Beyond Any
Context Window"**.

Related files: `docs/IDEA_COMPLETE_OR_SILENT.md` (the idea end to end),
`docs/COMPLETE_OR_SILENT.md` (method), `docs/PAPER_PLAN.md` (phases).
`REQUIREMENTS.md` stays the requirements file of GNN PRUNER, whose engine this project
reuses.

Status: ✅ done · 🟡 partly done · ⬜ not started

---

## 1. Goal

Answer questions over documents too large for any language model (contracts, data rooms,
books; 100 to 10,000+ pages) so that every answer carries a certificate:

| Outcome | Certificate | Meaning |
|---|---|---|
| Found | **Closure (A)** | every exception, override, amendment and definition that could change the answer was read |
| Not found | **Absence (B)** | if the answer were in the document, it would have been found with probability ≥ α, measured on this document |
| Neither | **Uncertified** | the system says why (budget, no reliable calibration) |

**Target venue:** ACL 2027 via the February 2027 ACL Rolling Review deadline (check dates
on aclrollingreview.org). Fallback: the next ARR cycle (EMNLP 2027), then Findings, then
the NLLP workshop.

## 2. Constraints

| ID | Constraint |
|---|---|
| C1 | Runs on one laptop: 8 GB RAM, CPU only. No local language models (they froze the laptop); hosted APIs only. |
| C2 | API keys live only in `~/.zshrc`; never in the project, the chat or git. |
| C3 | Development on the Gemini free tier (about 500 calls per day per model). Paid runs only after the user approves the estimated cost. |
| C4 | Runs at 10,000 pages are run alone, with `caffeinate`. |
| C5 | Every reported number is measured, with its sample size and confidence interval. Nothing is reported that was not run. |
| C6 | Calibration and graph building use no model calls. |
| C7 | Cite by sentence id everywhere; quotes are always exact source text. |
| C8 | The old GNN PRUNER repository (tag `gnn-pruner-final`) is frozen; this project goes to a new repository. |

## 3. Foundation reused from GNN PRUNER

| ID | Requirement | Status |
|---|---|---|
| F1 | Document tree with sentence ids; `quote(ids)` returns exact text | ✅ |
| F2 | Chapters from the PDF outline; schedules as chapters; long text sliced | ✅ |
| F3 | Relevance gate (lexical best sentence, optional dense, contents screen, 3 rounds) | ✅ |
| F4 | Chapter workers (open / read / search / done, ≤ 4 steps) | ✅ |
| F5 | Central board: id verification, merging, definitions, free-text follow | ✅ |
| F6 | Forward links: definitions, cross-references (dependency rules) | ✅ |
| F7 | Citation checks and abstention | ✅ |
| F8 | Nested benchmark: CUAD rooms 130 / 977 / 8,619 pages; NarrativeQA libraries 119 / 1,001 / 9,988 pages; planted facts | ✅ |
| F9 | Evaluation runner: model judge, rescoring, gate readings, resume | ✅ |
| F10 | Two-hop facts for workers (0% in run 1) | ✅ bridge step (no model): contracts S two-hop coverage 0.17 → 0.83, correct 0/2 → 2/2 |
| F11 | Lower cost of widening the search on unanswerable questions | ✅ word-evidence filter + 2-step widening workers: absent questions 10.0 → 8.5 calls, total 297 → 278, same refusals |

## 4. Functional requirements (new)

### 4.1 Upload (once per document)

| ID | Requirement | Status |
|---|---|---|
| U2.1 | Qualifier cues → kinds: qualifies / overrides / amends / precedence | 🟡 written, 5 tests |
| U2.2 | Targets: section, schedule, defined term, other contract, "the foregoing" | 🟡 written |
| U2.3 | Targets resolve inside the sentence's own contract (its slices and schedules) | 🟡 written, tested |
| U2.4 | Amendments resolve references into the contract their title names | 🟡 written, tested |
| U2.5 | Cues without a target are kept as unresolved (for the sweep) | 🟡 written, tested |
| U2.6 | Filter noisy defined terms (party names, "Agreement", "Parties") | ✅ 197 → 18 term links on the 130-page room |
| U2.7 | Link precision ≥ 90% on a hand-checked sample of real links | 🟡 sheet ready: `docs/qualifier_check_sheet.md` (all 54 resolved section / term / schedule links of the 130-page room); **you:** rate it |
| U3.1 | Reverse index: node → every link pointing at it, its ancestors below the chapter, or its descendants | ✅ written, tested (far override, amendment, other contract excluded) |
| U3.2 | Reverse-index recall on planted far exceptions ≥ 95% | ⬜ |
| U4.1 | Mutation canaries per class (fact / clause / qualifier) from the document's own sentences | ✅ numbers and names mutated; queried by common words; plus type canaries for clause-type questions (see READINGS) |
| U4.2 | Canaries planted in a hidden copy of the index; the real index is never changed | ✅ tested; source sentence blanked in the copy |
| U4.3 | Gate rank of each canary's chapter recorded | ✅ room rank and within-unit rank |
| U4.4 | Split-conformal reading budget k(α, class) for α = 90 / 95 / 98% | ✅ room and unit tables; too few canaries → open everything |
| U4.5 | Score threshold for "any X in the room?" questions | ⬜ |
| U4.6 | Measured miss rate with Clopper–Pearson upper bound, stored with the tree | 🟡 computed and saved as JSON; not yet cached next to the tree by `upload` |
| U4.7 | Calibration uses no model calls and runs at 10,000 pages within 8 GB | ✅ 8,619 pages: 18–19 min, ≤ 804 MB |

### 4.2 Per question

| ID | Requirement | Status |
|---|---|---|
| Q1 | Question class (fact / clause / list / existence) and target α | ✅ pattern-based, tested |
| Q2 | Calibrated reading plan k(α, class) replaces the fixed k; named units always opened | ✅ wired into `ask(calibration=, alpha=)`; records the level reached |
| Q4.1 | Closure: forward links of every accepted item | ✅ `certify/closure.py`, tested |
| Q4.2 | Closure: reverse-index lookup of every accepted item | ✅ tested (override, amendment, subject-to) |
| Q4.3 | Closure: whole-document text sweep for unlinked qualifiers on the same subject (no model) | ✅ candidates ranked by shared words; unverified without a model |
| Q4.4 | Qualifier worker: one model call per batch; labels qualifies / overrides / irrelevant by sentence id | ✅ tested with a scripted judge; not yet run with a real model |
| Q4.5 | Repeat to a fixpoint or the budget; fixpoint = COMPLETE | ✅ tested (fixpoint and budget stop); precedence clause joins on the first override |
| Q5 | Precedence: explicit clause > later amendment > specific over general; superseded text marked, never dropped | ✅ `certify/precedence.py`, tested (amendment supersedes; schedules-prevail rule; earlier-dated amendment flagged) |
| Q6.1 | Closure certificate: forward / reverse / sweep counts, precedence, COMPLETE or not | ✅ `certify/certificate.py`, tested |
| Q6.2 | Absence certificate: chapters read, canaries, miss rate and bound, certified level | ✅ tested |
| Q6.3 | Uncertified outcome with a reason | ✅ budget / unverified sweep / conflict / too few chapters / no calibration |
| Q6.4 | Certificates as readable text and JSON | ✅ `text()` and `to_dict()` |
| Q7.1 | Answer order: main clause → qualifications → overrides, all cited | 🟡 the answer prompt receives the roles note; order not yet checked on real answers |
| Q7.2 | Independent verifier checks every answer sentence against its quotes | ✅ `certify/verify.py`; an unsupported sentence makes the certificate uncertified; tested |

### 4.3 Trap benchmark

| ID | Requirement | Status |
|---|---|---|
| T1 | Far exceptions 10 / 100 / 1,000+ pages away (same contract, schedule, other contract) | ✅ built; structural recall 96/96 |
| T2 | Override chains (an amendment of an amendment) | ✅ built; 83/84, superseded marked |
| T3 | Unlinked qualifiers (no section number) | ✅ built; 84/84 offered to the qualifier worker |
| T4 | Precedence conflicts with an order-of-precedence clause | ✅ built; 84/84 offered |
| T5 | Existence yes / no, in one contract and room-wide | 🟡 one contract built ("no" = a clause type the contract truly lacks); room-wide not yet |
| T6 | Built by mutating real CUAD clauses; different seeds and texts from the calibration canaries | 🟡 templated now. Rule-based rewriting of real CUAD caps tried and rejected (3–6 usable of 672, broken grammar). Plan: model-rewritten real clauses, 1 call per trap, auto-checked (company + amount present, no other amounts) |
| T7 | Gold per question: answer, required qualifiers, correct certificate | ✅ |
| T8 | ≥ 300 questions per size tier (S, M; L where the budget allows) | 🟡 S 60, M 360; S to be raised, L not built |
| T9 | Data card and licences (CUAD CC BY 4.0, NarrativeQA Apache 2.0, Gutenberg public domain) | ⬜ |

## 5. Metrics

### 5.1 Certificate honesty (headline)

| Metric | Definition | Pass |
|---|---|---|
| Stated vs measured miss rate | canary-predicted miss rate vs the real miss rate on real clauses and planted facts | measured ≤ stated, within the 95% CI |
| False-certificate rate | "COMPLETE" with a missed required qualifier, or "absent" when present | ≤ 1 − α |
| Coverage at α | share of answers inside the k chapters opened | ≥ α |
| Certification rate | share of questions certified (not "uncertified") | reported; as high as honesty allows |

### 5.2 Answer quality

| Metric | Definition |
|---|---|
| Answer correctness | model judge, validated against human ratings |
| **Qualifier recall** (headline) | share of required exceptions / overrides / carve-outs the answer mentions |
| Override correctness | the right version wins (amendment, schedule, precedence) |
| Absence accuracy | correct vs wrong "not found" on existence questions |
| Evidence recall | gold sentences inside the evidence read |
| Citation precision / recall | ALCE-style |
| Faithfulness | share of answer sentences supported by their quotes |

### 5.3 Cost

Share of the document read; model calls, tokens and dollars per question; latency; all at
α = 90 / 95 / 98% (cost-of-certainty curves).

### 5.4 Components

Qualifier-graph link precision / recall (hand-checked sample); reverse-index recall on
planted far exceptions; ablation drops.

### 5.5 Human evaluation

Human-rated correctness and completeness (2 annotators, 50–100 answers); Cohen's κ;
model-judge agreement with humans.

### 5.6 Statistics

95% confidence intervals for every number: Clopper–Pearson for rates, bootstrap otherwise.
At least 300 questions per setting for the certificate metrics.

### 5.7 Metrics used by related work (for comparability)

| Line of work | Their metrics | We report too |
|---|---|---|
| Long-context (RULER, LongBench, Needle-in-a-Haystack, Lost in the Middle) | accuracy by length and position; F1 / ROUGE | accuracy by document size |
| RAPTOR, MemWalker, Chain of Agents | QuALITY accuracy, Qasper F1, NarrativeQA ROUGE | NarrativeQA judge score (+ ROUGE-L for reference) |
| RAG, GraphRAG | EM / F1, recall@k, MRR, LLM win rate | evidence recall, recall@k |
| ALCE, RAGAS | citation precision / recall, faithfulness | same |
| Conformal / abstention | coverage, set size, abstention rate, risk–coverage, AUROC, ECE | coverage, certification rate, risk–coverage |
| CUAD, ContractNLI, LegalBench | AUPR, precision at 80% recall, accuracy, F1 | clause evidence recall |

**New metrics nobody reports yet:** per-document stated-vs-measured miss rate,
false-certificate rate, qualifier recall, override correctness, cost of certainty.

## 6. Experiments

| ID | Research question | Experiment | Status |
|---|---|---|---|
| RQ1 | Do canaries predict the real miss rate on this document? | calibrate on canaries; measure on real CUAD clauses and planted facts at 100 / 1,000 / 10,000 pages | 🟡 with type canaries stated ≈ measured at 130 / 977 / 8,619 pages (gate level, no model); L has a library leak to fix with a held-out set; worker-level check still to do |
| RQ2 | Does closure find what others miss? | trap benchmark: ours vs the baselines (§7) | 🟡 first reading, traps S, n = 30: plain workers 11/30 → certified 30/30, certificates 30/30 right (READINGS) |
| RQ3 | Are certificates honest? | false-certificate rate vs α, with CIs | ⬜ |
| RQ4 | What does certainty cost? | cost curves at α = 90 / 95 / 98% | ⬜ |
| RQ5 | Which parts matter? | ablations (§8) | ⬜ |
| H1 | Do humans agree? | 2 annotators, κ, judge agreement | 🟡 pilot sheet ready, not rated |

**Decision after RQ1 / RQ3 (P3):** pass → full paper, ACL main · closure only → closure as
the main result, absence as analysis (main or Findings) · fail → analysis paper (Findings
or workshop).

## 7. Baselines

| ID | System | Status |
|---|---|---|
| B1 | Frontier long-context model with the whole document (S and M; L does not fit) | ⬜ needs budget |
| B2 | Strong RAG: BM25 + dense + reranker | 🟡 dense top-k exists; reranker ⬜ |
| B3 | Chain-of-Agents-style: sequential chunk workers + manager (re-implemented) | 🟡 `baselines/chain_of_agents.py` + `--system coa`, tested; not yet run (reads every chunk: ~12 calls / question at 130 pages) |
| B4 | Ours without closure / certificates (current workers) | ✅ |
| B5 | Ours, full | ✅ `--system workers_certify`; run on traps S (30 q) |
| B6 | A second answer model (closed or open) | ⬜ |

## 8. Ablations

| ID | Removed part | Status |
|---|---|---|
| A1 | reverse index | ⬜ |
| A2 | whole-document sweep | ⬜ |
| A3 | forward closure | ⬜ |
| A4 | canaries (fixed k instead) | ⬜ |
| A5 | mutation canaries (template canaries instead) | ⬜ |
| A6 | precedence step | ⬜ |
| A7 | GNN gate (lexical gate instead) | ⬜ |

## 9. Paper deliverables

| ID | Item | Status |
|---|---|---|
| D1 | Idea, method and plan documented | ✅ |
| D2 | ACL LaTeX template set up (from the GNN PRUNER paper) | ✅ |
| D3 | Related-work table from a systematic search (P0) | ⬜ blocked: web search needs your permission or your own search (see chat) |
| D4 | Formal definitions of "complete" and "absent" | ⬜ |
| D5 | Figures: system diagram, certificate examples, stated-vs-measured plot, cost curves, trap results table | ⬜ |
| D6 | Full draft: intro (BETA example), related work, problem, method, benchmark, experiments, limitations, ethics | ⬜ |
| D7 | Limitations: exchangeability assumption, structure quality, cost of the long-context baseline | ⬜ |
| D8 | Ethics: legal-advice disclaimer, data licences, AI-assistance statement | ⬜ |
| D9 | Code + benchmark + data card, anonymised for review | ⬜ |
| D10 | ARR responsible-NLP checklist; submission; arXiv preprint when allowed | ⬜ |

## 10. Timeline (from mid-November 2026)

| Weeks | Phase | Output |
|---|---|---|
| 1 | P0 literature search | related-work table, novelty go / no-go |
| 1–3 | P1 build the certificates | U2–U4, Q1–Q7 with tests |
| 2–3 | P2 trap benchmark | T1–T9 |
| 4 | P3 validity (make-or-break) | RQ1, RQ3, decision |
| 5–6 | P4 baselines | RQ2, RQ4 |
| 4–6 | P5 human evaluation | H1 |
| 6 | P6 ablations | RQ5 |
| 7–9 | P7 write and submit | paper, artefacts |

About 8–9 weeks of focused work, leaving 3–4 weeks of slack before the February 2027 ARR
deadline.

## 11. Tasks for the team

| ID | Task | When | Status |
|---|---|---|---|
| H-1 | Say go for P0 + P1 | now | ⬜ |
| H-2 | Read the 8–10 closest papers | week 1–2 | ⬜ |
| H-3 | Approve about $200–400 API budget for B1 / B6 | before week 5 | ⬜ |
| H-4 | Find 2 annotators (law students ideal) | by week 4 | ⬜ |
| H-5 | Rate the pilot sheet `docs/fact_check_sheet.md` | week 2 | ⬜ |
| H-6 | Ask the supervisor (Dr. Sumit Kumar Gupta) to review and co-author | now | ⬜ |
| H-7 | Agree the author order | before the draft | ⬜ |
| H-8 | Create the new GitHub repository and send the URL | now | ⬜ |
| H-9 | Review and edit the draft | weeks 7–9 | ⬜ |

## 12. Overall status

- Foundation (reused from GNN PRUNER): about 60% of the system, done.
- New research parts: about 5% (first U2 / U3 code with tests, documentation).
- Experiments, benchmark, baselines, human evaluation, paper: not started.
