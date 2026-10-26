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
