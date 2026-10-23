# GNN PRUNER: Project Workflow

GNN PRUNER answers questions about very large documents: contracts, books and reports,
of any size, including ones far too long for any AI model to read at once. It never
gives the whole document to an AI model. It turns the document into a tree, searches
that tree chapter by chapter, keeps only word-for-word quotes that it has checked against
the original, and has an AI model answer from those quotes alone. Every answer comes with
page and section citations, or with "The document does not say."

Status markers: ✅ built and tested · 🟡 partly built · 🔲 to build

---

## The workflow at a glance

```
 ┌───────────────────────────── A. UPLOAD: once per document ─────────────────────────────┐
 │  A1 Read the file ─► A2 Find the structure ─► A3 Build the tree ─► A4 Fill node data    │
 │                                                                     ─► A5 Index & store │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │ document tree saved on disk
                                        ▼
 ┌───────────────────────────── B. QUESTION: every query ─────────────────────────────────┐
 │  B1 Understand the question ─► B2 Central board sends it to every chapter worker        │
 │  B3 Each worker: relevance check ─► (if relevant) search its chapter step by step       │
 │  B4 Workers post quotes ─► B5 Board checks, merges, follows references, decides         │
 │  B6 Evidence pack ─► B7 AI model writes the answer ─► B8 Citations checked ─► user      │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │ accepted quotes + user feedback
                                        ▼
 ┌───────────────────────────── C. LEARN: continuously ───────────────────────────────────┐
 │  C1 Collect labels ─► C2 Retrain the relevance model ─► C3 Safety check ─► C4 Deploy     │
 └─────────────────────────────────────────────────────────────────────────────────────────┘

 D. MEASURE: every change is scored on fixed test sets and logged in docs/READINGS.md
```

---

## A. Upload: turning a document into a tree (once per document)

### A1. Read the file ✅
- **Formats:**
  - PDF (PyMuPDF; Docling for complex layouts)
  - Word, HTML, Markdown and plain text
- **Positions:** every piece of text keeps its **page number and position**, so answers
  can cite exact pages.
- ✅ **Cleaning:**
  - split words rejoined
  - ligatures (ﬁ → fi), invisible characters and curly quotes fixed
  - page headers / footers removed
  - drop caps ("I" + "t was") rejoined
  - indented paragraphs separated
- 🔲 **Scanned PDFs** need text recognition (OCR) first.

### A2. Find the structure ✅
The chapters and sections are found from, in this order:
1. ✅ the PDF's own table of contents / bookmarks: entries are matched to the text (also
   headings split over two lines), and "Book / Part" entries contain their chapters
2. ✅ headings and numbering ("Chapter 3", "Article V", "5.2", "(b)")
3. ✅ font size and style
4. ✅ if there is no structure at all: **virtual chapters** of about 25 pages (at most
   40), cut between paragraphs. Schedules, annexes, exhibits and appendices always
   become chapters.

### A3. Build the tree ✅
| Level | Book | Contract | Holds |
|---|---|---|---|
| **Root** | the book | the agreement | metadata + overall summary |
| **Chapter** | chapters | articles / parts | title, pages, summary |
| **Section** | sections | sections (5, 5.2) | title, summary |
| **Subsection** | subsections | clauses (5.2(a)) | key facts |
| **Text** | paragraphs / sentences | sentences | the **original text** |

**Links between nodes:**
- ✅ built:
  - parent / child
  - reading order
  - same section
  - cross-references ("see Section 4.2")
  - defined terms → their definitions
  - similar meaning
- 🔲 to build: the same character / party / concept across chapters (entity links, for
  books)

Missing levels get placeholder nodes, so every path has the same depth. Subsections
longer than ~2,000 tokens are sliced into ~1,200-token pieces. Every sentence has an ID, so
quotes are taken from the text by ID, never retyped. Code: `tree.py`.

### A4. Fill the node data ✅
Code: `metadata.py`, `enrich.py`; run with `python -m gnn_pruner.upload <file> --llm <model>`.
An AI model fills the tree **from the bottom up**:
1. **Subsections:** key facts (numbers, dates, names, obligations, kept exactly as
   written), several subsections per call
2. **Sections:** summaries built from their subsections' key facts
3. **Chapters:** summaries built from their section summaries
4. **Root:**
   - an overall summary
   - **metadata:**
     - books: title, author, year, publisher, ISBN
     - contracts: parties, date, governing law

Rules:
- A metadata value (an ISBN, a year) is kept **only if its text appears in the
  document**.
- Summaries and key facts are **signposts for finding things. They are never quoted as
  evidence.**
- Every key fact cites the IDs of the sentences it comes from. A fact citing unknown IDs,
  or a number its sentences don't contain, is dropped.
- Facts are typed (obligation, deadline, amount, definition, party, person, event, …) and
  gathered into **lists** for lookup questions.
- This runs once per document and is cached, so uploading the same file again is
  instant. A run stopped by a quota continues where it left off.

### A5. Index and store ✅
- **Search indexes:** keyword search (BM25) and meaning search (MiniLM vectors), built
  locally.
- **Relevance model:** the graph neural network's inputs are prepared.
- **Storage:** everything is saved as Parquet files on disk, so even a 10,000-page
  document is never fully loaded into memory.

---

## B. Question: from question to cited answer (every query)

### B1. Understand the question ✅
The question is turned into a plan:
- **question type:** definition, number, yes/no, comparison, summary, timeline
- **sections it names:** "Section 7.2", "Chapter 4"
- **defined terms and entities**
- **what kind of answer is expected:** a number, a date, a list or an explanation

### B2. Central board ✅
A shared record for this one question. It:
- sends the question to every chapter worker
- collects their posts
- tracks each worker's state
- tracks the AI-call budget
- decides when there is enough evidence

### B3. Chapter workers: one per chapter 🟡
Every chapter (or article) has its own worker.

**Step 1: relevance check.** This step uses no AI model.
- The worker scores its chapter with keyword search, meaning search and the trained
  **graph neural network**. ✅ All three are built and trained.
- One shared AI call reads the question and the table of contents (all chapter titles and
  summaries) and marks the chapters worth opening. 🔲
- **The worker opens its chapter if** its score is among the top few, **or** the table
  of contents screen marks it, **or** the question names it. Otherwise it reports "not
  relevant" at no cost.

**Step 2: search the chapter.** This step uses a reasoning AI model. ✅ (`workers/worker.py`)
- The worker moves down its own branch, one step at a time, at most 6 steps:
  - **open a section** and see its subsections and key facts
  - **read a passage** of original text
  - **search** inside the chapter
  - **post** a quote to the board
  - **ask the board to follow a reference** ("subject to Section 12.1")
  - **finish**
- It always sees the most promising sections first, ranked by the relevance model.
- Very long chapters are split across several workers. ✅ The splitting is built.
- Workers run at the same time.

### B4. What a worker posts ✅
Each post contains:
- the exact quote
- the node it came from
- why it matters
- which part of the question it answers
- any references to follow

### B5. Board processing 🟡
1. **Check:** workers cite sentence IDs; unknown IDs are rejected and the quote is taken
   from the text by ID. ✅
2. **Locate:** each quote is pinned to its page and chapter › section path. ✅
3. **Merge** overlapping quotes. ✅
4. **Follow references:** wake the worker for a referenced chapter (by ID, title, or
   free text looked up across the document), at most 2 hops. ✅
5. **Add required context:** the definitions of terms used, and the exceptions and
   conditions that change a clause's meaning. ✅
6. **Mark conflicts:** two clauses that disagree ("30 days" vs "45 days") are both kept
   and flagged. ✅
7. **Decide:**
   - **enough evidence:** go to B6
   - **nothing found:** answer "The document does not say". This uses both retrieval
     confidence ✅ and the AI model's own judgement ✅.
   - **budget reached:** answer with what was found, marked partial

### B6. Evidence pack ✅
- **Labels:** accepted quotes are labelled E1, E2 …, each with its page and section.
- **Prompt shape:** chosen from how the quotes relate: single fact, nested path, several
  related clauses, or a synthesis across chapters.
- **Size:** trimmed to about 250–300 tokens.

### B7. AI model writes the answer ✅
- **Input:** the question and the evidence pack only.
- **Output:** an answer, the labels it relies on, a confidence, and a flag when the
  evidence is not enough.
- **Providers:** Claude, OpenAI or Gemini (free tier used for development), or no-AI
  extractive mode.

### B8. Citation check and output ✅
- **Valid labels:** every cited label must exist in the pack. Bad citations are flagged,
  and strict mode refuses instead.
- **What the user receives:** the answer, the quoted text, **page numbers and chapter ›
  section** for every citation, and a confidence.

---

## C. Learn: getting better with use

| Step | What happens | Status |
|---|---|---|
| C1 Collect labels | Every quote the board accepts is a label: "for this question, this passage held the answer". User feedback (wrong answer, missed clause) adds more. | 🔲 |
| C2 Retrain | The relevance model (graph neural network) is retrained on these labels. The pipeline exists: automatic first labels ("Silver Paths") → supervised training → reinforcement fine-tuning (PPO) → coarse-to-fine scoring → cross-chapter attention. | ✅ pipeline, 🔲 feedback labels |
| C3 Safety check | A new model must not score worse than the current one on a fixed test suite (≤ 2% drop) before it replaces it. | 🔲 |
| C4 Deploy | The better model is used for B3's relevance check. A better check opens fewer chapters, so questions get cheaper and faster. | 🔲 |

---

## D. Measure: proving it works

- **Test sets:**
  - ✅ real contracts (short and 10-page versions)
  - ✅ generated contracts with known answers
  - ✅ adversarial questions
  - ✅ speed tests at 100 → 10,000 pages
- 🔲 **Long-document test:** contracts and books at 100, 1,000 and 10,000+ pages, with
  answers hidden deep inside, two-chapter questions, and questions with no answer.
- 🔲 **Head-to-head on the long-document test:**
  1. a long-context AI model given the whole document (only possible up to about
     1,500–2,500 pages)
  2. plain search + AI
  3. relevance model only
  4. workers on every chapter
  5. the full system
- **Measured:**
  - correct answers
  - correct numbers
  - citation accuracy
  - correct "does not say"
  - AI calls, cost and time per question
- **Rules:**
  - Every result is recorded in `docs/READINGS.md`.
  - Test sets are never used for tuning.
- **What proves the goal:**
  - beyond the largest AI model's limit, only this system can answer, with correct
    citations
  - below it, this system matches or beats the AI model at a small fraction of the cost

---

## Risks and solutions

These are the known weak spots in the workflow and how each one is handled. They are part
of the build order below. None is optional.

### R1. The relevance check skips the chapter that holds the answer
If B3 never opens the right chapter, nothing later can recover it, and the system wrongly
says "The document does not say".
- **Score each chapter by its best passage, not its average.** One key clause in a
  40-page chapter is not drowned out.
- **Any signal can open a chapter.** Keyword, meaning, the relevance model and the
  table-of-contents screen each vote, and one strong vote is enough.
- **Widen before giving up:**
  1. **Round 1:** the top few chapters.
  2. **Round 2:** if nothing is found, the next few.
  3. **Round 3:** an AI model rewrites the question in the document's own words ("fire"
     → "terminate for cause") and the search runs again.

  Only then "does not say".
- **Measured:** share of answer chapters opened (target ≈ 98% by round 2).

### R2. Whole-document questions have no single passage to quote
"Summarise the book", "list every obligation of the Supplier" and "how does X change over
the story" are not answered by one quote.
- **Classify scope first (B1):** single fact / multi-part / whole document.
- **Whole-document route:**
  - each chapter worker writes a short partial answer **with quotes attached**
  - the board combines them level by level (chapters → parts → book)
  - every sentence of the final answer must point to a checked quote
- **Precomputed lists at upload (A4):** every obligation, deadline, amount, definition
  and party, each with its quote. List questions become lookups with almost no AI calls.

### R3. Exact quotes break on real PDFs
Hyphenated line breaks, special characters (ﬁ), page headers / footers and scanned pages
make retyped quotes fail to match.
- **Cite by ID, not by retyping.** Workers point at sentence IDs (plus an optional word
  range), and the system extracts the quote itself from the original. A quote can never
  be mistyped or invented, and always maps to its exact page.
- **Clean the text at upload (A1):**
  - rejoin split words
  - fix special characters
  - normalise quotes and dashes
  - remove repeated page headers / footers
- **Backup:** near-exact matching on cleaned text when a model retypes a quote, mapped
  back to the original span.

### R4. The relevance model is weak on real documents
It was trained on generated contracts and currently trails plain keyword-and-meaning
search on real ones.
- **R1's "any signal can open" rule:** a weak model cannot hide a chapter.
- **Train and test on real documents:**
  - **CUAD:** ~500 real contracts with clauses labelled by lawyers
  - **NarrativeQA:** questions over full books
  - **QuALITY:** long articles

  Licences are checked before use.
- **Earned trust:** the model's weight in B3 rises only after it beats plain search on
  held-out real documents.

### R5. Schedules, annexes and amendments
In contracts, the real numbers often sit in "Schedule 3", and a later amendment can
override a clause.
- **Schedules as chapters:** Schedules, Annexes, Exhibits and Appendices become chapters;
  "Schedule 3" / "Exhibit B" references are linked.
- **Contract families:** the main agreement and its amendments are uploaded together as
  one tree.
- **Amendment links:** "Section 5.2 is deleted and replaced with…" creates a *changed
  by* link with its date. When the board accepts an old clause, it pulls in the
  amendment and marks the old text **replaced**.
- **Priority:** an "order of precedence" clause is applied where present; otherwise newer
  text wins, and the answer says so.

### R6. Hidden instructions inside the document
A document may contain text such as "ignore previous instructions…".
- **Data, not instructions:** document text is always wrapped and labelled as data, and
  models are told never to follow instructions inside it.
- **Read-only workers:** workers can only open, read, search and post inside the document.
  The worst case is a misdirected search, and ID-based quotes and the citation check stop
  false evidence.
- **Tested:** documents with hidden instructions are part of the adversarial test set.

### R7. The learning loop only learns from chapters it opened
Labels come only from chapters that were opened, so blind spots repeat.
- **Free "missed" labels:** when round 2 or 3 finds the answer in a chapter round 1
  skipped, that is a label.
- **Exploration:** 5–10% of questions open 1–2 extra low-ranked chapters.
- **Periodic full sweeps:** a sample of questions searches every chapter with a cheap
  model, offline.
- **Strong labels from users:** "it missed this clause".

### R8. Speed and cost
10–25 AI calls per question, several in sequence, would be slow and costly.
- **Fewer steps:** each worker gets its section list and its best passages at once, so
  most finish in 1–2 calls instead of 6.
- **Model sizes:** cheap models for workers; the strongest only for the final answer.
- **No-worker answers:** R2's precomputed lists answer common lookups directly.
- **Parallel and cached:** workers run in parallel; repeated questions on the same
  document are cached; upload work runs as discounted batch jobs.
- **Expected:** usually 4–8 AI calls and 5–15 seconds per question.

### R9. The evidence limit is too small for books
A fixed ~300 tokens suits one contract clause, not a question spanning five chapters.
- **Limit by question type:**

  | Question type | Evidence limit (tokens) |
  |---|---|
  | single fact | ~300 |
  | multi-clause | ~600 |
  | across chapters | ~1,500 |
  | whole document | combined level by level (R2) |

- **Still as short as possible:** shorter evidence was preferred in the human ratings
  whenever it was sufficient.

---

## Running on this machine

- **Laptop:** 8 GB RAM. The laptop runs only reading, search and the relevance model
  (about 0.5–1.5 GB).
- **AI models:** always hosted (Gemini, Claude, OpenAI), never run locally, because local
  models froze the laptop.
- **Keys:** stored in `~/.zshrc` as `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` or
  `OPENAI_API_KEY`. Never put keys in the project or in chat.

Commands that work today:

```zsh
# Ask a question about a document
python -m gnn_pruner examples/saas_agreement.md "Can the Provider increase fees on renewal?" \
    --llm gemini:gemini-3.1-flash-lite

# Score answers on the test sets (rows saved as they finish; reruns continue)
python -m gnn_pruner.answer_eval --sets eval/contracts_v1.jsonl eval/contracts_scale_v1.jsonl \
    --llm gemini:gemini-3.1-flash-lite --resume eval/runs/<name>.jsonl
```

---

## Build order from here

| Step | What | Risks solved | Done when |
|---|---|---|---|
| 1 | Text cleaning, table-of-contents structure, virtual chapters, schedules as chapters, metadata, bottom-up summaries and key facts, precomputed lists (A1–A4) | R3, R5 (part), R2 (lists) | a real PDF book becomes a full tree with checked metadata |
| 2 | Long-document test set with real documents (CUAD, NarrativeQA, QuALITY) + generated ones, first three baselines (D) | R4 (testing) | results logged for 100 → 10,000+ pages |
| 3 | Central board + chapter workers with cite-by-ID, widening rounds, model-judged "not found", hidden-instruction guard (B2–B5) | R1, R3, R6 | workers on every chapter measured |
| 4 | Relevance check (best-passage scores, any-signal opening, TOC screen) + whole-document route + question-type evidence limits + head-to-head (B3, D) | R1, R2, R8, R9 | full system measured against all baselines |
| 5 | Contract families and amendments; entity links for books (A3) | R5 | amendment and two-chapter questions measured |
| 6 | Learning loop with missed labels, exploration and sweeps; retrain on real documents (C) | R4, R7 | AI calls per question go down; relevance model beats plain search on real documents |
| 7 | Production: web API, dashboard, regression suite | — | the system runs as a service |

Detailed design: `docs/ARCHITECTURE.md`. Checklists and results: `REQUIREMENTS.md`,
`docs/READINGS.md`.
