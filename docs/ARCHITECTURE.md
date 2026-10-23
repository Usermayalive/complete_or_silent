# GNN-Pruner v4: Architecture

**Goal:** answer questions about contracts and books of any size, even ones far larger than
any AI model can read at once. Every answer is built only from text found in the document,
and every quote is checked against the original before it is used.

**Core idea:**
- The document becomes a tree: book → chapters → sections → subsections → original text.
- One worker is assigned to every chapter.
- Each worker reads the question, searches its own chapter, and posts what it finds to a
  central board.
- The board checks every post against the original text.
- A final LLM answers only from what the board accepted, citing pages and sections.

Status markers: ✅ built and tested · 🟡 partly built · 🆕 to build.

---

## 1. Overview

```
╔══════════════════════════════ UPLOAD (once per document) ═══════════════════════════════╗
║                                                                                           ║
║  PDF ─► [1] Parse ─► [2] Detect structure ─► [3] Build tree ─► [4] Enrich ─► [5] Index    ║
║         text+pages    TOC / headings /        L0 book            summaries,     search,   ║
║                       numbering               L1 chapters        key facts,     vectors,  ║
║                                               L2 sections        metadata,      GNN,      ║
║                                               L3 subsections     links          storage   ║
║                                               L4 original text                            ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
                                              │
                                              ▼   document tree on disk (Parquet, cached by file hash)
╔══════════════════════════════ QUESTION (every query) ════════════════════════════════════╗
║                                                                                           ║
║  question ─► [6] Plan ─► [7] CENTRAL BOARD  ◄──────────── posts ───────────┐              ║
║                               │ broadcasts the question                    │              ║
║                               ▼                                            │              ║
║               ┌──────────┬──────────┬──────────┬─ ··· ─┬──────────┐        │              ║
║               │ Worker   │ Worker   │ Worker   │       │ Worker   │        │              ║
║               │ Ch 1     │ Ch 2     │ Ch 3     │       │ Ch N     │  [8] one per chapter │
║               └────┬─────┴────┬─────┴────┬─────┴─ ··· ─┴────┬─────┘        │              ║
║                    │ gate: is my chapter relevant? (free, local)           │              ║
║                    │ if yes: descend sections → subsections → text (LLM)   │              ║
║                    └───────────────────────────────────────────────────────┘              ║
║                                                                                           ║
║  [9] Board: verify quotes · merge · follow references · add definitions · detect          ║
║             conflicts · decide "enough evidence?"                                         ║
║                               │                                                           ║
║                               ▼                                                           ║
║  [10] Evidence pack ─► [11] Answer LLM ─► [12] Citation check ─► answer + pages + sections║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
                                              │
                         [13] Learning loop: accepted posts become training labels
                              for the worker gate (GNN), so it gets cheaper over time
```

---

## 2. The document tree

### 2.1 Layers

| Layer | Book | Contract | What the node holds |
|---|---|---|---|
| **L0 root** | the book | the agreement | metadata + whole-document summary |
| **L1** | chapters | articles / parts | title, page range, chapter summary |
| **L2** | sections | sections (5, 5.2) | title, section summary |
| **L3** | subsections | clauses (5.2(a)) | key facts: names, numbers, dates, obligations, conditions |
| **L4** | paragraphs / sentences | sentences | **original text**, the only thing ever quoted |

**Missing levels:**
- A document without a level gets a virtual node. For example, a chapter with no
  sections gets one virtual section, so every path has the same depth. ✅ (the "#0" nodes
  in `layers.py`)
- A document with no chapters at all (one long report) is split into **virtual
  chapters** of about 20–40 pages each, at the nearest heading. 🆕

### 2.2 Node record

```json
{
  "id": "ch07.s03.ss02",
  "layer": 3,
  "kind": "subsection",
  "number": "7.3(b)",
  "title": "Late Payment",
  "path": ["Book", "7. Payment", "7.3 Invoices", "7.3(b) Late Payment"],
  "pages": [212, 213],
  "span": [481233, 482010],
  "summary": null,
  "key_facts": ["Interest of 1.5% per month on overdue amounts", "Starts 30 days after invoice"],
  "entities": ["Customer", "Supplier"],
  "defined_terms": ["Overdue Amount"],
  "children": ["ch07.s03.ss02.p1", "ch07.s03.ss02.p2"],
  "links": {"refers_to": ["ch12.s01"], "defines": [], "similar_to": ["ch09.s02.ss01"]},
  "tokens": 310
}
```

- **Text:** L4 nodes hold `text`.
- **Summaries:** L1–L2 nodes hold `summary`.
- **Key facts:** L3 nodes hold `key_facts`.
- **Root metadata:** the root holds `metadata`:
  - books: title, author(s), year, edition, publisher, ISBN
  - contracts: parties, effective date, governing law

### 2.3 Links between nodes

| Link | Example | Status |
|---|---|---|
| parent / child | Chapter 7 contains Section 7.3 | ✅ |
| reading order | 7.3(a) → 7.3(b) | ✅ |
| same section | siblings under 7.3 | ✅ |
| cross-reference | "subject to Section 12.1" → 12.1 | ✅ |
| definition | uses of "Overdue Amount" → its definition | ✅ |
| similarity | passages with similar meaning, anywhere | ✅ |
| **entity** | every mention of the same character, party or concept | 🆕 (needed for books) |

---

## 3. Upload pipeline (once per document)

### [1] Parse ✅
- **Readers:**
  - PyMuPDF for normal PDFs
  - Docling for complex layouts (columns, tables)
  - Word, HTML, Markdown
- **Positions:** every character keeps its page number and offset, so citations can name
  exact pages.
- 🆕 **Scanned PDFs:** OCR (text recognition) first. Later phase.

### [2] Detect structure 🟡
Sources, tried in order:
1. 🆕 the PDF's own **table of contents / bookmarks** (most books and long reports have them)
2. ✅ **headings and numbering** ("Chapter 3", "Article V", "5.2", "(b)")
3. ✅ font size and style for PDFs without either
4. 🆕 **virtual chapters** for documents with no usable structure (§2.1)

### [3] Build the tree ✅
- 4-layer DAG with virtual nodes and links (`layers.py`, `document_graph.py`).
- Checked by `validate_and_repair`: every node has one parent, the layers are in order,
  and there are no cycles.

### [4] Enrich 🆕
An LLM fills the node data **bottom-up**, so each summary rests on the level below:

| Step | Input | Output | Calls (1,000-page book) |
|---|---|---|---|
| a | L4 text under each L3 node | L3 **key facts** (exact numbers and names kept) | ~100–200 (10–20 subsections per call) |
| b | L3 key facts per section | L2 **section summary** | ~30–60 |
| c | L2 summaries per chapter | L1 **chapter summary** | ~1 per 5 chapters |
| d | L1 summaries + first pages | L0 **summary + metadata** | 1–2 |

**Metadata** (author, year, ISBN, publisher):
1. read from the PDF's own metadata fields
2. read from the title and copyright pages
3. extracted by the LLM, which must also return the exact text it found

An ISBN or year is kept **only if that text exists in the document**. No guessing.

**Rules:**
- Summaries are signposts for navigation only. **They are never quoted as evidence.**
- Key facts must keep numbers, dates and names exactly as written.
- Enrichment runs once. It is cached by the file's hash, so a re-upload is instant.

### [5] Index and store ✅
- **Keyword search:** BM25.
- **Vector search:** MiniLM vectors, built locally (small; fits 8 GB RAM).
- **GNN node features:** precomputed.
- **Storage:** Parquet shards on disk. A 10,000-page document is never fully loaded into
  memory.

---

## 4. Question pipeline (every query)

### [6] Plan ✅
Turns the question into a plan:
- **question type:** definition, number, yes/no, comparison, summary, timeline
- **named sections:** "Section 7.2", "Chapter 4"
- **entities and defined terms**
- **what kind of answer is expected:** a number, a date, a list, an explanation

### [7] Central board 🆕
A shared record for one question, sometimes called a blackboard:

```json
{
  "question": "...",
  "plan": {...},
  "workers": {"ch01": "skipped", "ch07": "searching", "ch12": "done"},
  "posts": [...],
  "accepted": [...],
  "rejected": [{"post": "...", "reason": "quote not found in node text"}],
  "pending_refs": ["ch12.s01"],
  "budget": {"llm_calls_used": 7, "llm_calls_max": 20},
  "status": "collecting | enough_evidence | not_found | budget_reached"
}
```

The board sends the question to every chapter worker and collects their posts. Only
accepted posts go on to the answer.

### [8] Chapter workers: one per chapter 🆕

Every L1 node (chapter or article) gets a worker. Each worker works in two stages, so a
book with 300 chapters does not cost 300 LLM calls.

**Stage 1: gate (is my chapter relevant?).** This stage is free.
- **Local score:** the chapter's text is scored with no LLM: keyword match + vector match
  + GNN score (the existing pruner).
- **Table-of-contents screen:** one LLM call for the whole book reads the question and
  all chapter titles and summaries together, and marks the chapters worth opening. This is
  the PageIndex-style "reason over the table of contents" step.
- **The worker opens its chapter if** its local score is in the top few, **or** the
  screen marks it, **or** the question names it.
  - The local score catches exact facts that a summary may leave out.
  - The screen catches meaning that keyword search may miss.
- **Otherwise** the worker posts "not relevant" and stops (no LLM call).

**Stage 2: descend (reasoning model).** Only workers that pass the gate run this.

The worker moves down its own branch of the tree in a short loop (at most 6 steps). It
sees only small pieces at a time:

```
sees:    question + chapter summary + list of its sections (title + summary)
action:  open("7.3")              → sees subsections of 7.3 with their key facts
action:  read("7.3(b)")           → sees the original text of 7.3(b)
action:  search("late interest")  → keyword/vector search inside this chapter only
action:  post(finding)            → sends a quote to the board
action:  follow("12.1")           → asks the board to check a referenced section
action:  done("nothing relevant") / done("posted")
```

**Search order:** the worker's view is pre-sorted by the local pruner (most promising
sections first). It reasons over the structure instead of reading the whole chapter.

**What a worker posts:**

```json
{
  "worker": "ch07",
  "findings": [
    {"node_id": "ch07.s03.ss02.p1",
     "quote": "interest at 1.5% per month on any Overdue Amount",
     "why": "states the late-payment interest rate",
     "answers_part": "rate"}
  ],
  "follow_refs": ["ch12.s01"],
  "confidence": "high"
}
```

**Very long chapters** (more than about 40 pages) are split into sub-workers by section,
so no worker ever reads more than a small window. The G5 partitioning (✅) already
produces balanced units.

### [9] Board processing 🆕, reusing ✅ checks

The board processes posts as they arrive:

1. **Verify:**
   - the quote must exist **word for word** in that node's original text (spacing and
     quote marks normalised)
   - if it does not, the post is rejected and logged
   - this removes any made-up evidence
2. **Locate:** each accepted quote is mapped to its L4 node, page number and
   chapter › section path.
3. **Merge:** overlapping quotes from different workers are combined.
4. **Follow references:**
   - a `follow_refs` request whose chapter is not already searching wakes that chapter's
     worker, aimed at the named section
   - at most 2 hops, within the call budget
5. **Add required context** ✅ (dependency rules from Phase 6):
   - the definition of every defined term used in an accepted quote
   - "subject to" / "except" / proviso clauses that change its meaning
   - the parent heading, for orientation
6. **Detect conflicts** ✅: two accepted quotes that disagree ("30 days" vs "45 days") are
   both kept and marked, so the answer can explain the difference.
7. **Decide when to stop:**
   - **enough evidence:** every part of the plan has an accepted quote and the confidence
     signal ✅ (Phase 7) is high
   - **not found:** all workers are done and nothing was accepted. The answer is "The
     document does not say", with no guessing.
   - **budget reached:** the system answers with what it has and marks it as partial

### [10] Evidence pack ✅
- **Prompt shape:** chosen from how the accepted quotes relate (Phase 9):
  - single fact
  - path through nested sections
  - several siblings
  - synthesis across distant chapters
- **Labels and budget:** quotes are labelled E1, E2 … with page and section, and trimmed
  to a token budget (~250–300 tokens).

### [11] Answer LLM ✅
- **Input:** the question and the evidence pack only.
- **Output:** a fixed JSON format:
  ```json
  {"answer": "...", "evidence": ["E1", "E3"], "confidence": "high", "insufficient": false}
  ```
- **Rules:** answer only from the evidence, cite every claim, and say "insufficient" when
  the evidence does not answer the question.

### [12] Citation check ✅
- **Valid labels:** every cited label must be in the pack. Invalid labels are flagged,
  and strict mode turns a flagged answer into "The document does not say".
- **What the user sees:** the answer, citations with **page numbers and chapter ›
  section**, the quoted text, and the confidence.

### [13] Learning loop 🆕 (Phase 11)
- **Training labels:** each accepted board post is a free label ("for this question,
  this passage was the answer").
- **Retraining:** the labels retrain the worker gate's GNN scorer.
- **Payoff:** as the gate gets better, fewer chapters need an LLM, so the system gets
  cheaper and faster with use.
- **User feedback** (wrong answer, missed clause) feeds the same loop.

---

## 5. Where each part of the existing work fits

| Existing work | Role in v4 |
|---|---|
| Parser, 4-layer DAG, links, Parquet (Phases 1–4, G1) | the document tree |
| BM25 + vectors + query planner (Phases 3–5) | worker gate (local score) and in-chapter `search` |
| CoarseGNN / FineGNN / attention (G2–G4) | worker gate scoring and ordering what a worker sees first |
| Partitioning + Ray workers (G5) | splitting long chapters; running gates in parallel on huge documents |
| Dependency rules, conflict index (Phase 6) | board steps 5–6 |
| Confidence signal (Phase 7) | board stopping rule, abstention |
| Evidence pack + routing (Phase 9) | step 10 |
| Providers, citation validation, readings (Phase 10) | steps 11–12; all LLM calls (Claude, OpenAI, Gemini) |

---

## 6. Running it

### 6.1 Parallelism and memory
- **Workers:** run at the same time as parallel network calls (async). The LLMs are
  hosted, never local.
- **The laptop runs only:**
  - parsing
  - search indexes
  - the GNN

  That is about 0.5–1.5 GB of memory, even for a 10,000-page document.
- **Huge documents** (thousands of chapters): gates run in Ray worker processes over
  Parquet shards (G5).

### 6.2 Models

| Job | Model type | Examples |
|---|---|---|
| Enrichment (key facts, summaries, metadata) | cheap, fast | Gemini Flash-Lite, Claude Haiku 4.5 |
| Table-of-contents screen, chapter workers | cheap reasoning | Gemini Flash-Lite / Flash, Claude Haiku 4.5 |
| Final answer | strongest affordable | Claude Sonnet / Opus 5.5, Gemini Pro |

Any provider can fill any role through the same `Provider` interface (`answer.py`).

### 6.3 LLM calls per question

| Step | Calls |
|---|---|
| Table-of-contents screen | 1 |
| Workers that pass the gate (typically 3–6) × up to 6 steps | ~8–20 |
| Reference follow-ups | 0–4 |
| Final answer | 1 |
| **Total** | **~10–25** (hard cap set by the board's budget) |

- **Upload** costs ~150–300 calls once for a 1,000-page book.
- ⚠️ **Free tiers** (e.g. Gemini's 20 calls per day for the main Flash model) are enough
  for testing only. Real use needs a paid key or a model with a larger free allowance.

---

## 7. Guarantees and risks

| Risk | Protection |
|---|---|
| Model invents evidence | board accepts only word-for-word quotes from the original text |
| Summary leaves out the key fact, so the chapter is skipped | gate also uses the local score on the original text, not just summaries |
| Answer spread over distant chapters | several workers post; references are followed; SYNTHESIS prompt |
| Contradictory clauses | conflict detection keeps and marks both |
| Answer not in the document | "The document does not say" instead of a guess |
| Cost runs away | board budget caps LLM calls per question |
| Wrong citation in the final answer | citation check; strict mode refuses |
| Bad structure detection | TOC first, then headings, then virtual chapters |

---

## 8. How we prove it works

**Long-document benchmark** 🆕:
- contracts and books at 100, 1,000 and 10,000+ pages
- known-answer questions placed deep inside, including questions that need two distant
  chapters
- unanswerable questions included

**Five systems compared on the same questions:**
1. a long-context LLM given the whole document (only possible up to ~1M tokens, about
   1,500–2,500 pages)
2. plain vector search + LLM
3. GNN pruner only (what is built now)
4. workers on every chapter with no gate
5. **v4: gated chapter workers + central board**

**Measured:**
- correct answers (LLM judge + number match)
- citation accuracy
- correct "does not say"
- LLM calls
- cost
- time

**What would prove the goal:**
- **beyond ~1M tokens:** system 1 cannot run at all, and v4 still answers correctly with
  citations
- **below that size:** v4 matches or beats it at a small fraction of the cost

---

## 9. Code layout (planned)

```
src/gnn_pruner/
  ingest/        ✅ pdf / docx / html / docling loaders     🆕 toc.py (bookmarks → structure), ocr later
  layers.py      ✅ 4-layer tree                            🆕 virtual chapters
  document_graph.py ✅ links                                🆕 entity links
  enrich.py      🆕 bottom-up key facts, summaries, metadata (cached)
  v4/
    board.py     🆕 central board: verify, merge, follow refs, stop rules
    worker.py    🆕 chapter worker: gate + descend loop (open / read / search / post / follow)
    screen.py    🆕 table-of-contents screen
    orchestrator.py 🆕 runs workers in parallel, budget, returns the evidence pack
  evidence.py, prompt.py, answer.py  ✅ steps 10–12
  answer_eval.py ✅ readings          🆕 long-document benchmark + 5-system comparison
```

---

## 10. Build order

| Step | What | Done when |
|---|---|---|
| v4-1 | TOC-based structure, virtual chapters, root metadata, bottom-up enrichment | a real PDF book becomes a tree with summaries and verified metadata |
| v4-2 | Long-document benchmark (100 → 10,000+ pages) | questions and answer keys exist, baselines 1–3 measured |
| v4-3 | Central board + chapter workers (descend loop, quote checks, references) | system 4 measured |
| v4-4 | Worker gate (local score + TOC screen) | system 5 measured, calls per question reported |
| v4-5 | Entity links for books | multi-chapter book questions measured |
| v4-6 | Learning loop (Phase 11) | the gate improves from board labels; calls per question go down |
