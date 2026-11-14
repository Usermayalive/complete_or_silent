# Related work and novelty check (P0)

First pass, 2026-11-15. Sources: arXiv API (relevance-sorted queries below), arXiv abstract
pages for the closest papers. Semantic Scholar rate-limited the requests; ACL Anthology and
Google Scholar were not searched yet. **Not exhaustive.** Before submission: a second pass
on ACL Anthology / Scholar, and the authors read the closest papers in full.

Queries: conformal + retrieval; canary / synthetic / needle + calibration + long context;
contract / legal + exception / override / amendment / cross-reference + LLM;
completeness / abstention / absence + guarantee / certificate + long document; conformal +
label-free / unlabeled + retrieval; due diligence / data room / contract review + LLM.

## Verdict: GO, with a sharper claim

Nothing found does what Complete or Silent does as a whole. Two neighbours are close enough
that the paper must cite them and narrow its wording:

| Our claim | Closest work | What changes in our claim |
|---|---|---|
| closure certificate | **Certify** (Zhang & Sun, arXiv 2610.34684, Oct 2026): requirement hypergraph per question, Horn closure gives a certificate of the minimal *sufficient* passages, or abstention naming open requirements; open-web QA, 21M Wikipedia passages | "closure certificate" alone is not new. Ours is a **defeasibility closure**: everything that could *change or overturn* a found answer (exceptions, overrides, amendments, precedence) inside a document collection beyond the context window. Certify asks "do I have enough to derive the answer?"; we ask "can anything I have not read change it?". Contract clauses are typically sufficient *and* overridden elsewhere. |
| absence certificate with a calibrated miss rate | **CONFLARE** (Rouzrokh et al., arXiv 2404.04287, 2024): conformal similarity threshold so the answer is in the retrieved context with probability 1−α; **TRAQ** (Li et al., arXiv 2307.04642), **C-RAG** (Kang et al., arXiv 2402.03181): conformal guarantees for RAG answers / risks | conformal retrieval coverage is not new. Ours is **label-free and per document**: calibrated on canaries planted in the document itself, no labelled calibration questions; and the finding that canaries must imitate the *question* (type canaries) or the guarantee is false (25% measured vs 10% stated). |

## Closest papers by topic

### Certificates, sufficiency, absence
| Paper | What it does | Relation |
|---|---|---|
| Certify — Knowing When the Evidence Is Enough: Requirement-Directed Search with Horn-Closure Certificates (Zhang & Sun, 2026, arXiv 2610.34684) | sufficiency certificate by Horn closure over a requirement hypergraph; abstains naming open requirements; sound given correct retrieval and links | closest; sufficiency, not defeasibility; no calibrated absence; open web, not one huge document |
| When Absence Is Evidence (CROWN-QA; Min et al., 2026, arXiv 2608.04591) | benchmark: LLMs fail to tell a justified negative (complete coverage) from insufficient evidence | motivation for absence certificates; no method or guarantee |
| Judge, Retrieve, or Abstain (Badshah, 2026, arXiv 2608.17994) | Clopper–Pearson-calibrated thresholds for LLM judging with risk guarantees | statistical tooling; different task |
| Bounding Hallucinations: Merlin–Arthur protocols (Deiseroth et al., 2025, arXiv 2512.11614) | answer-or-abstain via mutual-information bounds | abstention guarantee for the answer, not for reading coverage |
| ProbeGuard — Certified abstention for medical LLM consensus (Wang, 2026, arXiv 2610.07570) | certified abstention bounding selective risk | answer-level, not document coverage |

### Conformal retrieval and RAG
| Paper | What it does | Relation |
|---|---|---|
| CONFLARE (Rouzrokh et al., 2024, arXiv 2404.04287) | conformal similarity cutoff; answer in context w.p. 1−α; calibration set of answerable questions | closest calibration; needs labelled questions |
| TRAQ (Li et al., 2023, arXiv 2307.04642) | end-to-end conformal correctness for RAG | labelled calibration |
| C-RAG (Kang et al., 2024, arXiv 2402.03181) | certified generation-risk bounds for RAG | labelled calibration; risk of the answer |
| Principled Context Engineering for RAG (Chakraborty, 2025, arXiv 2511.17908) | coverage-controlled context filtering | labelled calibration |
| Conformal Factuality Control for Multi-Hop RAG (Rehman, 2026, arXiv 2609.38222) | conformal claim filtering | output side |
| Certified Domain Consistency (Manoharan, 2026, arXiv 2607.14157) | label-free conformal control of cross-domain contamination | label-free, but a different quantity |

### Legal structure, cross-references, amendments
| Paper | What it does | Relation |
|---|---|---|
| CRAwLeR — Cross-Reference Aware Legal Retrieval (Jalocha, 2026, arXiv 2606.21676) | detects legal cross-references for context-aware retrieval | forward references only; statutes |
| StrucChunk (Singh, 2026, arXiv 2610.26313) | structure-preserving legal chunking; resolves cross-references before embedding | forward references; no reverse index, no certificate |
| Poly-Vector Retrieval (Lima, 2025, arXiv 2504.10508) | separate embeddings for content and labels to resolve references | retrieval only |
| ViHERMES (Nguyen, 2026, arXiv 2602.07361) | multi-hop QA benchmark over regulations incl. amendment tracing | benchmark of amendments in regulations; no certificates; not contract data rooms |
| NitiBench+ (Akarajaradwong, 2026, arXiv 2611.03576) | temporal legal QA; amended or repealed statutes | temporal validity of statutes |
| GRAPH-GRPO-LEX (Dechtiar et al., 2025, arXiv 2511.06618) | contract graph ontology; LLM + GRPO extracts clause relationships and hidden dependencies | clause graph extraction; no QA, no completeness guarantee |
| PAKTON (Raptopoulos, 2025, arXiv 2506.00608) | multi-agent RAG for QA on long legal agreements | single agreements; no certificates |
| Towards a rigorous evaluation of RAG: due diligence (Martinon, 2025, arXiv 2507.21753) | evaluation of RAG for fund due diligence | evaluation; motivates the setting |

### Planted items / synthetic probes
No paper found that plants items into a document to calibrate a reader's miss rate on that
same document. Nearest: RCL (Ravuri, 2026, arXiv 2609.11023), synthetic APIs injected into
code repositories to detect insufficient retrieval; needle-in-a-haystack tests
(Kamradt 2023; RULER) measure models, not calibrate a system per document.

## What the paper must now say
1. Cite Certify, CROWN, CONFLARE, TRAQ, C-RAG, CRAwLeR / StrucChunk, ViHERMES in related
   work; position: *defeasibility* closure (not sufficiency) + *label-free per-document*
   calibration (not labelled conformal retrieval) + data rooms beyond the context window.
2. Do not call the closure certificate the first certificate for QA; call it the first
   certificate that nothing **unread can change** a found answer, for documents beyond
   the context window (to our knowledge, after this search).
3. An experiment that would answer the obvious reviewer question: Certify-style
   sufficiency checking on the trap benchmark (it would accept the base clause as
   sufficient and miss the override) — a strong, cheap baseline.

## Still to do (P0 second pass)
- ACL Anthology and Google Scholar searches (2024–2026), especially ACL/EMNLP/NAACL 2025–26
  long-document and legal QA papers.
- Read Certify and CONFLARE in full (method sections) and confirm the distinctions above.
- Verify every bibliographic entry from official exports.
- **You:** read Certify, CONFLARE, CROWN, TRAQ, ViHERMES and CRAwLeR in full.
