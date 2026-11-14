# Paper: Complete or Silent (draft)

`main.tex` + `references.bib` (ACL format; official `acl.sty` and `acl_natbib.bst` from
https://github.com/acl-org/acl-style-files are included). Every number is taken from
`docs/READINGS.md`.

## Compile
Upload the four files to an Overleaf project (or any TeX Live): `pdflatex main && bibtex main
&& pdflatex main && pdflatex main`.

## Draft status
- Red `[PENDING: …]` marks (`\pending{}`) are results or sections still to come. **Every one
  must be resolved, and the `\pending` macro removed, before submission.**
- Written: problem definitions (D4), method, benchmark, RQ1 (gate level), RQ2/RQ3 first
  reading (traps v1, 30 questions), RQ4 first numbers, RQ5 structural ablations,
  limitations, ethics.
- Pending: literature search (P0), traps2 with two answer models, external baselines (long
  context, Chain of Agents, reranked retrieval), worker-level RQ1, cost curves, human
  evaluation, AI-assistance statement, figures.

## Before submitting
- [ ] Replace every `references.bib` entry by its official export; fix all `% VERIFY` items.
- [ ] Keep `[review]` and "Anonymous ACL Submission" for review.
- [ ] 8 pages of content (limitations, ethics, references not counted).
- [ ] Re-check every number against `docs/READINGS.md`; report nothing that was not measured.
