# Paper: GNN-Pruner

`main.tex` + `custom.bib` (ACL format). Every number is taken from `docs/READINGS.md`.

## Compile
1. Get the official ACL template (`acl.sty`, `acl_natbib.bst`) from
   https://github.com/acl-org/acl-style-files and put both next to `main.tex`. Or upload
   all three files into an Overleaf project created from the ACL template.
2. `pdflatex main && bibtex main && pdflatex main && pdflatex main`

The pipeline figure and both plots (Figures 2–3) are drawn with TikZ / pgfplots from the measured numbers, so no image files are needed.

## Before submitting
- [ ] **References:** replace every entry in `custom.bib` with the BibTeX exported from ACL
      Anthology / DBLP / the DOI page / arXiv; fix every `% VERIFY` item.
- [ ] **Authors:** keep `[review]` + "Anonymous ACL Submission" for review. For a preprint,
      switch to `[preprint]` and use the commented 4-author block.
- [ ] **Use of AI assistants:** adjust the wording to what was actually used.
- [ ] **Length:** ACL long papers allow 8 pages of content (Limitations, Ethics and
      references do not count). If over, move the robustness paragraph and Tables 5–6 to
      an appendix.
- [ ] **Accuracy:** re-check any number you edit against `docs/READINGS.md`; never add
      results that were not measured.
