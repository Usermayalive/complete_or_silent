"""Build long versions of the example contracts for retrieval-at-scale evaluation.

Each original contract is kept verbatim (so gold snippets and section numbers are
unchanged) and followed by generated distractor sections that reuse the same
parties and vocabulary, so they compete with the real answers the way the rest of
a long contract does. Distractor topics avoid every unanswerable question's
subject (governing law, intellectual property, free trials, courts), so those
questions stay unanswerable.

    python eval/make_scale_docs.py     # writes examples/scale/*.md + eval/contracts_scale_v1.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from gnn_pruner.synthetic import distractors

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "scale"

DOCS = {
    "sample_contract.md": ("Supplier", "Customer", 11),
    "saas_agreement.md": ("Provider", "Customer", 7),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (a, b, seed) in DOCS.items():
        original = (ROOT / "examples" / name).read_text(encoding="utf-8").rstrip() + "\n\n"
        last = max(int(line.split(".")[0].lstrip("# ")) for line in original.splitlines()
                   if line.startswith("## ") and line[3].isdigit())
        (OUT / name).write_text(original + distractors(a, b, last + 1, seed), encoding="utf-8")
        print(f"wrote {OUT / name}")

    cases = [json.loads(l) for l in (ROOT / "eval" / "contracts_v1.jsonl").read_text().splitlines() if l.strip()]
    with open(ROOT / "eval" / "contracts_scale_v1.jsonl", "w", encoding="utf-8") as f:
        for case in cases:
            case["document"] = case["document"].replace("../examples/", "../examples/scale/")
            f.write(json.dumps(case) + "\n")
    print(f"wrote {ROOT / 'eval' / 'contracts_scale_v1.jsonl'} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
