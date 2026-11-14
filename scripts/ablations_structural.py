"""RQ5 structural ablations on traps2 (no model): with an oracle qualifier worker that
accepts exactly the gold provisions among the sweep candidates (so the sweep's reach is
measured, not a model's judgement), how many traps get every required provision into the
closure, and is the superseded clause marked?

Run: PYTHONPATH=src python scripts/ablations_structural.py traps2 M
"""
import sys, re, collections, json
from gnn_pruner.answer import Completion
from gnn_pruner.tree import tree_from_file
from gnn_pruner.certify import build_qualifier_graph, ReverseIndex
from gnn_pruner.certify.closure import expand
from gnn_pruner.certify.precedence import order, SUPERSEDED
from gnn_pruner.evaluation import load_cases

fam, tier = sys.argv[1], sys.argv[2]
cases = [c for c in load_cases(f"eval/long/{fam}_{tier}.jsonl") if c.tags.get("kind") in ("far_exception", "override_chain", "unlinked", "precedence")]
tree = tree_from_file(cases[0].document); g = build_qualifier_graph(tree); rev = ReverseIndex(tree, g)
by_text = {}
for p in tree.passages():
    for sid, txt in p.sentences: by_text.setdefault(txt, (p.id, sid))

class Oracle:
    """Qualifier worker that knows the gold: required provisions are accepted (restates for
    caps, qualifies otherwise), everything else is irrelevant."""
    def __init__(self, gold): self.gold = gold
    def complete(self, system, user, schema):
        v = []
        for m in re.finditer(r"^\[([^\]]+)\] (.*)$", user, re.M):
            t = m.group(2)
            hit = any(t in q or q in t for q in self.gold)
            verdict = ("restates" if "shall not exceed" in t.lower() or "liability" in t.lower() and "fraud" not in t.lower() else "qualifies") if hit else "irrelevant"
            v.append({"sentence_id": m.group(1), "verdict": verdict})
        return Completion(json.dumps({"verdicts": v}), "oracle", 0, 0)

VARIANTS = {
    "full": {},
    "A1 no reverse index": {"use_reverse": False},
    "A2 no sweep": {"sweep": False},
    "A3 no forward links": {"use_forward": False},
    "A6 no precedence": {"use_precedence": False},
}
res = collections.defaultdict(lambda: collections.Counter())
for c in cases:
    kind = c.tags["kind"]; base = by_text[c.gold_evidence[0]]
    req = c.tags["required"]
    for name, kw in VARIANTS.items():
        cl = expand(tree, g, rev, [(base[0], [base[1]])], provider=Oracle(req), **kw)
        found = {tree.quote(i.sentence_ids) for i in cl.items}
        ok = all(any(q in f or f in q for f in found) for q in req)
        res[(name, kind)]["n"] += 1; res[(name, kind)]["reach"] += ok
        if c.tags["superseded"]:
            st = order(tree, cl).status_of(base[0])
            res[(name, kind)]["sup_n"] += 1; res[(name, kind)]["sup_ok"] += (st == SUPERSEDED)
kinds = ["far_exception", "override_chain", "unlinked", "precedence"]
print(f"traps {fam} {tier}: every required provision in the closure (oracle qualifier worker); superseded base marked")
print("| Variant | " + " | ".join(kinds) + " | superseded marked |")
print("|---|" + "---|" * (len(kinds) + 1))
for name in VARIANTS:
    cells = [f"{res[(name,k)]['reach']}/{res[(name,k)]['n']}" for k in kinds]
    sn = sum(res[(name,k)]["sup_n"] for k in kinds); so = sum(res[(name,k)]["sup_ok"] for k in kinds)
    print(f"| {name} | " + " | ".join(cells) + f" | {so}/{sn} |")
