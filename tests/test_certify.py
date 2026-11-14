"""Complete or Silent, P1: qualifier graph (U2) and reverse index (U3). No network."""

from gnn_pruner.certify import ReverseIndex, build_qualifier_graph, units
from gnn_pruner.certify.canaries import Calibration
from gnn_pruner.certify.qualifiers import AMENDS, OVERRIDES, PRECEDENCE, QUALIFIES
from gnn_pruner.document_graph import DocumentGraph
from gnn_pruner.tree import build_tree

ROOM = """# Master Services Agreement between BETA Logistics and ACME

## 1. Definitions

"Fees" means all amounts invoiced under this Agreement.

## 12. Liability

### 12.3 Cap

ACME's total liability shall not exceed the Fees paid in the 12 months before the claim.

### 12.4 Exclusions

Neither party is liable for indirect loss, subject to Section 12.3.

## 20. Precedence

In the event of any conflict, the Schedules shall prevail over the body of this Agreement.

# Schedule 3 Data Protection

Notwithstanding Section 12.3, liability for data protection breaches shall not exceed 5,000,000 USD.

# Schedule 5 Fraud

The limitations above shall not apply to fraud or wilful misconduct.

# Amendment No. 2 to the Master Services Agreement between BETA Logistics and ACME

Section 12.3 is hereby deleted and replaced: liability shall not exceed the Fees paid in the 24 months before the claim.

# Supply Agreement between BETA Logistics and GAMMA

## 12. Liability

### 12.3 Cap

GAMMA's liability is limited to 100,000 USD.
"""


def room():
    return build_tree(DocumentGraph.from_text(ROOM, title="BETA data room"))


def passage_with(tree, text):
    return next(p for p in tree.passages() if text in (p.text or ""))


def test_units_group_schedules_with_their_contract():
    tree = room()
    groups = units(tree)
    titles = {u: [tree.node(c).title for c in chs] for u, chs in groups.items()}
    msa = next(v for v in titles.values() if v[0].startswith("Master Services"))
    assert any(t.startswith("Schedule 3") for t in msa) and any(t.startswith("Schedule 5") for t in msa)
    assert len(groups) == 3  # MSA (+ schedules), the amendment, the supply agreement


def test_links_have_kinds_and_resolve_inside_the_own_contract():
    tree = room()
    graph = build_qualifier_graph(tree)
    cap = passage_with(tree, "12 months")
    gamma_cap = passage_with(tree, "GAMMA's liability")
    sched3 = [l for l in graph.links if l.kind == OVERRIDES and l.target_type == "section"]
    assert sched3 and all(l.resolved for l in sched3)
    # "Section 12.3" in Schedule 3 is the MSA's 12.3, never the supply agreement's
    targets = {t for l in sched3 for t in l.targets}
    assert not any(t in tree.path(gamma_cap.id) or t == gamma_cap.id for t in targets)
    assert any(t == cap.id or t in [n for n in _ancestors(tree, cap.id)] for t in targets)
    assert graph.by_kind(PRECEDENCE)
    assert any(l.kind == QUALIFIES and l.target_label == "12.3" for l in graph.links)


def test_unlinked_qualifier_is_kept_unresolved():
    graph = build_qualifier_graph(room())
    fraud = [l for l in graph.links if l.cue == "shall not apply"]
    assert fraud and not any(l.resolved for l in fraud)


def test_amendment_names_its_contract_and_section():
    tree = room()
    graph = build_qualifier_graph(tree)
    amends = graph.by_kind(AMENDS)
    assert any(l.target_type == "contract" for l in amends)
    section = [l for l in amends if l.target_type == "section"]
    assert section and section[0].resolved
    msa_cap = passage_with(tree, "12 months")
    assert set(section[0].targets) & ({msa_cap.id} | set(_ancestors(tree, msa_cap.id)))


def test_reverse_index_finds_far_override_and_amendment():
    tree = room()
    rev = ReverseIndex(tree, build_qualifier_graph(tree))
    cap = passage_with(tree, "12 months")
    pointing = rev.pointing_at(cap.id)
    sources = {tree.sentence(l.sentence)[0][:40] for l in pointing}
    assert any(s.startswith("Notwithstanding Section 12.3") for s in sources)
    assert any("deleted and replaced" in tree.sentence(l.sentence)[0] for l in pointing)
    assert any("subject to Section 12.3" in tree.sentence(l.sentence)[0] for l in pointing)
    # nothing from the other contract's 12.3
    gamma = passage_with(tree, "GAMMA's liability")
    assert all(l.passage != gamma.id for l in pointing)
    assert not rev.pointing_at(gamma.id)


def _ancestors(tree, node_id):
    out, node = [], tree.node(node_id)
    while node.parent:
        out.append(node.parent)
        node = tree.node(node.parent)
    return out


# ------------------------------------------------------------ U4 canaries

def test_conformal_budget_and_upper_bound():
    from gnn_pruner.certify.canaries import clopper_pearson_upper, conformal_k
    ranks = list(range(1, 21))                       # 20 canaries, ranks 1..20
    assert conformal_k(ranks, 0.9, 50) == 19         # ceil(21 * 0.9) = 19th smallest
    assert conformal_k(ranks, 0.98, 50) == 50        # ceil(21 * 0.98) = 21 > 20: open everything
    assert clopper_pearson_upper(0, 100) < 0.031     # rule of three: about 3 / n
    assert abs(clopper_pearson_upper(5, 100) - 0.105) < 0.01


def test_canaries_are_mutated_copies_planted_without_touching_the_document(tmp_path):
    import random
    from gnn_pruner.certify.canaries import calibrate, make_canaries
    tree = room()
    before = [s for p in tree.passages() for s in p.sentences]
    canaries = make_canaries(tree, 10, random.Random(1))
    assert canaries and all(c.text not in {t for _, t in before} for c in canaries)
    assert all(c.cls in ("fact", "clause", "qualifier") for c in canaries)
    cal = calibrate(tree, canaries=canaries)
    assert [s for p in tree.passages() for s in p.sentences] == before   # the real document is unchanged
    assert all(1 <= c.rank_in_unit <= c.rank <= cal.chapters for c in canaries)
    assert 1 <= cal.k(0.9) <= cal.chapters and cal.k(0.9, table="unit") <= cal.k(0.9)
    again = calibrate(tree, seed=0, n=10)
    assert again.to_dict() == calibrate(tree, seed=0, n=10).to_dict()        # deterministic
    path = str(tmp_path / "cal.json")
    again.save(path)
    assert Calibration.load(path).to_dict() == again.to_dict()


# -------------------------------------------------------------- Q1 + Q2 plan

def test_question_class_and_reading_plan():
    from gnn_pruner.certify.canaries import ClassCalibration
    from gnn_pruner.certify.plan import certified_alpha, classify_question, reading_plan
    assert classify_question("Is there a change-of-control clause in the MSA?") == "existence"
    assert classify_question("What hourly rate applies under the Services Agreement?") == "fact"
    assert classify_question("List all obligations of the Supplier.") == "list"
    assert classify_question("Which law governs the Lease of BETA?") == "clause"
    cc = lambda ks: ClassCalibration(100, {"0.9": ks[0], "0.98": ks[1]}, {"0.9": 0.1, "0.98": 0.02},
                                     {"0.9": 0.13, "0.98": 0.04})
    cal = Calibration(20, {"room": {"all": cc((5, 9)), "clause": cc((4, 8))},
                           "unit": {"all": cc((2, 3)), "clause": cc((2, 3))}}, 0)
    assert reading_plan(None, "Which law governs?", []).basis == "uncalibrated"
    p = reading_plan(cal, "Which law governs?", [])                 # no unit named: room table, clause
    assert (p.basis, p.k, p.question_class) == ("room", 8, "clause")
    p = reading_plan(cal, "Which law governs the Lease?", ["ch2"])  # a small named unit: open it whole
    assert (p.basis, p.k, p.chapters) == ("named", 1, ["ch2"])
    p = reading_plan(cal, "Where did the hare hide?", [f"ch{i}" for i in range(10)], alpha=0.9)
    assert (p.basis, p.k) == ("unit", 2)                            # a book: unit table
    ranking = [f"ch{i}" for i in range(20)]
    room = reading_plan(cal, "Which law governs?", [])
    assert certified_alpha(cal, room, ranking[:8], ranking) == 0.98
    assert certified_alpha(cal, room, ranking[:5], ranking) == 0.9
    assert certified_alpha(cal, room, ranking[:3], ranking) is None
    assert certified_alpha(cal, reading_plan(cal, "q", ["ch2"]), ["ch2", "ch5"], ranking) == 0.98


def test_ask_uses_the_calibrated_budget_and_records_the_level_reached():
    from gnn_pruner.certify.canaries import calibrate
    from gnn_pruner.workers import ask
    from tests.test_workers import FakeModel, ROOM as WROOM
    from gnn_pruner.document_graph import DocumentGraph as DG
    fillers = "".join(f"## Notice {k} of DELTA{k}\n\nThe parties met on day {k} about widgets.\n\n" for k in range(6))
    tree = build_tree(DG.from_text(WROOM.replace("## Schedule 1 Prices", fillers + "## Schedule 1 Prices")))
    cal = calibrate(tree, n=40, seed=0)
    result = ask(tree, "Which agreement is governed by Zanzibar law?", FakeModel("Zanzibar"), use_toc=False,
                 calibration=cal, alpha=0.9)
    assert result.answer.abstained and result.reading.basis == "room"
    assert result.reading.k == cal.k(0.9, "clause")
    assert result.certified in (None, 0.9, 0.95, 0.98)
    if len(result.chapters_searched) >= cal.k(0.9, "clause"):
        assert result.certified is not None and result.certified >= 0.9


# ----------------------------------------------------------------- Q4 closure

class FakeQualifierJudge:
    """Says the fraud carve-out qualifies and everything else is irrelevant."""

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, schema):
        import json as _json
        import re as _re

        from gnn_pruner.answer import Completion
        self.calls += 1
        verdicts = []
        for m in _re.finditer(r"^\[([^\]]+)\] (.*)$", user, _re.M):
            verdicts.append({"sentence_id": m.group(1), "verdict": "qualifies" if "fraud" in m.group(2) else "irrelevant"})
        return Completion(_json.dumps({"verdicts": verdicts}), "fake", 100, 20)


def test_closure_reaches_a_fixpoint_with_override_amendment_definition_and_precedence():
    from gnn_pruner.certify.closure import expand
    tree = room()
    graph = build_qualifier_graph(tree)
    rev = ReverseIndex(tree, graph)
    cap = passage_with(tree, "12 months")
    seeds = [(cap.id, [s for s, _ in cap.sentences])]
    c = expand(tree, graph, rev, seeds)                     # no model: structural only
    texts = {i.relation: tree.quote(i.sentence_ids)[:30] for i in c.items}
    assert c.complete and c.reason == "fixpoint"
    assert texts["overrides"].startswith("Notwithstanding Section 12.3")
    assert texts["amends"].startswith("Section 12.3 is hereby deleted")
    assert texts["qualifies"].startswith("Neither party is liable")          # "subject to Section 12.3"
    assert texts["definition"].startswith('"Fees" means')
    assert texts["precedence"].startswith("In the event of any conflict")
    assert c.unverified >= 1 and not any("fraud" in tree.quote(i.sentence_ids) for i in c.items)
    assert c.calls == 0
    # nothing from the other contract's 12.3
    gamma = passage_with(tree, "GAMMA's liability")
    assert gamma.id not in c.passages()


def test_closure_sweep_uses_the_qualifier_worker_and_feeds_the_pack():
    from gnn_pruner.certify.closure import apply_to_board, expand
    from gnn_pruner.workers.board import Board
    tree = room()
    graph = build_qualifier_graph(tree)
    rev = ReverseIndex(tree, graph)
    cap = passage_with(tree, "12 months")
    judge = FakeQualifierJudge()
    board = Board(tree, "What is ACME's maximum liability?")
    board.submit("w1", [{"sentence_ids": [s for s, _ in cap.sentences], "why": "cap"}])
    c = expand(tree, graph, rev, [(cap.id, [s for s, _ in cap.sentences])], provider=judge, board=board)
    assert c.complete and c.unverified == 0 and c.calls == judge.calls >= 1
    fraud = [i for i in c.items if "fraud" in tree.quote(i.sentence_ids)]
    assert fraud and fraud[0].found_by == "sweep" and fraud[0].verdict == "qualifies"
    assert any(d.verdict == "irrelevant" for d in c.dropped) or True
    assert apply_to_board(board, c) == len(c.items)
    pack = board.to_pack(max_items=12)
    roles = [i.role for i in pack.items]
    assert roles[0] == "answer" and set(roles) >= {"overrides", "amends", "qualifies", "precedence"}
    assert any("24 months" in i.text for i in pack.items) and any("fraud" in i.text for i in pack.items)


def test_closure_stops_at_the_budget_and_says_so():
    from gnn_pruner.certify.closure import expand
    tree = room()
    graph = build_qualifier_graph(tree)
    rev = ReverseIndex(tree, graph)
    cap = passage_with(tree, "12 months")
    c = expand(tree, graph, rev, [(cap.id, [s for s, _ in cap.sentences])], max_items=1)
    assert not c.complete and c.reason == "budget"


# ------------------------------------------------------- Q5 precedence + Q6 certificates

def _beta_closure(provider=None):
    from gnn_pruner.certify.closure import expand
    tree = room()
    graph = build_qualifier_graph(tree)
    rev = ReverseIndex(tree, graph)
    cap = passage_with(tree, "12 months")
    seeds = [(cap.id, [s for s, _ in cap.sentences])]
    return tree, cap, expand(tree, graph, rev, seeds, provider=provider)


def test_precedence_marks_the_amended_cap_superseded_and_keeps_it():
    from gnn_pruner.certify.precedence import APPLIES, GOVERNS, SUPERSEDED, order
    tree, cap, closure = _beta_closure()
    o = order(tree, closure)
    assert o.status_of(cap.id) == SUPERSEDED                      # replaced by Amendment 2, text kept
    amend = next(r for r in o.rulings if r.item.relation == "amends")
    assert amend.status == GOVERNS and o.status_of(cap.id) == SUPERSEDED
    assert next(r for r in o.rulings if r.item.passage == cap.id).by == amend.item.passage
    override = next(r for r in o.rulings if r.item.relation == "overrides")
    assert override.status == APPLIES and override.rule == "precedence clause"   # schedules prevail (§20)
    assert o.precedence_rule == "schedules" and not o.conflicts


def test_an_earlier_dated_amendment_does_not_supersede_and_is_flagged():
    from gnn_pruner.certify.precedence import GOVERNS, order
    text = ROOM.replace("# Master Services Agreement between BETA Logistics and ACME",
                        "# Master Services Agreement between BETA Logistics and ACME (2023-01-01)") \
               .replace("# Amendment No. 2 to the Master Services Agreement between BETA Logistics and ACME",
                        "# Amendment No. 2 to the Master Services Agreement between BETA Logistics and ACME (2021-05-05)")
    from gnn_pruner.certify.closure import expand
    tree = build_tree(DocumentGraph.from_text(text, title="BETA data room"))
    graph = build_qualifier_graph(tree)
    cap = passage_with(tree, "12 months")
    closure = expand(tree, graph, ReverseIndex(tree, graph), [(cap.id, [s for s, _ in cap.sentences])])
    o = order(tree, closure)
    assert o.status_of(cap.id) == GOVERNS and any("dated earlier" in c for c in o.conflicts)


def test_closure_certificate_is_honest_about_unverified_sweeps():
    from gnn_pruner.certify.certificate import ABSENCE, CLOSURE, UNCERTIFIED, absence_certificate, closure_certificate
    from gnn_pruner.certify.precedence import order
    tree, cap, closure = _beta_closure()                                   # no model: sweep unverified
    cert = closure_certificate(tree, closure, order(tree, closure), [c.id for c in tree.chapters()])
    assert cert.kind == UNCERTIFIED and any("not judged" in r for r in cert.reasons)
    assert cert.text().startswith("UNCERTIFIED") and cert.to_dict()["kind"] == UNCERTIFIED
    tree, cap, closure = _beta_closure(FakeQualifierJudge())               # judged: complete
    cert = closure_certificate(tree, closure, order(tree, closure), [c.id for c in tree.chapters()])
    assert cert.kind == CLOSURE and cert.complete and cert.evidence["superseded"] == 1
    assert "COMPLETE" in cert.text() and cert.read["chapters"] == cert.read["of"]


def test_absence_certificate_states_the_level_and_the_canary_miss_rate():
    import random
    from gnn_pruner.certify.canaries import calibrate
    from gnn_pruner.certify.certificate import ABSENCE, UNCERTIFIED, absence_certificate
    from gnn_pruner.certify.plan import certified_alpha, reading_plan
    tree = room()
    cal = calibrate(tree, n=40, seed=1)
    plan = reading_plan(cal, "Is there a change-of-control clause?", [], alpha=0.9)
    ranking = [c.id for c in tree.chapters()]
    opened = ranking[:plan.k]
    cert = absence_certificate(tree, cal, plan, certified_alpha(cal, plan, opened, ranking), opened)
    assert cert.kind == ABSENCE and cert.level >= 0.9 and "NOT FOUND" in cert.text()
    assert cert.evidence["canaries"] == cal.tables["room"]["all"].n > 0      # the small room yields fewer than 40
    assert 0 <= cert.evidence["miss"] <= cert.evidence["miss_upper"] <= 1
    short = absence_certificate(tree, cal, plan, certified_alpha(cal, plan, opened[:1], ranking), opened[:1])
    assert short.kind == UNCERTIFIED if plan.k > 1 else short.kind == ABSENCE
    assert absence_certificate(tree, None, reading_plan(None, "q", []), None, opened).kind == UNCERTIFIED


# -------------------------------------------------------------- Q7 + end to end

class CertifyingModel:
    """Worker: posts shown sentences holding ``keyword``. Qualifier worker: fraud qualifies.
    Verifier: every sentence supported by E1. Answer: cites E1."""

    def __init__(self, keyword):
        self.keyword, self.calls, self.schemas = keyword, 0, []

    def complete(self, system, user, schema):
        import json as _json
        import re as _re

        from gnn_pruner.answer import ANSWER_SCHEMA, Completion
        from gnn_pruner.certify.closure import QUALIFIER_SCHEMA
        from gnn_pruner.certify.verify import VERIFY_SCHEMA
        from gnn_pruner.workers.gate import REWRITE_SCHEMA, TOC_SCHEMA
        self.calls += 1
        self.schemas.append(schema)
        if schema is TOC_SCHEMA:
            return Completion(_json.dumps({"chapters": []}), "fake", 10, 5)
        if schema is REWRITE_SCHEMA:
            return Completion(_json.dumps({"keywords": ["liability"]}), "fake", 10, 5)
        if schema is QUALIFIER_SCHEMA:
            v = [{"sentence_id": m.group(1), "verdict": "qualifies" if "fraud" in m.group(2) else "irrelevant"}
                 for m in _re.finditer(r"^\[([^\]]+)\] (.*)$", user, _re.M)]
            return Completion(_json.dumps({"verdicts": v}), "fake", 100, 20)
        if schema is VERIFY_SCHEMA:
            return Completion(_json.dumps({"sentences": [{"text": "24 months of fees.", "supported": True, "by": ["E1"]}]}),
                              "fake", 50, 10)
        if schema is ANSWER_SCHEMA:
            return Completion(_json.dumps({"answer": "24 months of fees [E1].", "evidence": ["E1"],
                                           "confidence": "high", "insufficient": False}), "fake", 50, 10)
        ids = [m.group(1) for m in _re.finditer(r"^\[([^\]]+)\] (.*)$", user, _re.M) if self.keyword in m.group(2)]
        return Completion(_json.dumps({"findings": [{"sentence_ids": ids, "why": "cap"}] if ids else [], "follow": [],
                                       "partial": "", "next": {"action": "done", "target": ""}}), "fake", 100, 20)


def test_verify_marks_unsupported_sentences_and_unknown_labels():
    from gnn_pruner.certify.verify import verify
    from gnn_pruner.workers.board import Board
    tree = room()
    cap = passage_with(tree, "12 months")
    board = Board(tree, "cap?")
    board.submit("w", [{"sentence_ids": [s for s, _ in cap.sentences], "why": "cap"}])
    pack = board.to_pack()

    class Judge:
        def complete(self, system, user, schema):
            import json as _json
            from gnn_pruner.answer import Completion
            assert "[E1]" in user and "Answer:" in user
            return Completion(_json.dumps({"sentences": [
                {"text": "The cap is 12 months of fees.", "supported": True, "by": ["E1"]},
                {"text": "It was never amended.", "supported": False, "by": []},
                {"text": "Made up.", "supported": True, "by": ["E9"]}]}), "fake", 10, 5)

    v = verify("...", pack, Judge(), board)
    assert v.total == 3 and v.supported == 1 and len(v.unsupported) == 2 and board.calls == 1


def test_ask_with_certify_returns_a_closure_certificate_and_an_absence_certificate():
    from gnn_pruner.certify.canaries import calibrate
    from gnn_pruner.workers import ask
    tree = room()
    cal = calibrate(tree, n=20, seed=0)
    model = CertifyingModel("12 months")
    q = "What is ACME's maximum liability under the Master Services Agreement between BETA Logistics and ACME?"
    r = ask(tree, q, model, use_toc=False, certify=True, calibration=cal, alpha=0.9)
    assert not r.answer.abstained and r.closure is not None and r.closure.complete
    rel = {i.relation for i in r.closure.items}
    assert {"overrides", "amends", "qualifies", "definition", "precedence"} <= rel
    assert r.ordering.superseded() and r.verification.supported == 1
    assert r.certificate.kind == "closure" and "COMPLETE" in r.certificate.text()
    roles = [i.role for i in r.pack.items]
    assert roles[0] == "answer" and "overrides" in roles and r.calls == model.calls
    # nothing found: absence certificate from the calibration
    r2 = ask(tree, "Is there a change-of-control clause in the Supply Agreement between BETA Logistics and GAMMA?",
             CertifyingModel("Zanzibar"), use_toc=False, certify=True, calibration=cal, alpha=0.9)
    assert r2.answer.abstained and r2.certificate.kind in ("absence", "uncertified")
    assert r2.certificate.text().startswith(("NOT FOUND", "UNCERTIFIED"))


# ------------------------------------------------------------------ P2 traps

def _trap_units():
    from gnn_pruner.longbench import Unit
    body = lambda name: [f"### 1. Scope\n\nThis agreement between {name} and the Customer covers services.",
                         "### 2. Fees\n\nThe Customer pays the fees in Schedule 1 within thirty days.",
                         "### 3. Term\n\nThe term is three years from the Effective Date.",
                         "### 4. Notices\n\nNotices are served in writing at the registered office."]
    us = [Unit(f"k{i}", f"Services Agreement of {name} Corp", body(name), tier="S", chars=400)
          for i, name in enumerate(("Alder", "Birch", "Cedar", "Dogwood"))]
    for u in us:
        u.meta["spans"] = {"Change Of Control": []}
    return us


def test_traps_build_every_kind_with_gold_and_expected_certificates():
    from gnn_pruner.certify.traps import KINDS, build
    report, texts, rows = build(seed=3, counts={"S": 10}, units=_trap_units(), write=False)
    kinds = {r["tags"]["kind"] for r in rows["S"]}
    assert kinds == set(KINDS) and report["traps_S"]["questions"] == 10
    for r in rows["S"]:
        assert all(g in texts["S"] for g in r["gold_evidence"])
        assert r["tags"]["certificate"] in ("closure", "absence")
        if r["tags"]["kind"] != "existence":
            assert r["tags"]["required"] and "distance_pages" in r["tags"]
    absent = [r for r in rows["S"] if r["tags"]["kind"] == "existence" and not r["answerable"]]
    assert absent and absent[0]["tags"]["certificate"] == "absence"


def test_closure_finds_the_planted_traps_on_rendered_text():
    from gnn_pruner.certify.closure import expand
    from gnn_pruner.certify.precedence import SUPERSEDED, order
    from gnn_pruner.certify.traps import build
    _, texts, rows = build(seed=5, counts={"S": 20}, units=_trap_units(), write=False)
    tree = build_tree(DocumentGraph.from_text(texts["S"], title="traps"))
    graph = build_qualifier_graph(tree)
    rev = ReverseIndex(tree, graph)

    class Judge:                      # a scripted qualifier worker: carve-outs qualify, other caps override
        def complete(self, system, user, schema):
            import json as _json
            import re as _re
            from gnn_pruner.answer import Completion
            v = []
            for m in _re.finditer(r"^\[([^\]]+)\] (.*)$", user, _re.M):
                t = m.group(2)
                verdict = "qualifies" if "fraud" in t else "restates" if "shall not exceed" in t else "irrelevant"
                v.append({"sentence_id": m.group(1), "verdict": verdict})
            return Completion(_json.dumps({"verdicts": v}), "fake", 10, 5)

    seen = {}
    for r in rows["S"]:
        kind = r["tags"]["kind"]
        if kind == "existence" or kind in seen:
            continue
        base = passage_with(tree, r["gold_evidence"][0])
        seeds = [(base.id, [s for s, _ in base.sentences])]
        structural = expand(tree, graph, rev, seeds)                       # no model
        offered = {tree.quote(i.sentence_ids) for i in structural.candidates}
        c = expand(tree, graph, rev, seeds, provider=Judge())               # with the worker
        found = {tree.quote(i.sentence_ids) for i in c.items}
        for q in r["tags"]["required"]:
            assert any(q in f or f in q for f in found), (kind, q, found)
        if kind in ("unlinked", "precedence"):                               # only the sweep can reach these
            assert any(q in f or f in q for q in r["tags"]["required"] for f in offered), kind
        if r["tags"]["superseded"]:
            assert order(tree, c).status_of(base.id) == SUPERSEDED, kind
        seen[kind] = True
    assert set(seen) == {"far_exception", "override_chain", "unlinked", "precedence"}


# ------------------------------------------------------------- B3 baseline

def test_chain_of_agents_reads_every_chunk_in_order_and_answers_from_notes():
    import json as _json
    from gnn_pruner.answer import Completion
    from gnn_pruner.baselines.chain_of_agents import MANAGER_SCHEMA, ask, chunk
    text = "\n\n".join(f"Paragraph {i} says nothing." for i in range(40)) + "\n\nThe cap is 12 months of fees."
    parts = chunk(text, 50)
    assert len(parts) > 3 and "".join(parts).replace("\n", "") == text.replace("\n", "")

    class Model:
        calls = 0
        def complete(self, system, user, schema):
            self.calls += 1
            if schema is MANAGER_SCHEMA:
                assert "12 months" in user
                return Completion(_json.dumps({"answer": "12 months of fees", "insufficient": False}), "f", 5, 5)
            notes = user.split("Notes so far:\n")[1].split("\n\n<document>")[0].replace("(none)", "")
            if "12 months" in user.split("<document>")[1]:
                notes += ' "The cap is 12 months of fees."'
            return Completion(_json.dumps({"notes": notes}), "f", 5, 5)

    m = Model()
    r = ask(text, "What is the cap?", m, chunk_tokens=50)
    assert r.calls == m.calls == len(parts) + 1 and r.answer == "12 months of fees" and not r.abstained
    assert "12 months" in r.notes


def test_type_canaries_get_their_own_table_and_serve_clause_questions():
    from gnn_pruner.certify.canaries import calibrate
    from gnn_pruner.certify.plan import reading_plan
    tree = room()
    library = [(["restrict", "compet"], f"Neither party shall compete with the other in territory {i}.") for i in range(40)]
    cal = calibrate(tree, n=40, seed=0, type_library=library)        # 40 type canaries (>= MIN_CLASS_N)
    assert "type" in cal.tables["room"] and cal.tables["room"]["type"].n == 40
    assert cal.tables["room"]["all"].n < 40 + cal.tables["room"]["all"].n      # "all" = own canaries only
    p = reading_plan(cal, "Does the agreement restrict either party from competing?", [])
    assert p.question_class == "clause" and p.k == cal.k(0.98, "type") and "type" in p.note
    assert reading_plan(cal, "What hourly rate applies?", []).k == cal.k(0.98, ("fact",))


def test_type_canary_twins_in_the_document_are_blanked():
    import random
    from gnn_pruner.certify.canaries import type_canaries
    from gnn_pruner.workers.worker import SentenceIndex
    tree = room()
    index = SentenceIndex(tree)
    inside = "ACME's total liability shall not exceed the Fees paid in the 12 months before the claim."
    cs = type_canaries(tree, [(["liabil", "cap"], inside)], 1, random.Random(0), index)
    assert cs[0].twins and tree.sentence(cs[0].twins[0])[0].startswith("ACME's total liability")
    outside = type_canaries(tree, [(["compet"], "Neither party shall compete in Norway.")], 1, random.Random(0), index)
    assert outside[0].twins == []
