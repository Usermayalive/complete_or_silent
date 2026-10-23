"""G6: the spec's tables 8-10, measured (spec §7, §8).

Systems:
- ``v3.0``: spec path (coarsening + FineGNN + cross-edge attention + merge + Phase 8
  dependencies)
- ``v3.0+cap``: the same with ``max_keep = 200`` (G3)
- ``g3``: two-stage without attention
- ``v2.0``: the spec's "no coarsening, no attention" column (FineGNN over every node)
- ``rule``: the Phase 4-8 rule-based pipeline

Commands:
- ``scale`` (Tables 8 and 10): generated 100 / 1,000 / 10,000-page documents with
  known-answer questions. Per system: latency per question, recall, pack size, tokens
  sent to the LLM as a share of the document, and throughput.
- ``quality`` (Table 9): Recall@10 (paragraphs), cross-chapter complete@10, evidence
  recall / complete / hit@1, structural coherence and pack share, on the test, dev,
  generated and adversarial sets.
- ``adversarial``: writes ``eval/adversarial_v1.jsonl`` (100 queries, deterministic)
  from generated validation questions. 20 base questions, each in 5 perturbations:
  distractor, wrong_number, party_swap, typo, paraphrase.
- ``sheet``: a blind A/B rating sheet (v3.0 vs rule-based evidence packs) for human
  preference.

Structural coherence (D11): for every primary evidence item, the context it needs
(definitions of terms it uses, clauses it cites) is looked up in the pack; reported as
the covered share. Ancestors / section paths are always present (each item carries its
citation path), so that part is 1.0 by construction and not reported.

Tokens are counted with the embedder's subword tokenizer (WordPiece, close to LLM
tokenizers). Cost is left as a formula; the per-token price is the user's to set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..evaluation import _norm
from ..models import CROSS_REFERENCE, DEFINITION_LINK, SENTENCE
from ..pruning import PRIMARY

ROOT = Path(__file__).resolve().parents[3]
SYSTEMS = {
    "v3.0": ("models/g4/cea_s0/joint", {}),
    "v3.0+cap": ("models/g4/cea_s0/joint", {"max_keep": 200}),
    "g3": ("models/two_stage_v1/joint", {}),
    "v2.0": ("models/g3_flat", {}),
    "rule": (None, {}),
}


# ---------------------------------------------------------------- shared

class PrecomputedEmbedder:
    """The real embedder, but texts with a stored vector are not re-encoded (cached
    Parquet shards). Same name, so models and calibrations still match."""

    def __init__(self, base, texts: Sequence[str], matrix: np.ndarray):
        self.base, self.name, self.cacheable = base, base.name, True
        self.lookup = {t: matrix[i] for i, t in enumerate(texts)}

    def fit(self, texts):
        return self

    def encode(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.lookup))
        if missing:
            for t, v in zip(missing, self.base.encode(missing)):
                self.lookup[t] = v
        return np.vstack([self.lookup[t] for t in texts]).astype(np.float32)

    def encode_query(self, text):
        return self.base.encode_query(text)


def token_counter():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        return lambda text: len(tok.tokenize(text))
    except Exception:  # noqa: BLE001 - fall back to a word-based estimate
        return lambda text: int(round(len(text.split()) * 1.3))


def coherence(graph, items) -> Optional[float]:
    """Share of primary items' required context (definitions used, clauses cited)
    present in the pack."""
    in_pack = {i.node_id for i in items}
    in_pack |= {graph.nodes[n].parent_id for n in list(in_pack) if graph.nodes[n].parent_id}
    needed, found = 0, 0
    for item in items:
        if item.role != PRIMARY:
            continue
        node = graph.nodes[item.node_id]
        source = graph.parent(node.id) if node.node_type == SENTENCE and node.parent_id else node
        for nbr, _, _ in graph.neighbors(source.id, (DEFINITION_LINK, CROSS_REFERENCE), "out"):
            target = nbr if nbr.is_content else next((c for c in graph.children(nbr.id) if c.is_content), None)
            if target is None:
                continue
            needed += 1
            found += target.id in in_pack or target.parent_id in in_pack
    return found / needed if needed else None


def case_metrics(graph, layered, items, ranking: Sequence[str], gold: Sequence[str],
                 doc_words: int) -> Dict[str, float]:
    from ..silver import gold_nodes

    texts = [_norm(i.text) for i in items]
    g = [_norm(x) for x in gold]
    groups = [paras for _, paras in gold_nodes(layered, gold) if paras]
    top10 = set(ranking[:10])
    chapters = {layered.path(p)[0] for grp in groups for p in grp}
    out = {
        "recall@10": float(np.mean([bool(grp & top10) for grp in groups])) if groups else 0.0,
        "complete@10": float(all(grp & top10 for grp in groups)) if groups else 0.0,
        "evidence_recall": sum(any(x in t for t in texts) for x in g) / len(g),
        "complete": float(all(any(x in t for t in texts) for x in g)),
        "hit@1": float(bool(texts) and any(x in texts[0] for x in g)),
        "pack_words": sum(len(t.split()) for t in texts),
        "pack_share": sum(len(t.split()) for t in texts) / max(doc_words, 1),
        "cross_chapter": float(len(chapters) >= 2),
    }
    c = coherence(graph, items)
    if c is not None:
        out["coherence"] = c
    return out


def rule_ranking(result, layered) -> List[str]:
    out, seen = [], set()
    for s in result.scored:
        node = layered.nodes.get(s.node.id)
        if node is None:
            continue
        pid = node.id if node.layer == 3 else (node.parent_id if node.layer == 4 else None)
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


class Systems:
    """Every system, built once per document."""

    def __init__(self, names: Sequence[str], embedder):
        self.names, self.embedder = list(names), embedder

    def build(self, graph, layered=None, vectors=None, semantic_edges: bool = True) -> Dict[str, object]:
        from ..layers import four_layer
        from ..pipeline import DocumentQA
        from .spec_qa import SpecQA
        from .two_stage import StageConfig

        layered = layered if layered is not None else four_layer(graph, "doc")
        if vectors is None:
            from ..corpus import embed_layered

            _, ids, matrix = embed_layered(layered, self.embedder)
            vectors = (ids, matrix)
        from .two_stage import fine_table

        built, timings = {}, {}
        shared = fine_table(layered, vectors[0], vectors[1], "mean") if any(n != "rule" for n in self.names) else None
        for name in self.names:
            model, overrides = SYSTEMS[name]
            t = time.perf_counter()
            if name == "rule":
                texts = [layered.nodes[i].text for i in vectors[0]]
                embedder = PrecomputedEmbedder(self.embedder, texts, vectors[1])
                built[name] = DocumentQA(graph, embedder=embedder, semantic_edges=semantic_edges)
            else:
                qa = SpecQA(graph, str(ROOT / model), self.embedder, layered=layered, vectors=vectors,
                            table=shared)
                if overrides:
                    qa.config = StageConfig(**{**qa.config.__dict__, **overrides})
                built[name] = qa
            timings[name] = time.perf_counter() - t
        built["_layered"], built["_build_seconds"] = layered, timings
        return built

    @staticmethod
    def ask(name: str, system, question: str, layered):
        t = time.perf_counter()
        if name == "rule":
            result = system.ask(question)
            ms = (time.perf_counter() - t) * 1000
            return result.evidence.items, rule_ranking(result, layered), ms
        answer = system.ask(question)
        ms = (time.perf_counter() - t) * 1000
        return answer.evidence.items, answer.paragraph_ranking, ms


def mean(rows: List[Dict], keys=None) -> Dict[str, float]:
    keys = keys or sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        values = [r[k] for r in rows if k in r]
        if values:
            out[k] = round(float(np.mean(values)), 4)
    out["n"] = len(rows)
    return out


# ---------------------------------------------------------------- adversarial suite

PARTY_SWAP = {"Supplier": "Customer", "Customer": "Supplier", "Provider": "Client", "Client": "Provider",
              "Contractor": "Company", "Company": "Contractor", "Licensor": "Licensee",
              "Licensee": "Licensor", "Vendor": "Purchaser", "Purchaser": "Vendor", "Consultant": "Client"}
WRONG_NUMBERS = ["eleven", "twelve", "eighteen", "twenty-five", "forty", "seventy", "eighty"]


def _typos(text: str, rng: random.Random, n: int = 2) -> str:
    words = text.split()
    candidates = [i for i, w in enumerate(words) if len(w) > 5 and w[0].islower()]
    for i in rng.sample(candidates, k=min(n, len(candidates))):
        w = words[i]
        j = rng.randrange(1, len(w) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def build_adversarial(corpus: str = "data/corpus_v1", out: str = "eval/adversarial_v1.jsonl",
                      bases: int = 20, seed: int = 13) -> int:
    from ..corpus import TOPICS, VERBS

    rng = random.Random(seed)
    root = ROOT / corpus
    questions = [json.loads(l) for l in (root / "questions_val.jsonl").read_text().splitlines() if l]
    lexical = [q for q in questions if not q["tags"].get("paraphrase")
               and q["tags"]["kind"] not in ("definition", "cross_chapter")]
    rng.shuffle(lexical)
    picked = lexical[:bases]
    # Topics present in each document, for distractors.
    topics_by_doc: Dict[str, List[str]] = {}
    for q in questions:
        for t in TOPICS:
            if t.lower() in " ".join(q["gold_evidence"]).lower():
                topics_by_doc.setdefault(q["document"], []).append(t)
    rows = []
    for i, q in enumerate(picked, 1):
        text = q["question"]
        topic = next((t for t in TOPICS if t.lower() in text.lower()), None)
        others = [t for t in topics_by_doc.get(q["document"], []) if t != topic] or ["warehouse operations"]
        variants = {
            "distractor": f"{text} I am not asking about the {rng.choice(others).lower()}.",
            "wrong_number": f"{text} Is it {rng.choice(WRONG_NUMBERS)} days?",
            "party_swap": re.sub(r"\b(" + "|".join(PARTY_SWAP) + r")\b", lambda m: PARTY_SWAP[m.group(1)], text),
            "typo": _typos(text, rng),
            "paraphrase": text,
        }
        para = text
        if topic:
            para = re.sub(re.escape(topic.lower()), TOPICS[topic], para, flags=re.I)
        for verb, syn in VERBS.items():
            para = re.sub(rf"\b{verb}\b", syn, para)
        variants["paraphrase"] = para
        for kind, variant in variants.items():
            rows.append({"id": f"adv-{i:02d}-{kind}", "document": f"../{corpus}/{q['document']}",
                         "question": variant, "query_type": q["query_type"], "gold_evidence": q["gold_evidence"],
                         "answerable": True, "reference_answer": "",
                         "tags": {"base_id": q["id"], "base_question": text, "perturbation": kind}})
    path = ROOT / out
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


# ---------------------------------------------------------------- quality (Table 9)

def run_quality(sets: Sequence[str], names: Sequence[str]) -> Dict:
    from ..document_graph import DocumentGraph
    from ..embeddings import load_embedder
    from ..evaluation import load_cases

    embedder = load_embedder("st:all-MiniLM-L6-v2")
    systems = Systems(names, embedder)
    report = {}
    for path in sets:
        cases = [c for c in load_cases(path) if c.answerable]
        built: Dict[str, Dict] = {}
        rows: Dict[str, List[Dict]] = {n: [] for n in names}
        base_rows: Dict[str, Dict[str, Dict]] = {n: {} for n in names}
        for case in cases:
            if case.document not in built:
                graph = DocumentGraph.from_file(case.document)
                built[case.document] = (graph, systems.build(graph))
            graph, sys_ = built[case.document]
            layered = sys_["_layered"]
            doc_words = sum(len(n.text.split()) for n in layered.nodes.values() if n.layer == 4)
            for name in names:
                items, ranking, ms = Systems.ask(name, sys_[name], case.question, layered)
                m = case_metrics(graph, layered, items, ranking, case.gold_evidence, doc_words)
                m["ms"] = ms
                m["perturbation"] = case.tags.get("perturbation")
                rows[name].append(m)
        summary = {}
        for name, r in rows.items():
            numeric = [{k: v for k, v in x.items() if isinstance(v, (int, float)) and k != "cross_chapter"}
                       for x in r]
            entry = {"all": mean(numeric)}
            cross = [x for x, raw in zip(numeric, r) if raw["cross_chapter"]]
            if cross:
                entry["cross_chapter"] = mean(cross, ["recall@10", "complete@10", "complete", "evidence_recall"])
            kinds = sorted({raw["perturbation"] for raw in r if raw["perturbation"]})
            for kind in kinds:
                entry[kind] = mean([x for x, raw in zip(numeric, r) if raw["perturbation"] == kind],
                                   ["recall@10", "evidence_recall", "hit@1"])
            summary[name] = entry
        report[Path(path).stem] = summary
        print(f"== {Path(path).stem} ({len(cases)} questions)", flush=True)
        for name, entry in summary.items():
            a = entry["all"]
            print(f"  {name:<9} R@10 {a.get('recall@10')} complete {a.get('complete')} ev_recall "
                  f"{a.get('evidence_recall')} hit@1 {a.get('hit@1')} coherence {a.get('coherence')} "
                  f"pack {a.get('pack_words')}w share {a.get('pack_share')} {a.get('ms')}ms "
                  f"| cross {entry.get('cross_chapter')}", flush=True)
    return report


# ---------------------------------------------------------------- scale (Tables 8, 10)

def cached_document(pages: int, embedder, cache: str = "data/bench", questions: int = 8):
    """Generated document of ``pages`` pages (seed 21, as in G3/G5 benches), its graph,
    four-layer graph and stored vectors; embeddings cached as Parquet."""
    import tempfile

    from ..corpus import embed_layered, generate_contract
    from ..document_graph import DocumentGraph
    from ..layers import four_layer
    from ..parquet_store import read_document, read_embeddings, write_document

    doc = generate_contract(f"bench{pages}", pages, seed=21, questions_per_doc=questions)
    graph = DocumentGraph.from_text(doc.text, title=f"bench{pages}")
    shard = ROOT / cache / f"bench{pages}" / f"bench{pages}"
    if not (shard / "embeddings.parquet").exists():
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "doc.md"
            p.write_text(doc.text)
            layered = four_layer(DocumentGraph.from_file(str(p)), f"bench{pages}")
        write_document(layered, str(shard.parent), embed_layered(layered, embedder))
    layered = read_document(str(shard))
    _, ids, matrix = read_embeddings(str(shard))
    return doc, graph, layered, (ids, matrix)


def run_scale(pages_list: Sequence[int], names: Sequence[str], flat_max_nodes: int = 120_000,
              rule_semantic_max: int = 5_000) -> List[Dict]:
    from ..embeddings import load_embedder

    embedder = load_embedder("st:all-MiniLM-L6-v2")
    count = token_counter()
    results = []
    for pages in pages_list:
        t = time.perf_counter()
        doc, graph, layered, vectors = cached_document(pages, embedder)
        prep = time.perf_counter() - t
        nodes = len(layered.nodes)
        paragraphs = sum(1 for n in layered.nodes.values() if n.layer == 3)
        use = [n for n in names if not (n == "v2.0" and nodes > flat_max_nodes)]
        systems = Systems(use, embedder).build(graph, layered, vectors,
                                               semantic_edges=paragraphs <= rule_semantic_max)
        doc_text = " ".join(n.text for n in layered.nodes.values() if n.layer == 4 and not n.virtual) + " " + \
            " ".join(n.text for n in layered.nodes.values() if n.layer == 3 and
                     all(c.virtual for c in layered.children(n.id)))
        doc_tokens = count(doc_text)  # same counter as the packs
        doc_words = len(doc_text.split())
        row = {"pages": pages, "words": doc.words, "nodes": nodes, "paragraphs": paragraphs,
               "sentences": sum(1 for n in layered.nodes.values() if n.layer == 4),
               "doc_tokens": doc_tokens, "prepare_s": round(prep, 1),
               "build_s": {k: round(v, 1) for k, v in systems["_build_seconds"].items()},
               "rule_semantic_edges": paragraphs <= rule_semantic_max, "systems": {}}
        for name in names:
            if name not in use:
                row["systems"][name] = "not run: needs more memory than this 8 GB machine"
                continue
            per = []
            for q in doc.questions:
                items, ranking, ms = Systems.ask(name, systems[name], q.question, layered)
                m = case_metrics(graph, layered, items, ranking, q.gold_evidence, doc_words)
                pack_text = " ".join(i.text for i in items)
                m.update(ms=ms, pack_tokens=count(pack_text))
                m["token_share"] = m["pack_tokens"] / max(doc_tokens, 1)
                per.append({k: v for k, v in m.items() if k != "cross_chapter"})
            s = mean(per)
            s["p50_ms"] = round(float(np.percentile([p["ms"] for p in per], 50)), 1)
            s["p99_ms"] = round(float(np.percentile([p["ms"] for p in per], 99)), 1)
            s["queries_per_hour"] = round(3_600_000 / s["ms"], 0) if s["ms"] else None
            row["systems"][name] = s
            print(json.dumps({"pages": pages, "system": name, **{k: s.get(k) for k in (
                "recall@10", "complete", "evidence_recall", "hit@1", "coherence", "pack_tokens",
                "token_share", "p50_ms", "p99_ms")}}), flush=True)
        results.append(row)
    return results


# ---------------------------------------------------------------- human preference sheet

def write_sheet(path: str = "eval/contracts_v1.jsonl", n: int = 20, seed: int = 5,
                out: str = "docs/human_preference_sheet.md") -> None:
    from ..document_graph import DocumentGraph
    from ..embeddings import load_embedder
    from ..evaluation import load_cases

    rng = random.Random(seed)
    embedder = load_embedder("st:all-MiniLM-L6-v2")
    systems = Systems(["v3.0", "rule"], embedder)
    cases = [c for c in load_cases(str(ROOT / path)) if c.answerable]
    rng.shuffle(cases)
    built, key, lines = {}, {}, []
    lines += ["# Human preference rating sheet (G6)", "",
              "For each question, read the two evidence packs. Which one lets you answer the question "
              "better, with less to read? Mark **A**, **B** or **Same** in the Rating line, and optionally "
              "whether the preferred pack is *sufficient* on its own. The system behind A / B is hidden "
              "(key: `docs/human_preference_key.json`; please don't open it until you have finished).", ""]
    for i, case in enumerate(cases[:n], 1):
        if case.document not in built:
            graph = DocumentGraph.from_file(case.document)
            built[case.document] = systems.build(graph)
        sys_ = built[case.document]
        packs = {}
        for name in ("v3.0", "rule"):
            items, _, _ = Systems.ask(name, sys_[name], case.question, sys_["_layered"])
            packs[name] = items
        order = ["v3.0", "rule"]
        rng.shuffle(order)
        key[f"Q{i}"] = {"A": order[0], "B": order[1], "case": case.id}
        lines += [f"## Q{i}. {case.question}", "", f"*Reference answer:* {case.reference_answer}", ""]
        for label, name in zip("AB", order):
            lines.append(f"**Pack {label}** ({len(packs[name])} items, "
                         f"{sum(len(x.text.split()) for x in packs[name])} words)")
            lines += [f"- [{x.section_path}] {x.text}" for x in packs[name]]
            lines.append("")
        lines += ["Rating: A / B / Same — preferred pack sufficient on its own? yes / no", "", "---", ""]
    (ROOT / out).write_text("\n".join(lines))
    (ROOT / "docs" / "human_preference_key.json").write_text(json.dumps(key, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scale")
    s.add_argument("--pages", nargs="+", type=int, default=[100, 1000, 10000])
    s.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    s.add_argument("--json")
    q = sub.add_parser("quality")
    q.add_argument("--sets", nargs="+", required=True)
    q.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    q.add_argument("--json")
    sub.add_parser("adversarial")
    h = sub.add_parser("sheet")
    h.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    if args.cmd == "adversarial":
        print(build_adversarial(), "queries written to eval/adversarial_v1.jsonl")
    elif args.cmd == "sheet":
        write_sheet(n=args.n)
        print("wrote docs/human_preference_sheet.md")
    else:
        report = run_scale(args.pages, args.systems) if args.cmd == "scale" else run_quality(args.sets, args.systems)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
