"""The spec's inference path end to end (spec §5 phases 1-6, single process):

    document -> four-layer graph -> coarse GNN on paragraph super-nodes -> keep rule
    -> FineGNN on the kept paragraphs' sentences -> merge (> 0.8 x mean)
    -> Phase 8 dependency rules -> evidence pack

No planner and no candidate retrieval (D9). Merged sentences become evidence items:
- a paragraph when it is a single claim, or when at least half of its sentences
  survived the merge
- otherwise the sentence itself

Items are ranked by fine score, capped at ``max_evidence`` (a safety cap, as in Phase
8), then dependencies are added (continuations, definitions, references, qualifiers,
conflicts; at most 4).

    python -m gnn_pruner.gnn.spec_qa --model models/two_stage_v1/joint --sets eval/contracts_v1.jsonl
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..document_graph import DocumentGraph
from ..evidence import EvidencePack, build_evidence_pack
from ..layers import four_layer
from ..pruning import PRIMARY, EvidenceNode, PruneConfig, PrunedSubgraph, dependencies
from ..query_planner import plan_query
from .cross_attention import load as load_models
from .two_stage import StageConfig, fine_table, run_stages

MAX_EVIDENCE = 8


@dataclass
class SpecAnswer:
    question: str
    evidence: EvidencePack
    pruned: PrunedSubgraph
    merged: List[str]                         # merged sentence ids (four-layer ids)
    paragraph_ranking: List[str] = field(default_factory=list)  # L3 ids, best first
    timings: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, float] = field(default_factory=dict)


class SpecQA:
    def __init__(self, graph: DocumentGraph, model_dir: str, embedder=None,
                 stage_config: Optional[StageConfig] = None, max_evidence: int = MAX_EVIDENCE,
                 layered=None, vectors=None, table=None, budgets=None):
        """``layered`` / ``vectors=(ids, matrix)``: a pre-built four-layer graph and its
        stored embeddings (e.g. a cached Parquet shard), to skip embedding."""
        from ..corpus import embed_layered
        from ..embeddings import load_embedder

        self.coarse, self.fine, self.meta = load_models(model_dir)  # G3 or G4 (with attention)
        self.embedder = embedder or load_embedder(self.meta["embedder"])
        if self.embedder.name != self.meta["embedder"]:
            raise ValueError(f"model trained with {self.meta['embedder']}, got {self.embedder.name}")
        self.graph = graph
        started = time.perf_counter()
        self.layered = layered if layered is not None else four_layer(graph, "doc")
        if table is not None:  # shared by several systems with the same pooling (saves memory)
            self.table = table
        else:
            if vectors is None:
                _, ids, matrix = embed_layered(self.layered, self.embedder)
            else:
                ids, matrix = vectors
            self.table = fine_table(self.layered, ids, matrix, self.meta.get("pooling", "mean"))
        self.index_seconds = time.perf_counter() - started
        self.config = stage_config or StageConfig(**self.meta["stage_config"])
        self.max_evidence = max_evidence
        self.budgets = budgets  # Phase 9 per-route token budgets (None = defaults, {} = none)

    @classmethod
    def from_file(cls, path: str, model_dir: str, **kwargs) -> "SpecQA":
        return cls(DocumentGraph.from_file(path), model_dir, **kwargs)

    def ask(self, question: str) -> SpecAnswer:
        t0 = time.perf_counter()
        qvec = np.asarray(self.embedder.encode_query(question), dtype=np.float32)
        t1 = time.perf_counter()
        result = run_stages(self.coarse, self.fine, self.table, question, qvec, self.config)
        t2 = time.perf_counter()
        table = self.table
        sentence_rows = [i for i, r in enumerate(result.rows) if table.layer[r] == 4]
        probs = result.fine_probs[sentence_rows]
        threshold = self.config.merge_ratio * probs.mean() if len(probs) else 0.0
        score = {table.ids[result.rows[sentence_rows[i]]]: float(p) for i, p in enumerate(probs)}
        # Paragraph ranking: paragraphs of the fine graph by their best sentence, then the rest
        # of the kept order (coarse), for Recall@k comparisons across systems.
        best: Dict[str, float] = {}
        for sid, p in score.items():
            pid = self.layered.nodes[sid].parent_id
            best[pid] = max(best.get(pid, 0.0), p)
        ranking = sorted(best, key=lambda n: -best[n])
        ranking += [table.doc.ids[c] for c in result.kept_l3 if table.doc.ids[c] not in best]
        merged = [nid for nid, p in sorted(score.items(), key=lambda kv: -kv[1]) if p > threshold]

        # Group merged sentences into evidence items.
        by_paragraph: Dict[str, List[str]] = {}
        for nid in merged:
            by_paragraph.setdefault(self.layered.nodes[nid].parent_id, []).append(nid)
        items: Dict[str, float] = {}
        for pid, kept in by_paragraph.items():
            claims = [c for c in self.layered.children(pid) if c.layer == 4]
            best = max(score[n] for n in kept)
            if len(claims) == 1 or claims[0].virtual or len(kept) * 2 >= len(claims):
                items[pid] = max(items.get(pid, 0.0), best)
            else:
                for n in kept:
                    items[n] = score[n]
        ranked = sorted(items, key=lambda n: -items[n])
        kept_items = [n for n in ranked if n in self.graph.nodes][:self.max_evidence]
        evidence = [EvidenceNode(self.graph.nodes[n], PRIMARY, items[n]) for n in kept_items]
        kept_ids = {e.node.id for e in evidence}
        added = 0
        for ev in list(evidence):
            for target, role in dependencies(self.graph, ev.node, PruneConfig()):
                if added >= 4:
                    break
                if target.id in kept_ids or target.parent_id in kept_ids:
                    continue
                evidence.append(EvidenceNode(target, role, items.get(target.id, 0.0), required_by=ev.node.id))
                kept_ids.add(target.id)
                added += 1
        pruned = PrunedSubgraph(evidence, {e.node.id: [a.id for a in self.graph.ancestors(e.node.id)]
                                           for e in evidence})
        pack = build_evidence_pack(self.graph, plan_query(question), pruned, budgets=self.budgets)
        t3 = time.perf_counter()
        n_sentences = max(len(table.sentences), 1)
        doc_words = sum(len(table.doc.layered.nodes[table.ids[r]].text.split()) for r in table.sentences)
        pack_words = sum(len(i.text.split()) for i in pack.items)
        return SpecAnswer(question, pack, pruned, merged, ranking,
                          timings={"encode_ms": (t1 - t0) * 1000, "coarse_ms": result.coarse_ms,
                                   "fine_ms": result.fine_ms, "merge_pack_ms": (t3 - t2) * 1000},
                          stats={"kept_paragraph_share": len(result.kept_l3) / max(len(table.doc.l3), 1),
                                 "merged_sentence_share": len(merged) / n_sentences,
                                 "pack_word_share": pack_words / max(doc_words, 1),
                                 "fine_nodes": len(result.rows), "document_nodes": len(table.ids)})


def evaluate(model_dir: str, path: str) -> Dict[str, float]:
    from ..evaluation import _norm, load_cases

    systems: Dict[str, SpecQA] = {}
    rows = []
    for case in load_cases(path):
        if not case.answerable:
            continue
        if case.document not in systems:
            systems[case.document] = SpecQA.from_file(case.document, model_dir)
        answer = systems[case.document].ask(case.question)
        texts = [_norm(i.text) for i in answer.evidence.items]
        primary = [_norm(i.text) for i in answer.evidence.items if i.role == PRIMARY]
        gold = [_norm(g) for g in case.gold_evidence]
        rows.append({
            "evidence_recall": sum(any(g in t for t in texts) for g in gold) / len(gold),
            "complete": float(all(any(g in t for t in texts) for g in gold)),
            "hit_at_1": float(bool(texts) and any(g in texts[0] for g in gold)),
            "primary_precision": (sum(any(g in t for g in gold) for t in primary) / len(primary)) if primary else 0.0,
            "pack_items": len(texts), "pack_words": sum(len(t.split()) for t in texts),
            **answer.stats, **answer.timings})
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]} | {"n": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/two_stage_v1/joint")
    parser.add_argument("--sets", nargs="+", default=["eval/contracts_v1.jsonl"])
    args = parser.parse_args()
    for path in args.sets:
        print(Path(path).stem, evaluate(args.model, path))


if __name__ == "__main__":
    main()
