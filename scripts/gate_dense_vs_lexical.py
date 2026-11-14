"""Gate-level RQ1/RQ4: lexical vs dense vs fused chapter ranking, with type canaries.
No LLM calls; MiniLM (local). Room: contracts M (977 pp, 55 chapters)."""
import re, sys, random, numpy as np
from gnn_pruner.evaluation import load_cases
from gnn_pruner.tree import tree_from_file
from gnn_pruner.workers.routing import route, words
from gnn_pruner.workers.worker import SentenceIndex
from gnn_pruner.embeddings import load_embedder
from gnn_pruner.certify.canaries import cuad_type_library, conformal_k, clopper_pearson_upper
from gnn_pruner.longbench import CONTRACT_QUESTIONS

tier = sys.argv[1] if len(sys.argv) > 1 else "M"
cases = [c for c in load_cases(f"eval/long/contracts_{tier}.jsonl") if c.gold_evidence and c.tags.get("kind") == "cuad"]
tree = tree_from_file(cases[0].document); index = SentenceIndex(tree)
chapters = [c.id for c in tree.chapters() if c.role != "contents"]; ch_pos = {c: i for i, c in enumerate(chapters)}
emb = load_embedder("st:all-MiniLM-L6-v2")
passages = [p for c in chapters for p in tree.passages(c)]
p_ch = np.array([ch_pos[c] for c in chapters for _ in tree.passages(c)])
P = np.asarray(emb.encode([p.text for p in passages]), dtype=np.float32); P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
def encq(texts):
    Q = np.asarray([emb.encode_query(t) for t in texts], dtype=np.float32).reshape(len(texts), -1)
    return Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
def chapter_best(sims):
    best = np.full(len(chapters), -1.0, dtype=np.float32); np.maximum.at(best, p_ch, sims); return best
def lexical_best(terms):
    b = index.best_per_chapter(terms); return np.array([b.get(c, 0.0) for c in chapters], dtype=np.float32)
def ranks_of(score_vec, own_idx):   # pessimistic ties
    return int((score_vec >= score_vec[own_idx]).sum())
def fused_rank(lex, den, own_idx, K=10):
    rl = np.argsort(np.argsort(-lex, kind="stable"), kind="stable"); rd = np.argsort(np.argsort(-den, kind="stable"), kind="stable")
    f = 1.0 / (K + rl + 1) + 1.0 / (K + rd + 1); return int((f >= f[own_idx]).sum())

# ---- type canaries: real CUAD clause of category X from outside the room, planted in a random chapter
lib = cuad_type_library(exclude_titles=[c.title for c in tree.chapters()])
rng = random.Random(0); sample = rng.sample(lib, 300)
qtext = {}
for cat, tmpl in CONTRACT_QUESTIONS.items():
    qtext[tuple(w for w in words(tmpl.format(X="the agreement")) if w != "agreement")] = tmpl.format(X="the agreement")
can = {"lexical": [], "dense": [], "fused": []}
Cq = encq([qtext.get(tuple(q), " ".join(q)) for q, _ in sample]); Cc = np.asarray(emb.encode([t for _, t in sample]), dtype=np.float32)
Cc /= np.linalg.norm(Cc, axis=1, keepdims=True) + 1e-9
for i, (q, text) in enumerate(sample):
    ch = rng.randrange(len(chapters))
    lex = lexical_best(list(q)); from gnn_pruner.index import BM25
    # canary lexical score in its chapter: its own BM25 score against the room's statistics (approx.: idf of the room)
    bm = index.bm25; cols = [bm.vocab[w] for w in set(q) if w in bm.vocab]
    toks = words(text); L = len(toks); avg = np.mean([len(words(t)) for t in index.texts[:2000]])
    s_c = sum(bm.idf[cw] * toks.count(w) * 2.5 / (toks.count(w) + 1.5 * (0.25 + 0.75 * L / avg)) for w in set(q) if w in bm.vocab for cw in [bm.vocab[w]])
    lex[ch] = max(lex[ch], s_c)
    den = chapter_best(P @ Cq[i]); den[ch] = max(den[ch], float(Cc[i] @ Cq[i]))
    can["lexical"].append(ranks_of(lex, ch)); can["dense"].append(ranks_of(den, ch)); can["fused"].append(fused_rank(lex, den, ch))

# ---- real CUAD questions
def unit_chapters(title):
    return {ch.id for ch in tree.chapters() if re.sub(r"\s*\(\d+/\d+\)$", "", ch.title) == title and ch.role == "body"}
real = {"lexical": [], "dense": [], "fused": []}
Rq = encq([c.question for c in cases])
for j, c in enumerate(cases):
    allowed = unit_chapters(c.tags.get("unit")) or None
    gold = {index.best_chapter(g, exclude=set(index.chapter_names) - allowed if allowed else set()) for g in c.gold_evidence} - {None}
    if not gold: continue
    terms = route(tree, c.question).content or words(c.question)
    lex = lexical_best(terms); den = chapter_best(P @ Rq[j])
    for name, fn in (("lexical", lambda g: ranks_of(lex, ch_pos[g])), ("dense", lambda g: ranks_of(den, ch_pos[g])),
                     ("fused", lambda g: fused_rank(lex, den, ch_pos[g]))):
        real[name].append(max(fn(g) for g in gold if g in ch_pos))
n = len(chapters)
print(f"room {tier}: {n} chapters, 300 type canaries, {len(real['lexical'])} real CUAD questions")
print("| Gate | α | k | canary miss | real miss (upper) |"); print("|---|---|---|---|---|")
for name in ("lexical", "dense", "fused"):
    for a in (0.9, 0.95, 0.98):
        k = conformal_k(can[name], a, n); m = sum(r > k for r in real[name]); N = len(real[name])
        print(f"| {name} | {a} | {k} | {sum(r>k for r in can[name])/len(can[name]):.3f} | {m/N:.3f} ({clopper_pearson_upper(m,N):.3f}) |")
