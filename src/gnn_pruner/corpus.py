"""G1: generated training corpus of long contracts, with questions whose answers are known.

The GNN track trains on these documents only. They share no topics or sentence
templates with the test documents (``examples/``, including the scale set's
distractors), so the test sets stay unseen.

Every fact sentence the generator writes records a question about it (and a
paraphrased variant that avoids its key words), so Silver Path labels can be scored
against the true answer. Structure follows the spec's four layers:

    ## 7. Payroll Processing                 L1 chapter
    ### 7.2 Timing                           L2 sub-section
    paragraph of 1-3 sentences, or a list    L3
    sentence / list item                     L4

Cross-chapter facts pair a duty in one chapter with an exception elsewhere that
cites it ("Notwithstanding Section 12.2, ..."); their questions need both (G4).

    python -m gnn_pruner.corpus --out data/corpus_v1 --docs 48 --embedder st:all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WORDS_PER_PAGE = 500  # design decision D3

PARTIES = [("Supplier", "Customer"), ("Provider", "Client"), ("Contractor", "Company"),
           ("Licensor", "Licensee"), ("Vendor", "Purchaser"), ("Consultant", "Client")]
KINDS = ["Master Services Agreement", "Outsourcing Agreement", "Software Licence Agreement",
         "Consultancy Agreement", "Supply Agreement", "Managed Services Agreement"]

# Topic -> paraphrase that avoids the topic's words (for semantic questions).
TOPICS: Dict[str, str] = {
    "Hosting Services": "the servers our systems run on",
    "Data Backups": "the spare copies of our information",
    "Penetration Testing": "the simulated hacking exercises",
    "Vendor Onboarding": "bringing new suppliers on board",
    "Warehouse Operations": "running the storage depots",
    "Fleet Maintenance": "upkeep of the vehicles",
    "Payroll Processing": "running staff salaries",
    "Call Centre Support": "the telephone help desk",
    "Network Monitoring": "watching the communications links",
    "Disaster Recovery": "getting back up after an outage",
    "Translation Services": "converting documents into other languages",
    "Marketing Materials": "promotional content",
    "Catering Services": "food and drink provision",
    "Facilities Management": "looking after the buildings",
    "Waste Disposal": "getting rid of rubbish",
    "Energy Supply": "electricity provision",
    "Cloud Storage": "online file space",
    "Device Procurement": "buying laptops and phones",
    "Identity Management": "user logins and permissions",
    "Mobile Applications": "the phone apps",
    "Clinical Trials": "the medical studies",
    "Freight Forwarding": "shipping goods abroad",
    "Customs Clearance": "border paperwork",
    "Quality Inspections": "checking products meet standards",
    "Product Recalls": "pulling faulty goods from sale",
    "Spare Parts": "replacement components",
    "Field Engineering": "on-site technicians",
    "Learning Platforms": "the online courses",
    "Security Guarding": "guards at the premises",
    "Print Services": "printing jobs",
    "Courier Deliveries": "parcel drops",
    "Market Research": "studying customer opinions",
    "Payment Processing": "handling card transactions",
    "Tax Reporting": "filing tax returns",
    "Archive Storage": "keeping old boxes of files",
    "Event Management": "organising conferences",
    "Recruitment Services": "hiring new staff",
    "Data Analytics": "crunching the numbers",
    "Laboratory Testing": "lab analysis",
    "Vehicle Leasing": "renting cars",
    "Telecoms Services": "phone lines",
    "Helpdesk Ticketing": "logging support requests",
}
ASPECTS = ["Scope", "Timing", "Charges", "Records", "Remedies", "Reporting", "Responsibilities"]
VERBS = {"provide": "supply", "maintain": "keep up", "update": "refresh", "deliver": "hand over",
         "review": "look over", "test": "try out", "document": "write up", "complete": "finish"}
RIGHTS = {"inspect": "look into", "suspend": "pause", "examine": "go through", "reject": "turn down"}
NUMBERS = ["five", "seven", "ten", "fourteen", "fifteen", "twenty", "twenty-one", "thirty",
           "forty-five", "sixty", "ninety"]
PERIODS = ["days", "business days", "weeks", "months"]
FREQUENCIES = {"calendar month": "monthly", "calendar quarter": "quarterly", "contract year": "yearly"}
CONDITIONS = {"obtained written approval": "got sign-off in writing",
              "submitted a remediation plan": "put forward a plan to fix things",
              "paid all outstanding Charges": "settled the unpaid bills",
              "completed the onboarding checklist": "finished the joining steps"}
REMEDIES = ["withhold payment", "appoint a replacement provider", "escalate the matter to the steering committee",
            "require a remediation plan", "claim Service Credits"]
EVENTS = ["planned shutdown", "public holiday period", "declared emergency", "system migration window"]
DEFINITIONS = {
    "Deliverables": "all documents, software and materials produced under a Statement of Work.",
    "Business Day": "a day other than a Saturday, Sunday or bank holiday.",
    "Key Personnel": "the individuals named as key personnel in a Statement of Work.",
    "Change Request": "a written request to vary the scope of the services.",
    "Acceptance Criteria": "the tests that Deliverables must pass before they are accepted.",
    "Security Incident": "any unauthorised access to systems used to perform the services.",
    "Service Credit": "a reduction in the Charges payable for a failure to meet a service level.",
    "Milestone": "an event or date by which a stage of the services must be completed.",
    "Work Product": "anything created by the parties in performing a Statement of Work.",
    "Steering Committee": "the governance body made up of senior representatives of both parties.",
    "Charges": "the fees payable for the services as set out in the pricing schedule.",
    "Statement of Work": "a document describing services to be performed under this agreement.",
}
FILLER = [
    "The parties shall cooperate in good faith in relation to the {obj}.",
    "All Deliverables relating to the {obj} must satisfy the Acceptance Criteria.",
    "The {a} shall ensure that its Key Personnel understand the requirements for the {obj}.",
    "Any Change Request affecting the {obj} shall be dealt with under the change procedure.",
    "Nothing in this Section limits the {b}'s other rights in respect of the {obj}.",
    "The {b} shall give the {a} reasonable access to its premises for the purposes of the {obj}.",
    "Each party shall bear its own internal costs of meetings about the {obj}.",
    "The {a} shall perform the {obj} with reasonable skill and care.",
]
LIST_ITEMS = ["keep accurate records", "follow the {b}'s reasonable instructions",
              "notify the {b} of any delay", "use suitably qualified staff",
              "comply with the {b}'s site rules", "protect the {b}'s property"]


@dataclass
class FactTemplate:
    kind: str
    aspect: str
    text: str
    question: str
    paraphrase: str


FACTS = [
    FactTemplate("deadline", "Timing",
                 "The {a} shall {verb} the {obj} within {n} {period} of a written request from the {b}.",
                 "Within what time must the {a} {verb} the {obj} after a written request?",
                 "How long does the {a} have to {verb_syn} {para} once asked?"),
    FactTemplate("notice", "Remedies",
                 "The {b} may {right} the {obj} on {n} {period}' written notice to the {a}.",
                 "How much notice must the {b} give to {right} the {obj}?",
                 "How far ahead must the {b} warn the {a} before it can {right_syn} {para}?"),
    FactTemplate("retention", "Records",
                 "Records relating to the {obj} shall be retained by the {a} for {n} years.",
                 "For how many years must records about the {obj} be retained?",
                 "How long do the files on {para} need to be held onto?"),
    FactTemplate("costs", "Charges",
                 "The {a} shall bear the cost of the {obj} except where the {b} requests additional work.",
                 "Who bears the cost of the {obj}?",
                 "Who pays for {para}?"),
    FactTemplate("condition", "Responsibilities",
                 "The {b} is not responsible for the {obj} unless the {a} has first {condition}.",
                 "When is the {b} responsible for the {obj}?",
                 "In what circumstances does the {b} take responsibility for {para}?"),
    FactTemplate("reporting", "Reporting",
                 "The {a} shall report on the {obj} to the {b} at least once in each {freq}.",
                 "How often must the {a} report on the {obj}?",
                 "How regularly does the {a} have to update the {b} about {para}?"),
    FactTemplate("remedy", "Remedies",
                 "If the {a} fails to {verb} the {obj}, the {b} may {remedy} after {n} {period}.",
                 "What can the {b} do if the {a} fails to {verb} the {obj}?",
                 "What happens if the {a} does not {verb_syn} {para}?"),
    FactTemplate("cap", "Charges",
                 "The {a}'s total liability in respect of the {obj} shall not exceed EUR {amount}.",
                 "What is the limit on the {a}'s liability for the {obj}?",
                 "What is the most the {a} can be made to pay over {para}?"),
    FactTemplate("withhold", "Charges",
                 "The {b} may withhold {pct} percent of the Charges for the {obj} until the Acceptance Criteria are met.",
                 "What share of the Charges for the {obj} may the {b} withhold?",
                 "How much of the bill for {para} can the {b} hold back?"),
]


@dataclass
class Question:
    id: str
    document: str
    question: str
    query_type: str
    gold_evidence: List[str]
    answerable: bool = True
    reference_answer: str = ""
    tags: Dict = field(default_factory=dict)


@dataclass
class GeneratedDoc:
    doc_id: str
    text: str
    questions: List[Question]
    pages: int
    words: int


def _fill(template: str, **values) -> str:
    return template.format(**values)


def _topic_names(rng: random.Random, count: int) -> List[Tuple[str, str]]:
    """``count`` distinct (topic, paraphrase) names; topics repeat with a work-package
    qualifier once the list is used up."""
    base = list(TOPICS.items())
    out = []
    for i in range(count):
        if i % len(base) == 0:
            rng.shuffle(base)
        name, para = base[i % len(base)]
        round_ = i // len(base)
        if round_:
            name, para = f"{name} for Work Package {round_ + 1}", f"{para} for Work Package {round_ + 1}"
        out.append((name, para))
    return out


def generate_contract(doc_id: str, pages: int, seed: int, questions_per_doc: int = 24,
                      paraphrase_share: float = 0.4, cross_refs: int = 4) -> GeneratedDoc:
    rng = random.Random(seed)
    a, b = rng.choice(PARTIES)
    kind = rng.choice(KINDS)
    target_words = pages * WORDS_PER_PAGE
    lines = [f"# {kind} {doc_id}", ""]
    facts: List[Dict] = []

    # Chapter 1: definitions.
    terms = rng.sample(sorted(DEFINITIONS), k=rng.randint(7, len(DEFINITIONS)))
    lines += ["## 1. Definitions", ""]
    for term in terms:
        sentence = f"\"{term}\" means {DEFINITIONS[term]}"
        lines += [sentence, ""]
        facts.append({"kind": "definition", "answer": sentence, "chapter": 1,
                      "question": f"What does \"{term}\" mean?",
                      "paraphrase": f"How does the agreement define {term.lower()}?"})

    words = sum(len(l.split()) for l in lines)
    n_chapters_guess = max(2, int(target_words / 420))
    topics = _topic_names(rng, n_chapters_guess * 2 + 10)
    chapter = 1
    duties: List[Dict] = []
    while words < target_words or chapter < 3:
        chapter += 1
        name, para = topics[chapter - 2]
        obj = name.lower()
        chapter_lines = [f"## {chapter}. {name}", ""]
        filler = rng.sample(FILLER, k=len(FILLER))  # boilerplate, each used once per chapter
        templates = rng.sample(FACTS, k=rng.randint(4, len(FACTS)))
        by_aspect: Dict[str, List[FactTemplate]] = {}
        for t in templates:
            by_aspect.setdefault(t.aspect, []).append(t)
        aspects = [x for x in ASPECTS if x in by_aspect] + \
            rng.sample([x for x in ASPECTS if x not in by_aspect], k=rng.randint(0, 1))
        for sub, aspect in enumerate(aspects, 1):
            number = f"{chapter}.{sub}"
            chapter_lines += [f"### {number} {aspect}", ""]
            sentences: List[str] = []
            for t in by_aspect.get(aspect, []):
                verb, right = rng.choice(sorted(VERBS)), rng.choice(sorted(RIGHTS))
                condition = rng.choice(sorted(CONDITIONS))
                freq = rng.choice(sorted(FREQUENCIES))
                values = dict(a=a, b=b, obj=obj, para=para, verb=verb, verb_syn=VERBS[verb],
                              right=right, right_syn=RIGHTS[right], n=rng.choice(NUMBERS),
                              period=rng.choice(PERIODS), freq=freq, condition=condition,
                              condition_syn=CONDITIONS[condition], remedy=rng.choice(REMEDIES),
                              amount=f"{rng.randint(1, 40) * 25},000", pct=rng.choice(NUMBERS[:6]))
                sentence = _fill(t.text, **values)
                sentences.append(sentence)
                fact = {"kind": t.kind, "answer": sentence, "chapter": chapter, "section": number,
                        "question": _fill(t.question, **values),
                        "paraphrase": _fill(t.paraphrase, **values)}
                facts.append(fact)
                if t.kind == "deadline":
                    duties.append({**fact, "verb": verb, "obj": obj, "para": para})
            for _ in range(min(rng.randint(1, 3), len(filler))):
                sentences.append(_fill(filler.pop(), a=a, b=b, obj=obj))
            rng.shuffle(sentences)
            # Group into paragraphs of 1-3 sentences; sometimes add a list paragraph.
            i = 0
            while i < len(sentences):
                size = rng.randint(1, 3)
                chapter_lines += [" ".join(sentences[i:i + size]), ""]
                i += size
            if rng.random() < 0.3:
                items = rng.sample(LIST_ITEMS, k=3)
                body = "; ".join(f"({'abc'[j]}) {_fill(x, b=b)}" for j, x in enumerate(items[:2]))
                chapter_lines += [f"In performing the {obj}, the {a} shall: {body}; and "
                                  f"(c) {_fill(items[2], b=b)}.", ""]
        lines += chapter_lines
        words += sum(len(l.split()) for l in chapter_lines)

    # Cross-chapter exceptions: a later chapter qualifies a duty in another chapter.
    chosen = rng.sample(duties, k=min(cross_refs, len(duties)))
    exceptions = []
    for duty in chosen:
        event = rng.choice(EVENTS)
        sentence = (f"Notwithstanding Section {duty['section']}, the {a} need not {duty['verb']} "
                    f"the {duty['obj']} during any {event}.")
        exceptions.append(sentence)
        facts.append({"kind": "cross_chapter", "answer": [duty["answer"], sentence],
                      "chapter": chapter + 1, "section": duty["section"],
                      "question": f"Are there exceptions to the {a}'s duty to {duty['verb']} the {duty['obj']}?",
                      "paraphrase": f"Is the {a} ever excused from having to {VERBS[duty['verb']]} {duty['para']}?"})
    if exceptions:
        lines += [f"## {chapter + 1}. Exceptions and Relief", ""]
        for i, sentence in enumerate(exceptions, 1):
            lines += [f"### {chapter + 1}.{i} Relief Event {i}", "", sentence, ""]

    text = "\n".join(lines).rstrip() + "\n"
    questions = _questions(doc_id, facts, rng, questions_per_doc, paraphrase_share)
    return GeneratedDoc(doc_id, text, questions, pages, len(text.split()))


def _questions(doc_id: str, facts: List[Dict], rng: random.Random, limit: int,
               paraphrase_share: float) -> List[Question]:
    cross = [f for f in facts if f["kind"] == "cross_chapter"]
    single = [f for f in facts if f["kind"] != "cross_chapter"]
    picked = rng.sample(single, k=min(limit, len(single))) + cross
    out = []
    for i, fact in enumerate(picked, 1):
        paraphrased = fact["kind"] != "definition" and rng.random() < paraphrase_share
        gold = fact["answer"] if isinstance(fact["answer"], list) else [fact["answer"]]
        out.append(Question(
            id=f"{doc_id}-q{i:03d}", document="", query_type=(
                "definition_lookup" if fact["kind"] == "definition" else
                "cross_reference_reasoning" if fact["kind"] == "cross_chapter" else "fact_lookup"),
            question=fact["paraphrase"] if paraphrased else fact["question"],
            gold_evidence=gold,
            tags={"kind": fact["kind"], "paraphrase": paraphrased,
                  "cross_chapter": fact["kind"] == "cross_chapter",
                  "chapter": fact["chapter"], "section": fact.get("section")}))
    return out


def page_counts(n: int, min_pages: int, max_pages: int, seed: int) -> List[int]:
    """Log-uniform document lengths: many short contracts, a few long ones."""
    rng = random.Random(seed)
    return [max(min_pages, int(round(math.exp(rng.uniform(math.log(min_pages), math.log(max_pages))))))
            for _ in range(n)]


def build_corpus(out: str, docs: int = 48, seed: int = 7, min_pages: int = 2, max_pages: int = 120,
                 val_fraction: float = 0.2, embedder_spec: Optional[str] = None,
                 questions_per_doc: int = 24) -> Dict:
    """Generate documents, split them by document into train / val, export each as a
    four-layer Parquet shard (with embeddings if ``embedder_spec``), and write the
    questions. Returns the manifest (also written to ``manifest.json``)."""
    from .document_graph import DocumentGraph
    from .layers import check, four_layer
    from .parquet_store import write_document

    root = Path(out)
    started = time.perf_counter()
    embedder = None
    if embedder_spec:
        from .embeddings import load_embedder

        embedder = load_embedder(embedder_spec)
    sizes = page_counts(docs, min_pages, max_pages, seed)
    rng = random.Random(seed)
    order = list(range(docs))
    rng.shuffle(order)
    val = set(order[:max(1, int(round(docs * val_fraction)))])
    manifest: Dict = {"seed": seed, "docs": [], "words_per_page": WORDS_PER_PAGE,
                      "embedder": embedder.name if embedder else None}
    questions: Dict[str, List[Question]] = {"train": [], "val": []}
    for i, pages in enumerate(sizes):
        doc_id = f"gen{seed:02d}-{i:03d}"
        split = "val" if i in val else "train"
        doc = generate_contract(doc_id, pages, seed=seed * 1000 + i, questions_per_doc=questions_per_doc)
        path = root / "docs" / split / f"{doc_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc.text, encoding="utf-8")
        graph = DocumentGraph.from_file(str(path))
        layered = four_layer(graph, doc_id)
        problems = check(layered)
        if problems:
            raise ValueError(f"{doc_id}: invalid four-layer graph: {problems[:3]}")
        embeddings = None
        if embedder is not None:
            embeddings = embed_layered(layered, embedder)
        write_document(layered, str(root / "parquet"), embeddings)
        for q in doc.questions:
            q.document = f"docs/{split}/{doc_id}.md"
            q.tags["split"] = split
        questions[split] += doc.questions
        manifest["docs"].append({"doc_id": doc_id, "split": split, "pages": pages, "words": doc.words,
                                 "questions": len(doc.questions), **layered.stats()})
    for split, items in questions.items():
        with open(root / f"questions_{split}.jsonl", "w", encoding="utf-8") as f:
            for q in items:
                f.write(json.dumps(q.__dict__) + "\n")
    manifest["seconds"] = round(time.perf_counter() - started, 1)
    manifest["totals"] = {k: sum(d[k] for d in manifest["docs"])
                          for k in ("pages", "words", "questions", "L1", "L2", "L3", "L4", "virtual_L4")}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def embed_layered(layered, embedder, batch: int = 256):
    """(model, ids, matrix) for every L3 and L4 node. A virtual L4 node shares its
    paragraph's text, so it reuses the paragraph's vector."""
    ids = [n.id for n in sorted(layered.nodes.values(), key=lambda x: (x.position, x.layer, x.id))
           if n.layer >= 3]
    real = [nid for nid in ids if not layered.nodes[nid].virtual]
    texts = [layered.nodes[nid].text for nid in real]
    if hasattr(embedder, "fit"):
        embedder.fit(texts)
    import numpy as np

    vectors = np.vstack([embedder.encode(texts[i:i + batch]) for i in range(0, len(texts), batch)]) \
        if texts else np.zeros((0, 1), dtype=np.float32)
    row = {nid: vectors[i] for i, nid in enumerate(real)}
    matrix = np.vstack([row[nid] if nid in row else row[layered.nodes[nid].parent_id] for nid in ids])
    return embedder.name, ids, matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/corpus_v1")
    parser.add_argument("--docs", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-pages", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=120)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--questions-per-doc", type=int, default=24)
    parser.add_argument("--embedder", default=None, help="e.g. st:all-MiniLM-L6-v2 (stored in the Parquet shards)")
    args = parser.parse_args()
    manifest = build_corpus(args.out, args.docs, args.seed, args.min_pages, args.max_pages,
                            args.val_fraction, args.embedder, args.questions_per_doc)
    print(json.dumps({"totals": manifest["totals"], "seconds": manifest["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
