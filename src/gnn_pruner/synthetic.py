"""Synthetic contract text for retrieval-at-scale evaluation and benchmarks.

Distractor sections reuse a contract's parties and vocabulary so they compete with
the real clauses, while avoiding the topics of the eval set's unanswerable
questions (governing law, intellectual property, free trials, courts).
"""

from __future__ import annotations

import random

TOPICS = [
    "Audit Rights", "Insurance", "Subcontracting", "Publicity", "Export Controls",
    "Anti-Bribery", "Non-Solicitation", "Records Retention", "Security Testing",
    "Source Code Escrow", "Benchmarking", "Change Control", "Acceptance Testing",
    "Business Continuity", "Environmental Compliance", "Health and Safety",
    "Background Checks", "Equipment", "Travel Expenses", "Training", "Documentation",
    "Management Reporting", "Account Management", "Governance Meetings", "Step-In Rights",
    "Transition Assistance", "Most Favoured Customer", "Accessibility",
    "Open Source Software", "Data Migration", "Invoice Disputes", "Service Reviews",
    "Capacity Planning", "Release Management", "Customer Satisfaction Surveys",
]
NUMBERS = ["five", "ten", "fifteen", "twenty", "thirty", "forty-five", "sixty", "ninety"]
PERIODS = ["days", "business days", "weeks", "months"]
TEMPLATES = [
    "The {a} shall {verb} all {obj} within {n} {period} of a written request from the {b}.",
    "The {b} may {right} the {obj} on {n} {period}' written notice to the {a}.",
    "Any {obj} must be {done} in accordance with the {a}'s published {policy}.",
    "The {b} is not responsible for {obj} unless the {a} has first {condition}.",
    "If the {a} fails to {verb} the {obj}, the {b} may {remedy} after {n} {period}.",
    "Records relating to {obj} shall be retained by the {a} for {n} years.",
    "The {a} shall bear its own costs of {obj} except where the {b} requests additional work.",
    "Each party shall nominate a representative responsible for {obj}.",
    "The {a} shall report on {obj} to the {b} at least once in each calendar quarter.",
    "Charges for {obj} are included in the fees unless agreed otherwise in writing.",
]
VERBS = ["provide", "maintain", "update", "deliver", "review", "certify"]
RIGHTS = ["inspect", "suspend", "request changes to", "approve", "reject"]
DONE = ["documented", "approved", "recorded", "carried out", "reported"]
POLICIES = ["policies", "procedures", "standards", "guidelines"]
CONDITIONS = ["approved it in writing", "given reasonable notice", "agreed a plan", "paid the relevant charges"]
REMEDIES = ["withhold payment", "appoint a third party", "escalate the matter", "require a remediation plan"]

def distractors(a: str, b: str, first_number: int, seed: int, n_sections: int = len(TOPICS)) -> str:
    """``n_sections`` distractor sections (topics repeat with a suffix beyond the list)."""
    rng = random.Random(seed)
    sections = []
    for offset in range(n_sections):
        topic = TOPICS[offset % len(TOPICS)]
        if offset >= len(TOPICS):
            topic = f"{topic} {offset // len(TOPICS) + 1}"
        obj = topic.lower()
        paragraphs = []
        for template in rng.sample(TEMPLATES, k=rng.randint(6, 9)):
            n = rng.choice(NUMBERS)
            period = rng.choice(PERIODS) if "years" not in template else "years"
            text = template.format(a=a, b=b, verb=rng.choice(VERBS), obj=obj, n=n,
                                   period=period, right=rng.choice(RIGHTS), done=rng.choice(DONE),
                                   policy=rng.choice(POLICIES), condition=rng.choice(CONDITIONS),
                                   remedy=rng.choice(REMEDIES))
            paragraphs.append(text)
        sections.append(f"## {first_number + offset}. {topic}\n\n" + "\n\n".join(paragraphs))
    return "\n\n".join(sections) + "\n"
