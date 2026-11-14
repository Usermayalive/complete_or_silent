# Data card: the trap benchmark (`traps`, `traps2`)

Questions whose correct answer depends on a provision planted **far from** the clause that
seems to answer them, in data rooms of real contracts from 100 to 10,000 pages. Built for
the Complete or Silent project: a system must find the exception, the override or the
amendment, and say what it read (closure certificate), or certify that a clause is absent.

## Versions

| Version | Text of the planted clauses | Rooms | Questions | Status |
|---|---|---|---|---|
| `traps` (v1) | hand-written templates (2–3 wordings per role) | S 139 pp, M 1,034 pp | 60, 360 | frozen: the first reported results use it; rebuilt byte-identically by `--name traps` |
| `traps2` | real CUAD clauses rewritten into templates by a model, validated (T6) | S 144 pp, M 1,065 pp, L 8,782 pp | 62, 361, 661 | current |

Rebuild (data is not in git; CUAD is downloaded by the long-benchmark builder):

```bash
PYTHONPATH=src python -m gnn_pruner.longbench build
PYTHONPATH=src python -m gnn_pruner.certify.traps build --name traps --counts S=60,M=300,L=0
PYTHONPATH=src python -m gnn_pruner.certify.traps build --name traps2 \
  --library eval/traps2/clause_templates.json --counts S=300,M=300,L=300
```

Outputs: `data/longbench/{name}_{S,M,L}.md` (rooms) and `eval/long/{name}_{S,M,L}.jsonl`
(questions; in git). Seed 7 throughout; the build is deterministic.

## Source data and licences

| Source | Licence | Use |
|---|---|---|
| CUAD v1 (The Atticus Project), 510 commercial contracts with expert clause labels | CC BY 4.0 | the contracts that form the rooms; the clauses rewritten into templates; the labels used for "absent" and "room-wide" gold |
| Model rewrites (gemini-3.1-flash-lite, 2026-11-14) | generated text, released with the benchmark | `eval/traps2/clause_templates.json`: each template records its source contract and CUAD category |

The rooms are nested: the 1,000-page room contains the 100-page room's contracts, and so on,
as in the long benchmark (`docs/READINGS.md`, step 2).

## What is planted

Every trap plants a numbered **base clause** (a named company's liability cap, "Section N")
in one contract, and a **look-alike** cap for another company in another contract. Then:

| Kind | Planted | Correct answer | Expected certificate |
|---|---|---|---|
| far exception | "Notwithstanding Section N …" carve-out with a second amount: later in the same contract, in a schedule of it, or in a side letter (a separate document naming the contract) | base amount, except the carve-out amount | closure |
| override chain | Amendment No. 1 replaces Section N, Amendment No. 2 replaces it again (two separate documents) | the second amendment's amount; the base and Amendment 1 are superseded | closure |
| unlinked carve-out | a fraud / wilful-misconduct exclusion with **no section number** | base amount, but no limit for fraud | closure |
| precedence | a schedule states another cap for the same company; the contract says schedules prevail | the schedule's amount; the base is superseded | closure |
| existence, yes | a change-of-control clause | yes | closure |
| existence, no | nothing: a (contract, clause type) pair where the contract truly lacks the clause (no CUAD label, no planted clause, none of the type's key words) | no | absence |
| room-wide (`traps2`) | escrow clauses planted in 2–3 contracts | every contract of the room holding one (planted, CUAD-labelled, or containing the word) | closure |

Gold per question (in `tags`): `required` (sentences a complete answer must rest on),
`superseded`, `certificate`, `distance_pages` (base to the farthest required sentence,
measured inside the case's own documents), `depth`, the contract, and for side letters and
amendments the second document.

## Size and shape (`traps2`)

| Room | Pages | Documents | Questions | Per trap kind | Distance base → farthest provision |
|---|---|---|---|---|---|
| S | 144 | 33 + planted amendments / side letters | 62 | 12 each, + 2 room-wide | see `distance_pages` |
| M | 1,065 | ~330 | 361 | 72 each, + 1 room-wide | median ~13 pp, up to ~1,000 pp |
| L | 8,782 | ~930 | 661 | 132 each, + 1 room-wide | up to several thousand pp |

**Density:** at most 12 traps per contract. The S room has only ~7 contracts, so it holds
62 questions, not the 300 planned; reaching 300 at the 100-page size needs several
independent 100-page rooms (future work).

## How the clause templates were made (T6)

1. For each role, real CUAD clauses of a related category (Cap On Liability, Uncapped
   Liability, Change Of Control, Source Code Escrow, Third Party Beneficiary) were taken
   from contracts **outside** the S and M rooms.
2. A model rewrote each into the role's required content, keeping the clause's style, with
   placeholders (`{co}`, `{amount}`, `{amount2}`, `{amount3}`, `{n}`, `{title}`).
3. Every rewrite was **validated** (no stray amounts or numbers, required placeholders and
   section references present, the unlinked carve-out cites no section and names fraud, a
   precedence clause must make the schedules prevail by the same rule the system uses);
   failures were dropped, never repaired. Yield: 10–20 per role.
4. Schedule caps and both amendments are also derived from the 20 base templates (the cap
   moved, the amendment frame added), so every role has 10–24 real-style variants.

## Known limitations

- **Templates, even real-style ones, are still templates:** a system tuned to them could
  learn the wording. The cue patterns of the system were **not** extended to the
  templates' wording (5 of the 72 unlinked carve-outs at M use wording the patterns miss;
  this is reported as a limitation, not fixed by adding those phrases).
- **The traps were designed together with the system.** External baselines (long-context
  model, Chain of Agents, strong retrieval) are needed for a fair comparison.
- **Room-wide questions are few** (1–2 per room) because, in the larger rooms, every
  candidate "absent" clause type occurs somewhere.
- **Liability caps only** for the far-exception, chain, carve-out and precedence kinds;
  other clause families (termination, IP, exclusivity) are future work.
- **One seed.** Variance over seeds is not yet measured.

## Responsible use

The rooms consist of public contracts (CUAD) with fictitious planted provisions and
fictitious company names. Nothing here is legal advice, and planted provisions do not
describe the real parties to those contracts.
