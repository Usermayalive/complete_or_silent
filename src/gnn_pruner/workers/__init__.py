"""Chapter workers and the central board (WORKFLOW B2-B5).

    from gnn_pruner.workers import ask
    result = ask(tree, "Which law governs the Lease of BETA?", provider)

- ``routing``: which chapters a question names, and what to look for inside them
- ``board``: the central board (checks quotes by sentence id, merges, adds definitions,
  builds the evidence pack)
- ``worker``: one chapter worker (looks at its chapter, opens / reads / searches, posts
  sentence ids)
- ``orchestrator``: question -> workers in parallel -> board -> evidence pack -> answer
"""

from .orchestrator import WorkerResult, ask  # noqa: F401
