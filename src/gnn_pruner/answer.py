"""Phase 10: grounded answers from an evidence pack (spec §5 phase 8).

Providers (one interface, ``complete(system, user, schema) -> Completion``):
- ``ClaudeProvider`` (``claude``, ``claude:<model>``): Anthropic Python SDK.
  - default model ``claude-opus-5-5``, effort set explicitly (default ``medium``)
  - the answer schema enforced with ``output_config.format``
  - server-side refusal fallback on (``fallbacks: "default"``), so a request a safety
    classifier declines is re-run on a suitable model instead of failing; ``fallback=False``
    turns it off. A refusal that survives is reported, never parsed as an answer.
- ``OpenAIProvider`` (``openai:<model>``): OpenAI Chat Completions with a JSON-schema
  ``response_format``. The model is your choice; there is no default.
- ``OllamaProvider`` (``ollama:<model>``): a local model through Ollama's
  OpenAI-compatible endpoint (``OLLAMA_HOST``, default ``http://localhost:11434``). Free,
  and the contract text never leaves the machine.
- ``GeminiProvider`` (``gemini:<model>``): Google Gemini's OpenAI-compatible endpoint
  (``GEMINI_API_KEY``), paced for the free tier's rate limits.
- ``ExtractiveProvider`` (``extractive``): no LLM. The top evidence item, cited. The
  offline fallback, used when no provider is configured.
- a plain callable ``str -> str`` (the older ``llm=`` argument) still works; its free-text
  reply is parsed for Answer / Evidence / Confidence.

**Citation validation:** every cited label must exist in the pack. Invalid labels and
answers with no valid citation are flagged; ``strict=True`` turns a flagged answer into an
abstention.

**Abstention:** no model call when the pack is empty or Phase 7's confidence is
``insufficient`` (when abstention is on); a model reply marked ``insufficient`` also
abstains.

Credentials come from the environment (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` /
``GEMINI_API_KEY``, or an
``ant auth login`` profile for Claude); nothing is stored in the project.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

from .evidence import EvidencePack
from .prompt import count_tokens, render_parts

ABSTAIN_TEXT = "The document does not say."

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "insufficient": {"type": "boolean"},
    },
    "required": ["answer", "evidence", "confidence", "insufficient"],
    "additionalProperties": False,
}

# USD per 1M tokens (input, output). Anthropic first-party rates (2026-09); OpenAI prices
# are not assumed - pass ``prices=`` to OpenAIProvider to get costs.
CLAUDE_PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
                 "claude-haiku-4-5": (1.0, 5.0), "claude-fable-5-1": (10.0, 50.0),
                 # models the refusal fallback can route to
                 "claude-opus-5": (5.0, 25.0), "claude-opus-4-8": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0)}


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: Optional[str] = None
    refused: bool = False
    cost_usd: Optional[float] = None
    served_by: Optional[str] = None  # set when a fallback model answered


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str, schema: Dict) -> Completion: ...


class ClaudeProvider:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 16000,
                 fallback: bool = True, client=None):
        import anthropic

        self.model, self.effort, self.max_tokens, self.fallback = model, effort, max_tokens, fallback
        self.client = client or anthropic.Anthropic()
        self.name = f"claude:{model}"

    def complete(self, system: str, user: str, schema: Dict) -> Completion:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}],
                      output_config={"effort": self.effort,
                                     "format": {"type": "json_schema", "schema": schema}})
        if self.fallback:
            response = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                        fallbacks="default", **kwargs)
        else:
            response = self.client.messages.create(**kwargs)
        usage = response.usage
        served = response.model if response.model and response.model != self.model else None
        price = CLAUDE_PRICES.get(served or self.model)
        cost = (usage.input_tokens * price[0] + usage.output_tokens * price[1]) / 1e6 if price else None
        if response.stop_reason == "refusal":
            return Completion("", self.model, usage.input_tokens, usage.output_tokens, "refusal", True, cost, served)
        text = next((b.text for b in response.content if b.type == "text"), "")
        return Completion(text, self.model, usage.input_tokens, usage.output_tokens, response.stop_reason,
                          False, cost, served)


class OpenAIProvider:
    def __init__(self, model: str, max_tokens: int = 4000, prices: Optional[tuple] = None, client=None,
                 base_url: Optional[str] = None, api_key: Optional[str] = None):
        from openai import OpenAI

        self.model, self.max_tokens, self.prices = model, max_tokens, prices
        self.client = client or (OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI())
        self.name = f"openai:{model}"

    def complete(self, system: str, user: str, schema: Dict) -> Completion:
        response = self.client.chat.completions.create(
            model=self.model, max_completion_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "grounded_answer", "schema": schema, "strict": True}})
        choice = response.choices[0]
        usage = response.usage
        cost = None
        if self.prices and usage:
            cost = (usage.prompt_tokens * self.prices[0] + usage.completion_tokens * self.prices[1]) / 1e6
        refusal = getattr(choice.message, "refusal", None)
        return Completion(choice.message.content or "", self.model,
                          usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0,
                          choice.finish_reason, bool(refusal), cost)


class OllamaProvider(OpenAIProvider):
    """A local model served by Ollama (OpenAI-compatible API, JSON-schema output). Cost 0."""

    def __init__(self, model: str, max_tokens: int = 1000, host: Optional[str] = None, client=None):
        import os

        host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not host.startswith("http"):
            host = "http://" + host
        super().__init__(model, max_tokens, prices=(0.0, 0.0), client=client,
                         base_url=f"{host}/v1", api_key="ollama")
        self.name = f"ollama:{model}"


class QuotaExhausted(RuntimeError):
    """The provider's daily quota is used up; retrying today will not help."""


def _daily_quota(exc: Exception) -> bool:
    """A 429 whose suggested wait is an hour or more (a per-day quota, not per-minute)."""
    match = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?", str(exc))
    return bool(match and match.group(1))


class GeminiProvider(OpenAIProvider):
    """Google Gemini through its OpenAI-compatible endpoint (``GEMINI_API_KEY``).

    Built for the free tier: calls are spaced ``min_interval`` seconds apart and a
    rate-limit reply (429) is retried with backoff. Cost is reported as 0 (free tier).
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Gemini 3.x counts its reasoning tokens against the output cap: leave room for both.
    def __init__(self, model: str, max_tokens: int = 16000, min_interval: float = 6.5, retries: int = 6,
                 client=None):
        import os

        super().__init__(model, max_tokens, prices=(0.0, 0.0), client=client, base_url=self.BASE_URL,
                         api_key=None if client else os.environ["GEMINI_API_KEY"])
        self.name = f"gemini:{model}"
        self.min_interval, self.retries, self._last = min_interval, retries, 0.0
        import threading

        self._pace = threading.Lock()  # parallel workers share the pacing

    def complete(self, system: str, user: str, schema: Dict) -> Completion:
        for attempt in range(self.retries + 1):
            with self._pace:
                wait = self.min_interval - (time.monotonic() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            try:
                return super().complete(system, user, schema)
            except Exception as exc:  # openai.RateLimitError / transient 5xx
                status = getattr(exc, "status_code", None)
                if status == 429 and _daily_quota(exc):
                    raise QuotaExhausted(f"{self.name}: free-tier daily quota used up") from exc
                if status not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise
                time.sleep(min(60.0, 5.0 * 2 ** attempt))
        raise RuntimeError("unreachable")


class ExtractiveProvider:
    """No LLM: the top evidence item verbatim. Used offline and as the baseline."""

    name = "extractive"

    def __init__(self, pack: Optional[EvidencePack] = None):
        self.pack = pack

    def complete(self, system: str, user: str, schema: Dict) -> Completion:
        pack = self.pack
        if pack is None or not pack.items:
            return Completion(json.dumps({"answer": ABSTAIN_TEXT, "evidence": [], "confidence": "low",
                                          "insufficient": True}), "extractive")
        top = pack.items[0]
        confidence = {"high": "high", "low": "low", "insufficient": "low"}.get(pack.confidence or "", "medium")
        return Completion(json.dumps({"answer": f"{top.text} [{top.label}]", "evidence": [top.label],
                                      "confidence": confidence, "insufficient": False}), "extractive")


class CallableProvider:
    """Wraps the older ``llm(prompt) -> str`` callable (free-text reply)."""

    name = "callable"

    def __init__(self, fn: Callable[[str], str], pack: EvidencePack):
        self.fn, self.pack = fn, pack

    def complete(self, system: str, user: str, schema: Dict) -> Completion:
        return Completion(self.fn(self.pack.render_prompt()), "callable")


def load_provider(spec: Optional[str]) -> Optional[Provider]:
    """``None`` / ``"extractive"`` -> extractive; ``"claude"`` / ``"claude:<model>"``;
    ``"openai:<model>"`` (an OpenAI model must be named)."""
    if not spec or spec == "extractive":
        return None
    if spec == "claude":
        return ClaudeProvider()
    if spec.startswith("claude:"):
        return ClaudeProvider(model=spec.split(":", 1)[1])
    if spec.startswith("openai:") and spec.split(":", 1)[1]:
        return OpenAIProvider(model=spec.split(":", 1)[1])
    if spec.startswith("ollama:") and spec.split(":", 1)[1]:
        return OllamaProvider(model=spec.split(":", 1)[1])
    if spec.startswith("gemini:") and spec.split(":", 1)[1]:
        return GeminiProvider(model=spec.split(":", 1)[1])
    raise ValueError(f"unknown provider {spec!r}: use extractive, claude[:model], openai:<model>, "
                     "ollama:<model> or gemini:<model>")


# ---------------------------------------------------------------- answers

@dataclass
class Answer:
    question: str
    answer: str
    evidence: List[Dict] = field(default_factory=list)   # cited items: label, section, node_id, offsets, text
    confidence: str = "low"
    abstained: bool = False
    cited: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)       # invalid_citation, uncited, refused, unparsed
    provider: str = "extractive"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None
    latency_ms: float = 0.0
    route: str = ""
    served_by: Optional[str] = None

    @property
    def references(self) -> List[str]:
        return [f"[{e['label']}] {e['section']}" for e in self.evidence]

    def render(self) -> str:
        lines = [f"Answer: {self.answer}", "", f"Confidence: {self.confidence}"
                 + (" (abstained)" if self.abstained else "")]
        if self.evidence:
            lines += ["", "Evidence:"] + [f"  [{e['label']}] {e['section']}: {e['text'][:160]}" for e in self.evidence]
        if self.flags:
            lines += ["", "Flags: " + ", ".join(self.flags)]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return asdict(self)


_LABEL = re.compile(r"\bE(\d+)\b")


def _parse(text: str) -> Optional[Dict]:
    """JSON reply, or the free-text Answer / Evidence / Confidence format."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "answer" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"Answer:\s*(.+?)(?:\n\s*Evidence:|\Z)", text, re.S)
    if not m:
        return None
    evidence = re.search(r"Evidence:\s*(.+?)(?:\n\s*Confidence:|\Z)", text, re.S)
    confidence = re.search(r"Confidence:\s*(high|medium|low)", text, re.I)
    answer = m.group(1).strip()
    return {"answer": answer, "evidence": [f"E{n}" for n in _LABEL.findall(evidence.group(1))] if evidence else [],
            "confidence": confidence.group(1).lower() if confidence else "low",
            "insufficient": answer.lower().startswith("the document does not say")}


def _as_label(entry: str, pack: EvidencePack) -> str:
    """A cited entry as a pack label: "E2", "[E2: 4.2 Exclusions]", or a quote of an item."""
    m = _LABEL.search(entry)
    if m:
        return f"E{m.group(1)}"
    quote = " ".join(entry.strip(" \"'[]").lower().split())
    if len(quote.split()) >= 4:
        for item in pack.items:
            if quote in " ".join(item.text.lower().split()):
                return item.label
    return entry.strip()


def answer_question(pack: EvidencePack, provider: Optional[Provider] = None, abstain: bool = True,
                    strict: bool = False, notes: str = "") -> Answer:
    """Ground an answer in ``pack`` with ``provider`` (extractive when None).

    ``notes``: orientation for the model (e.g. chapter workers' partial answers for a
    whole-document question). They are not evidence: only E labels can be cited."""
    provider = provider or ExtractiveProvider(pack)
    out = Answer(pack.question, ABSTAIN_TEXT, provider=getattr(provider, "name", "custom"), route=pack.route)
    if not pack.items or (abstain and pack.confidence == "insufficient"):
        out.abstained, out.flags = True, ["no_evidence" if not pack.items else "low_retrieval_confidence"]
        return out
    system, user = render_parts(pack, structured=True)
    if notes:
        user += ("\n\nNotes from the search (orientation only, not evidence; cite only the E labels "
                 "above):\n" + notes)
    started = time.perf_counter()
    completion = provider.complete(system, user, ANSWER_SCHEMA)
    out.latency_ms = (time.perf_counter() - started) * 1000
    out.model, out.served_by = completion.model, completion.served_by
    out.input_tokens, out.output_tokens, out.cost_usd = (completion.input_tokens, completion.output_tokens,
                                                         completion.cost_usd)
    if completion.refused:
        out.abstained, out.flags = True, ["refused"]
        return out
    data = _parse(completion.text)
    if data is None:
        out.abstained, out.flags = True, ["unparsed"]
        return out
    out.answer = str(data.get("answer", "")).strip() or ABSTAIN_TEXT
    out.confidence = data.get("confidence", "low") if data.get("confidence") in ("high", "medium", "low") else "low"
    if data.get("insufficient"):
        out.abstained, out.answer = True, out.answer or ABSTAIN_TEXT
        return out
    # Citation validation: labels from the evidence list and inline [E#] markers. A
    # quoted passage (some models cite by quoting) counts when the quote is really in an
    # item, which is a stricter check than a label.
    by_label = {i.label: i for i in pack.items}
    cited = list(dict.fromkeys([_as_label(str(x), pack) for x in data.get("evidence", [])]
                               + [f"E{n}" for n in _LABEL.findall(out.answer)]))
    out.cited = [c for c in cited if c in by_label]
    out.invalid_citations = [c for c in cited if c not in by_label]
    if out.invalid_citations:
        out.flags.append("invalid_citation")
    if not out.cited:
        out.flags.append("uncited")
    if strict and out.flags:
        out.abstained, out.answer = True, ABSTAIN_TEXT
        return out
    out.evidence = [{"label": c, "section": by_label[c].section_path, "node_id": by_label[c].node_id,
                     "start": by_label[c].start, "end": by_label[c].end, "page": by_label[c].page,
                     "text": by_label[c].text} for c in out.cited]
    return out


def estimate_tokens(pack: EvidencePack) -> int:
    system, user = render_parts(pack, structured=True)
    return count_tokens(system) + count_tokens(user)
