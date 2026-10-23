"""Phase 10: providers (with fake clients, no network), citation validation, abstention, eval helpers."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gnn_pruner import DocumentQA
from gnn_pruner.answer import (ABSTAIN_TEXT, ANSWER_SCHEMA, ClaudeProvider, Completion, OpenAIProvider,
                               answer_question, load_provider)
from gnn_pruner.answer_eval import answer_f1, number_match

ROOT = Path(__file__).resolve().parent.parent
SAAS = ROOT / "examples" / "saas_agreement.md"


@pytest.fixture(scope="module")
def pack():
    return DocumentQA.from_file(str(SAAS)).ask("Can the Provider increase fees on renewal?").evidence


def reply(**kw):
    data = {"answer": "Yes, by up to five percent [E1].", "evidence": ["E1"], "confidence": "high",
            "insufficient": False}
    data.update(kw)
    return json.dumps(data)


class FakeAnthropic:
    def __init__(self, text, stop="end_turn", model="claude-opus-5-5"):
        self.calls = []
        response = SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""),
                                            SimpleNamespace(type="text", text=text)],
                                   usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
                                   stop_reason=stop, model=model)

        def create(**kwargs):
            self.calls.append(kwargs)
            return response

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))
        self.messages = SimpleNamespace(create=create)


def test_claude_request_shape_and_cost(pack):
    client = FakeAnthropic(reply())
    provider = ClaudeProvider(client=client)
    ans = answer_question(pack, provider)
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
    assert call["output_config"]["effort"] == "medium"
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": ANSWER_SCHEMA}
    assert "only the evidence" in call["system"] and call["messages"][0]["content"].startswith("Question:")
    assert ans.answer.startswith("Yes") and ans.cited == ["E1"] and not ans.flags and not ans.abstained
    assert ans.cost_usd == pytest.approx(1000 * 4 / 1e6 + 100 * 20 / 1e6)
    assert ans.evidence[0]["node_id"] == pack.items[0].node_id and ans.evidence[0]["start"] == pack.items[0].start


def test_claude_without_fallback_uses_plain_create(pack):
    client = FakeAnthropic(reply())
    answer_question(pack, ClaudeProvider(client=client, fallback=False))
    assert "betas" not in client.calls[0] and "fallbacks" not in client.calls[0]


def test_claude_refusal_and_fallback_attribution(pack):
    refused = answer_question(pack, ClaudeProvider(client=FakeAnthropic("", stop="refusal")))
    assert refused.abstained and refused.flags == ["refused"] and refused.answer == ABSTAIN_TEXT
    rescued = answer_question(pack, ClaudeProvider(client=FakeAnthropic(reply(), model="claude-opus-4-8")))
    assert rescued.served_by == "claude-opus-4-8" and not rescued.abstained
    assert rescued.cost_usd == pytest.approx(1000 * 5 / 1e6 + 100 * 25 / 1e6)  # billed at the fallback's rates


def test_openai_request_shape(pack):
    calls = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply(), refusal=None),
                                                        finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=900, completion_tokens=80))

    def create(**kwargs):
        calls.append(kwargs)
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    ans = answer_question(pack, OpenAIProvider("some-model", client=client, prices=(1.0, 2.0)))
    call = calls[0]
    assert call["model"] == "some-model" and call["messages"][0]["role"] == "system"
    assert call["response_format"]["type"] == "json_schema" and call["response_format"]["json_schema"]["strict"]
    assert "max_completion_tokens" in call
    assert ans.cited == ["E1"] and ans.input_tokens == 900 and ans.cost_usd == pytest.approx((900 + 160) / 1e6)


class Scripted:
    name = "scripted"

    def __init__(self, text):
        self.text, self.calls = text, 0

    def complete(self, system, user, schema):
        self.calls += 1
        return Completion(self.text, "scripted")


def test_citation_validation(pack):
    bad = answer_question(pack, Scripted(reply(evidence=["E1", "E99"], answer="Yes [E1] and [E42].")))
    assert bad.cited == ["E1"] and set(bad.invalid_citations) == {"E99", "E42"} and "invalid_citation" in bad.flags
    uncited = answer_question(pack, Scripted(reply(evidence=[], answer="Yes.")))
    assert uncited.flags == ["uncited"]
    strict = answer_question(pack, Scripted(reply(evidence=["E99"], answer="Yes [E99].")), strict=True)
    assert strict.abstained and strict.answer == ABSTAIN_TEXT


def test_abstention(pack):
    empty = type(pack)(pack.question, pack.query_type, items=[])
    model = Scripted(reply())
    assert answer_question(empty, model).abstained and model.calls == 0
    low = type(pack)(pack.question, pack.query_type, items=list(pack.items), confidence="insufficient",
                     route=pack.route)
    assert answer_question(low, model).flags == ["low_retrieval_confidence"] and model.calls == 0
    assert not answer_question(low, model, abstain=False).abstained and model.calls == 1
    said_no = answer_question(pack, Scripted(reply(insufficient=True, answer=ABSTAIN_TEXT, evidence=[])))
    assert said_no.abstained


def test_free_text_reply_and_render(pack):
    text = "Answer: Yes, five percent [E1].\nEvidence: [E1: 3. Fees], [E2]\nConfidence: medium"
    ans = answer_question(pack, Scripted(text))
    assert ans.cited == ["E1", "E2"] and ans.confidence == "medium"
    rendered = ans.render()
    assert rendered.startswith("Answer: Yes") and "Confidence: medium" in rendered and "[E1] 3. Fees" in rendered


def test_pipeline_providers():
    qa = DocumentQA.from_file(str(SAAS), llm=Scripted(reply()))
    result = qa.ask("Can the Provider increase fees on renewal?")
    assert result.grounded.cited == ["E1"] and result.answer.startswith("Answer: Yes")
    offline = DocumentQA.from_file(str(SAAS)).ask("Can the Provider increase fees on renewal?")
    assert offline.grounded.provider == "extractive" and offline.grounded.cited == ["E1"]


def test_load_provider_specs():
    assert load_provider(None) is None and load_provider("extractive") is None
    with pytest.raises(ValueError):
        load_provider("openai:")  # an OpenAI model must be named
    with pytest.raises(ValueError):
        load_provider("gemini")


def test_eval_helpers():
    ref = "Undisputed invoices within thirty days of receipt (Section 2)."
    assert answer_f1("Invoices must be paid within 30 days of receipt [E1].", ref) > 0.6
    assert answer_f1("No idea.", ref) == 0.0
    assert number_match("within 30 days [E1]", ref) == 1.0  # the section number is not a fact to match
    assert number_match("within 60 days", ref) == 0.0
    assert number_match("Yes", "Yes (Section 2).") is None


def test_quoted_citations_map_to_labels(pack):
    quote = " ".join(pack.items[0].text.split()[:8])
    ans = answer_question(pack, Scripted(reply(evidence=[quote, "a sentence that is nowhere in the pack"],
                                               answer="Yes.")))
    assert ans.cited == ["E1"] and ans.invalid_citations == ["a sentence that is nowhere in the pack"]


def test_ollama_provider_points_at_local_server(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    provider = load_provider("ollama:some-local-model")
    assert provider.name == "ollama:some-local-model" and provider.prices == (0.0, 0.0)
    assert str(provider.client.base_url).startswith("http://127.0.0.1:11434/v1")


def test_gemini_provider_paces_and_retries_rate_limits(pack, monkeypatch):
    from gnn_pruner import answer as answer_mod
    from gnn_pruner.answer import GeminiProvider

    monkeypatch.setattr(answer_mod.time, "sleep", lambda s: None)
    calls = []
    ok = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply(), refusal=None),
                                                  finish_reason="stop")],
                         usage=SimpleNamespace(prompt_tokens=900, completion_tokens=80))

    class RateLimited(Exception):
        status_code = 429

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RateLimited()
        return ok

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = GeminiProvider("gemini-test", client=client)
    ans = answer_question(pack, provider)
    assert len(calls) == 2 and ans.cited == ["E1"] and ans.cost_usd == 0.0
    assert provider.name == "gemini:gemini-test" and calls[1]["model"] == "gemini-test"


def test_gemini_daily_quota_stops_instead_of_retrying(pack, monkeypatch):
    from gnn_pruner import answer as answer_mod
    from gnn_pruner.answer import GeminiProvider, QuotaExhausted

    monkeypatch.setattr(answer_mod.time, "sleep", lambda s: None)
    calls = []

    class DailyQuota(Exception):
        status_code = 429

        def __str__(self):
            return "Quota exceeded for metric: free_tier_requests, limit: 20. Please retry in 23h18m2s."

    def create(**kwargs):
        calls.append(kwargs)
        raise DailyQuota()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(QuotaExhausted):
        answer_question(pack, GeminiProvider("gemini-test", client=client))
    assert len(calls) == 1  # no pointless retries
