import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from groq import APIConnectionError, APIStatusError

from extractor import ExtractionError, extract_meeting

TEXT = "Ayesha: I'll write tests."
CONTEXT = {"sentences": [{"text": "I'll write tests.", "speaker": "Ayesha", "start": 8, "end": len(TEXT)}], "entities": []}


def response(content, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))])


def client_with(value):
    client = Mock()
    client.chat.completions.create.return_value = value
    return client


def test_verified_and_discarded():
    data = {"action_items": [{"task": "Write tests", "owner": "Ayesha", "deadline": None, "evidence": "I'll write tests."}],
            "decisions": [{"decision": "Use PostgreSQL", "evidence": "We chose PostgreSQL."}]}
    client = client_with(response(json.dumps(data)))
    result, warnings = extract_meeting(TEXT, CONTEXT, "test-key", client=client)
    assert result.action_items[0].owner == "Ayesha"
    assert result.decisions == [] and len(warnings) == 1
    assert client.chat.completions.create.call_args.kwargs["response_format"]["json_schema"]["strict"] is True


@pytest.mark.parametrize("content", ["not json", "{}", '{"action_items": [], "decisions": [3]}'])
def test_invalid_output(content):
    client = client_with(response(content))
    with pytest.raises(ExtractionError, match="invalid JSON"):
        extract_meeting(TEXT, CONTEXT, "test-key", client=client)
    assert client.chat.completions.create.call_count == 1


def test_truncated():
    with pytest.raises(ExtractionError, match="incomplete"):
        extract_meeting(TEXT, CONTEXT, "test-key", client=client_with(response("{}", "length")))


@pytest.mark.parametrize("status,expected_calls", [(401, 1), (403, 1), (400, 1), (404, 1), (429, 2), (500, 2)])
def test_api_failures(status, expected_calls, monkeypatch):
    monkeypatch.setattr("extractor.time.sleep", lambda _: None)
    client = Mock()
    error = APIStatusError("failure", response=httpx.Response(status, request=httpx.Request("POST", "https://example.test")), body=None)
    client.chat.completions.create.side_effect = error
    with pytest.raises(ExtractionError):
        extract_meeting(TEXT, CONTEXT, "test-key", client=client)
    assert client.chat.completions.create.call_count == expected_calls


def test_transient_recovery(monkeypatch):
    monkeypatch.setattr("extractor.time.sleep", lambda _: None)
    client = Mock()
    client.chat.completions.create.side_effect = [APIConnectionError(request=httpx.Request("POST", "https://example.test")),
                                                response('{"action_items": [], "decisions": []}')]
    result, warnings = extract_meeting(TEXT, CONTEXT, "test-key", client=client)
    assert result.action_items == [] and warnings == []
    assert client.chat.completions.create.call_count == 2


def test_missing_key_and_empty_input():
    client = Mock()
    with pytest.raises(ExtractionError, match="GROQ_API_KEY"):
        extract_meeting(TEXT, CONTEXT, "", client=client)
    with pytest.raises(ValueError):
        extract_meeting(" ", CONTEXT, "test-key", client=client)
    client.chat.completions.create.assert_not_called()
