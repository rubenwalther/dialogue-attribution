import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from conftest import FakeClient

from dialogue_attribution import llm as llm_module
from dialogue_attribution.llm import CachedLLM, CacheMiss, ChatClient, LLMError, model_slug

MESSAGES = [{"role": "user", "content": "hi"}]


def test_cache_stores_answer_and_usage_but_never_the_key(tmp_path):
    client = FakeClient('{"labels": []}')
    client.api_key = "sk-secret-should-not-appear"
    llm = CachedLLM(tmp_path, "m", client)
    assert llm.ask(MESSAGES) == '{"labels": []}'
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    stored = files[0].read_text(encoding="utf-8")
    assert "sk-secret" not in stored
    assert json.loads(stored)["usage"]["prompt_tokens"] == 100
    assert llm.usage.requests == 1 and llm.usage.prompt_tokens == 100


def test_second_ask_comes_from_cache(tmp_path):
    llm = CachedLLM(tmp_path, "m", FakeClient("answer"))
    llm.ask(MESSAGES)
    assert llm.ask(MESSAGES) == "answer"  # FakeClient has no second answer: this is the cache
    assert llm.usage.cache_hits == 1


def test_offline_cache_miss_explains_what_to_do(tmp_path):
    with pytest.raises(CacheMiss, match="--live"):
        CachedLLM(tmp_path, "m").ask(MESSAGES)


def test_model_slug():
    assert model_slug("openrouter/google/gemini:free") == "openrouter__google__gemini_free"


def test_from_env_needs_key_and_model(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        ChatClient.from_env()


class _Server:
    """A tiny local stand-in for an OpenAI-compatible endpoint."""

    def __init__(self, statuses):
        self.statuses = list(statuses)  # HTTP status per request; 200 = answer
        self.bodies = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.bodies.append((self.headers["Authorization"], body))
                status = server.statuses.pop(0)
                payload = {"choices": [{"message": {"content": "ok"}}],
                           "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
                data = json.dumps(payload).encode() if status == 200 else b"{}"
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def close(self):
        self.httpd.shutdown()


@pytest.fixture(autouse=True)
def no_sleep_no_proxy(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")  # the local test server must not go through a proxy


def test_client_sends_bearer_key_and_json_mode():
    server = _Server([200])
    try:
        answer, usage = ChatClient(server.url, "test-key", "m").chat(MESSAGES)
    finally:
        server.close()
    auth, body = server.bodies[0]
    assert answer == "ok" and usage["prompt_tokens"] == 5
    assert auth == "Bearer test-key"
    assert body["temperature"] == 0 and body["response_format"] == {"type": "json_object"}


def test_client_drops_json_mode_after_400_and_retries_429():
    server = _Server([400, 429, 200])
    try:
        answer, _ = ChatClient(server.url, "k", "m").chat(MESSAGES)
    finally:
        server.close()
    assert answer == "ok"
    assert "response_format" in server.bodies[0][1]
    assert "response_format" not in server.bodies[1][1]


def test_client_gives_up_on_401():
    server = _Server([401])
    try:
        with pytest.raises(LLMError, match="HTTP 401 .*: {}"):  # server text is included
            ChatClient(server.url, "k", "m").chat(MESSAGES)
    finally:
        server.close()


def test_from_env_takes_the_model_from_the_argument(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert ChatClient.from_env("my/model").model == "my/model"


@pytest.mark.parametrize("raw, reason", [
    (b"<html>gateway error</html>", "Unexpected response"),
    (b'{"error": {"message": "quota"}}', "Unexpected response"),
    (b'{"choices": [{"message": {"content": null}}]}', "empty answer"),
])
def test_unusable_responses_raise_a_clear_error(raw, reason):
    with pytest.raises(LLMError, match=reason):
        ChatClient._parse(raw)


def test_only_token_counts_are_kept_from_usage():
    raw = b'{"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": 3, "cost": 0.1}}'
    assert ChatClient._parse(raw) == ("x", {"prompt_tokens": 3, "completion_tokens": 0})


def test_forget_removes_a_cached_answer_only_in_live_mode(tmp_path):
    live = CachedLLM(tmp_path, "m", FakeClient("bad"))
    live.ask(MESSAGES)
    CachedLLM(tmp_path, "m").forget(MESSAGES)  # offline: keep the committed cache
    assert len(list(tmp_path.glob("*.json"))) == 1
    live.forget(MESSAGES)
    assert list(tmp_path.glob("*.json")) == []
