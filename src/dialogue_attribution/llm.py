"""A minimal client for OpenAI-compatible chat APIs, with a file cache.

Every answer is stored on disk under a hash of the request. Committing the
cache makes the whole pipeline reproducible without an API key: offline
runs read the stored answers, live runs fill the cache. API keys are only
read from the environment and are never written to the cache.
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


class CacheMiss(RuntimeError):
    """Raised in offline mode when an answer is not in the cache."""


class LLMError(RuntimeError):
    """Raised when the API keeps failing or answers with something unusable."""


@dataclass
class Usage:
    """What the run cost, summed over all requests (measured, not estimated)."""
    requests: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


@dataclass
class ChatClient:
    base_url: str
    api_key: str
    model: str
    timeout: float = 180.0
    max_attempts: int = 3

    @classmethod
    def from_env(cls, model: str | None = None) -> "ChatClient":
        """Read LLM_API_KEY, LLM_BASE_URL and (unless given) LLM_MODEL from the environment."""
        model = model or os.environ.get("LLM_MODEL")
        if not os.environ.get("LLM_API_KEY"):
            raise LLMError("Missing environment variable LLM_API_KEY")
        if not model:
            raise LLMError("No model: pass --model or set LLM_MODEL")
        return cls(base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
                   api_key=os.environ["LLM_API_KEY"], model=model)

    def chat(self, messages: list[dict]) -> tuple[str, dict]:
        """Send one chat request and return (answer text, token counts)."""
        body = {"model": self.model, "messages": messages, "temperature": 0,
                "response_format": {"type": "json_object"}}
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._parse(self._post(body))
            except urllib.error.HTTPError as err:
                reason = err.read()[:300].decode("utf-8", errors="replace")
                if err.code == 400 and "response_format" in body:
                    body.pop("response_format")  # some servers do not support JSON mode
                    continue
                if err.code not in (429, 500, 502, 503, 504) or attempt == self.max_attempts:
                    raise LLMError(f"HTTP {err.code} from {self.base_url}: {reason}") from err
            except OSError as err:  # no connection, timeout, connection dropped
                if attempt == self.max_attempts:
                    raise LLMError(f"Cannot reach {self.base_url}: {err}") from err
            time.sleep(2 ** attempt)  # wait 2 s, 4 s, ... before the next attempt
        raise LLMError("Request failed")

    def _post(self, body: dict) -> bytes:
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read()

    @staticmethod
    def _parse(raw: bytes) -> tuple[str, dict]:
        """Pull the answer text out of the response, or explain why it is unusable."""
        try:
            data = json.loads(raw)
            answer = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as err:
            raise LLMError(f"Unexpected response: {raw[:200]!r}") from err
        if not isinstance(answer, str) or not answer.strip():
            raise LLMError("The model returned an empty answer")
        usage = data.get("usage") or {}
        # Keep only token counts: providers may add cost or account details.
        return answer, {k: usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens")}


def model_slug(model: str) -> str:
    """Folder name for a model, e.g. 'openrouter/google/x' -> 'openrouter__google__x'."""
    return model.replace("/", "__").replace(":", "_")


@dataclass
class CachedLLM:
    """Answers from the cache first; asks the client only if one is given.

    `cache_dir` is the folder of one model (see `model_slug`), so the same
    prompt sent to two models is cached twice.
    """
    cache_dir: Path
    model: str
    client: ChatClient | None = None
    usage: Usage = field(default_factory=Usage)

    @staticmethod
    def key(messages: list[dict]) -> str:
        request = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(request.encode("utf-8")).hexdigest()[:16]

    def _path(self, messages: list[dict]) -> Path:
        return Path(self.cache_dir) / f"{self.key(messages)}.json"

    def ask(self, messages: list[dict]) -> str:
        path = self._path(messages)
        if path.exists():
            self.usage.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["answer"]
        if self.client is None:
            raise CacheMiss(f"No cached answer for this request ({path.name}). "
                            "Run once with --live and an API key to fill the cache.")
        started = time.perf_counter()
        answer, usage = self.client.chat(messages)
        self.usage.seconds += time.perf_counter() - started
        self.usage.requests += 1
        self.usage.prompt_tokens += usage.get("prompt_tokens", 0)
        self.usage.completion_tokens += usage.get("completion_tokens", 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "model": self.model,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "usage": usage,
            "answer": answer,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return answer

    def forget(self, messages: list[dict]) -> None:
        """Delete a cached answer that turned out to be unusable, so a live run asks again."""
        if self.client is not None:
            self._path(messages).unlink(missing_ok=True)
