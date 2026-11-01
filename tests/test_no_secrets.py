"""Guard against committing an API key by accident."""

import re

from conftest import ROOT

KEY_PATTERNS = re.compile(
    r"sk-or-v1-[0-9a-f]{20,}"        # OpenRouter
    r"|sk_[0-9a-f]{40,}"             # other providers
    r"|AIza[0-9A-Za-z_-]{30,}"       # Google
    r"|sk-(?:proj-)?[A-Za-z0-9_-]{30,}"  # OpenAI / LiteLLM style
)
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}


def test_no_api_keys_in_repository():
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if KEY_PATTERNS.search(text):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_no_env_file():
    assert not (ROOT / ".env").exists()
