import json
from pathlib import Path

import pytest

from dialogue_attribution.schema import CharacterList

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@pytest.fixture
def alice_text() -> str:
    return (DATA / "alice_ch07.txt").read_text(encoding="utf-8")


@pytest.fixture
def characters() -> CharacterList:
    return CharacterList.model_validate_json((DATA / "alice_ch07.characters.json").read_text(encoding="utf-8"))


class FakeClient:
    """Stands in for the API: returns prepared answers and records the requests."""

    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.requests: list[list[dict]] = []

    def chat(self, messages):
        self.requests.append(messages)
        return self.answers.pop(0), {"prompt_tokens": 100, "completion_tokens": 10}


def labels_json(*labels) -> str:
    """labels_json((1, "Alice", "happy"), ...) -> the JSON the model would send."""
    return json.dumps({"labels": [{"id": i, "speaker": s, "emotion": e} for i, s, e in labels]})
