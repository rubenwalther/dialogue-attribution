import json

import pytest
from conftest import DATA, labels_json

from dialogue_attribution.attribute import build_messages
from dialogue_attribution.cli import main
from dialogue_attribution.llm import CachedLLM, model_slug
from dialogue_attribution.schema import CharacterList
from dialogue_attribution.segment import segment

CHARS = str(DATA / "alice_ch07.characters.json")


@pytest.fixture
def tiny(tmp_path):
    """A two-quote text plus a cache that already holds the model's answer."""
    text = tmp_path / "tiny.txt"
    text.write_text("“Have some wine,” the March Hare said.\n\n“There isn’t any,” said Alice.",
                    encoding="utf-8")
    characters = CharacterList.model_validate_json(
        (DATA / "alice_ch07.characters.json").read_text(encoding="utf-8"))
    messages = build_messages(segment(text.read_text(encoding="utf-8")), characters)
    folder = tmp_path / "cache" / model_slug("demo/model")
    folder.mkdir(parents=True)
    (folder / f"{CachedLLM.key(messages)}.json").write_text(json.dumps({
        "model": "demo/model",
        "answer": labels_json((0, "March Hare", "happy"), (2, "Alice", "neutral")),
    }), encoding="utf-8")
    return tmp_path, text


def test_offline_run_reads_the_cache_and_finds_the_model(tiny, monkeypatch, capsys):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    root, text = tiny
    out = root / "out.jsonl"
    main(["run", str(text), "--characters", CHARS, "--out", str(out), "--cache", str(root / "cache")])
    rows = [json.loads(r) for r in out.read_text(encoding="utf-8").splitlines()]
    assert [r["speaker"] for r in rows] == ["March Hare", "Narrator", "Alice", "Narrator"]
    assert "0 live requests, 1 from cache" in capsys.readouterr().out
    assert not out.with_suffix(".meta.json").exists()  # nothing measured offline


def test_offline_run_without_cache_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "demo/model")
    text = tmp_path / "t.txt"
    text.write_text("“Hi,” said Alice.", encoding="utf-8")
    with pytest.raises(SystemExit, match="--live"):
        main(["run", str(text), "--characters", CHARS, "--out", str(tmp_path / "o.jsonl"),
              "--cache", str(tmp_path / "cache")])


def test_baseline_and_eval_on_the_real_chapter(tmp_path, capsys):
    out = tmp_path / "baseline.jsonl"
    report = tmp_path / "report.md"
    main(["baseline", str(DATA / "alice_ch07.txt"), "--characters", CHARS, "--out", str(out)])
    main(["eval", str(out), str(DATA / "alice_ch07.gold.jsonl"), "--name", "Baseline",
          "--report", str(report)])
    assert "| 147 |" in capsys.readouterr().out
    assert report.read_text(encoding="utf-8").startswith("### Baseline")


def test_live_run_without_key_fails_with_a_message_not_a_traceback(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    text = tmp_path / "t.txt"
    text.write_text("“Hi,” said Alice.", encoding="utf-8")
    with pytest.raises(SystemExit, match="LLM_API_KEY"):
        main(["run", str(text), "--characters", CHARS, "--out", str(tmp_path / "o.jsonl"),
              "--cache", str(tmp_path / "cache"), "--model", "demo/model", "--live"])
