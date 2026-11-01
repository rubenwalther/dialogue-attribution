import pytest
from conftest import FakeClient, labels_json

from dialogue_attribution.attribute import attribute, build_messages, chunks, parse_answer
from dialogue_attribution.llm import CachedLLM
from dialogue_attribution.segment import segment

TEXT = "“Have some wine,” the March Hare said.\n\nAlice looked round. “I don’t see any wine,” she remarked."


def test_messages_mark_every_quote_and_keep_narration(characters):
    user = build_messages(segment(TEXT), characters)[1]["content"]
    assert "[0]“Have some wine,” the March Hare said." in user
    assert "Alice looked round. [3]“I don’t see any wine,” she remarked." in user
    assert "- Dormouse: sleepy" in user


def test_labels_quotes_and_gives_narration_to_the_narrator(characters, tmp_path):
    client = FakeClient(labels_json((0, "March Hare", "happy"), (3, "Alice", "neutral")))
    llm = CachedLLM(tmp_path, "fake", client)
    lines = attribute(segment(TEXT), characters, llm)
    assert [(line.kind, line.speaker, line.emotion) for line in lines] == [
        ("quote", "March Hare", "happy"),
        ("narration", "Narrator", "neutral"),
        ("narration", "Narrator", "neutral"),
        ("quote", "Alice", "neutral"),
        ("narration", "Narrator", "neutral"),
    ]


def test_invalid_answer_gets_one_retry_with_the_reason(characters, tmp_path):
    client = FakeClient(labels_json((0, "March Hare", "happy")),  # marker 3 missing
                        labels_json((0, "March Hare", "happy"), (3, "Alice", "neutral")))
    lines = attribute(segment(TEXT), characters, CachedLLM(tmp_path, "fake", client))
    assert lines[3].speaker == "Alice"
    assert "Missing markers: [3]" in client.requests[1][-1]["content"]


def test_second_invalid_answer_raises(characters, tmp_path):
    client = FakeClient("not json", "still not json")
    with pytest.raises(ValueError):
        attribute(segment(TEXT), characters, CachedLLM(tmp_path, "fake", client))


@pytest.mark.parametrize("answer, reason", [
    (labels_json((0, "Queen", "angry"), (3, "Alice", "neutral")), "not in the character list"),
    (labels_json((0, "Alice", "bored"), (3, "Alice", "neutral")), "not valid JSON"),
    (labels_json((0, "Alice", "happy"), (3, "Alice", "happy"), (9, "Alice", "happy")), "Unknown or repeated"),
    (labels_json((0, "Alice", "happy"), (0, "Alice", "happy"), (3, "Alice", "happy")), "Unknown or repeated"),
])
def test_parse_answer_rejects(answer, reason):
    with pytest.raises(ValueError, match=reason):
        parse_answer(answer, {0, 3}, {"Alice", "March Hare"})


def test_parse_answer_accepts_unknown_and_code_fences():
    answer = "```json\n" + labels_json((0, "Unknown", "neutral")) + "\n```"
    assert parse_answer(answer, {0}, {"Alice"}).labels[0].speaker == "Unknown"


def test_chunks_keep_paragraphs_together():
    text = "\n\n".join(f"Paragraph {i}. “Quote {i}.”" for i in range(5))
    groups = chunks(segment(text), max_chars=60)
    assert len(groups) > 1
    for group in groups:
        paragraphs = {s.paragraph for s in group}
        for p in paragraphs:  # no paragraph is split across two chunks
            assert sum(1 for g in groups if any(s.paragraph == p for s in g)) == 1


def test_one_request_per_chunk_and_answers_cached(characters, tmp_path):
    text = "“One,” said Alice.\n\n“Two,” said the Hatter."
    answers = [labels_json((0, "Alice", "neutral")), labels_json((2, "Hatter", "neutral"))]
    llm = CachedLLM(tmp_path, "fake", FakeClient(*answers))
    attribute(segment(text), characters, llm, max_chars=25)
    assert llm.usage.requests == 2

    offline = CachedLLM(tmp_path, "fake", client=None)  # same prompts, no client
    lines = attribute(segment(text), characters, offline, max_chars=25)
    assert [line.speaker for line in lines] == ["Alice", "Narrator", "Hatter", "Narrator"]
    assert offline.usage.cache_hits == 2


def test_failed_answers_are_removed_from_the_cache(characters, tmp_path):
    client = FakeClient("not json", "still not json")
    with pytest.raises(ValueError):
        attribute(segment(TEXT), characters, CachedLLM(tmp_path, "fake", client))
    assert list(tmp_path.glob("*.json")) == []  # the next live run asks again
