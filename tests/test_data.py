"""The shipped data files must agree with each other and with the segmenter."""

from conftest import DATA

from dialogue_attribution.evaluate import load_gold
from dialogue_attribution.segment import segment


def test_gold_labels_cover_every_quote_in_order(alice_text):
    quotes = [s for s in segment(alice_text) if s.kind == "quote"]
    gold = load_gold(DATA / "alice_ch07.gold.jsonl")
    assert [g.index for g in gold] == [q.index for q in quotes]
    for g, q in zip(gold, quotes, strict=True):
        assert q.text.startswith(g.text_start)


def test_gold_speakers_are_in_the_character_list(characters):
    names = set(characters.names())
    for g in load_gold(DATA / "alice_ch07.gold.jsonl"):
        assert set(g.speaker) <= names


def test_chapter_text_has_no_project_gutenberg_boilerplate(alice_text):
    assert "Gutenberg" not in alice_text
    assert alice_text.startswith("There was a table set out under a tree")
