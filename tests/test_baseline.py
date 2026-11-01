from dialogue_attribution.baseline import baseline
from dialogue_attribution.segment import segment

NAMES = ["Alice", "Hatter", "March Hare", "Dormouse"]


def speakers(text):
    return [(line.text, line.speaker) for line in baseline(segment(text), NAMES) if line.kind == "quote"]


def test_name_after_the_quote_wins():
    assert speakers("“There isn’t any,” said the March Hare to Alice.") == [
        ("“There isn’t any,”", "March Hare")]


def test_name_before_the_quote_is_used_when_nothing_follows():
    assert speakers("The Hatter said: “Why is a raven like a writing-desk?”") == [
        ("“Why is a raven like a writing-desk?”", "Hatter")]


def test_longer_name_is_not_mistaken_for_a_shorter_one():
    assert speakers("“Nor I,” said the March Hare.")[0][1] == "March Hare"


def test_pronoun_quote_borrows_from_same_paragraph():
    text = "“I don’t know,” said Alice; “it’s laid for more.”"
    assert [s for _, s in speakers(text)] == ["Alice", "Alice"]


def test_unresolved_quote_is_unknown_and_emotion_always_neutral():
    lines = baseline(segment("“Wake up!” And they pinched it."), NAMES)
    assert lines[0].speaker == "Unknown"
    assert {line.emotion for line in lines} == {"neutral"}
