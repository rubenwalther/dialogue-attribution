import pytest

from dialogue_attribution.evaluate import report, score
from dialogue_attribution.schema import GoldLabel, Line


def line(i, kind, speaker, emotion, text):
    return Line(index=i, kind=kind, speaker=speaker, emotion=emotion, text=text)


PREDICTED = [
    line(0, "quote", "Alice", "angry", "“There’s plenty of room!”"),
    line(1, "narration", "Narrator", "neutral", "said Alice."),
    line(2, "quote", "Hatter", "neutral", "“No room!”"),
]


def test_score_counts_speaker_and_emotion_and_accepts_any_listed_speaker():
    gold = [GoldLabel(index=0, text_start="“There’s", speaker=["Alice"], emotion=["neutral"]),
            GoldLabel(index=2, text_start="“No room", speaker=["Hatter", "March Hare"], emotion=["neutral"])]
    s = score(PREDICTED, gold)
    assert (s.quotes, s.speaker_correct, s.emotion_correct) == (2, 2, 1)
    assert s.speaker_accuracy == 1.0 and s.emotion_accuracy == 0.5


def test_speaker_errors_are_listed():
    gold = [GoldLabel(index=0, text_start="“There’s", speaker=["Hatter"], emotion=["angry"])]
    s = score(PREDICTED, gold)
    assert s.speaker_correct == 0
    assert "predicted Alice, gold Hatter" in s.errors[0]
    assert "Speaker errors" in report("test", s)


@pytest.mark.parametrize("gold", [
    GoldLabel(index=1, text_start="said", speaker=["Alice"], emotion=["neutral"]),       # not a quote
    GoldLabel(index=0, text_start="“Different", speaker=["Alice"], emotion=["neutral"]),  # text drifted
    GoldLabel(index=9, text_start="“", speaker=["Alice"], emotion=["neutral"]),           # no such segment
])
def test_gold_out_of_sync_is_an_error(gold):
    with pytest.raises(ValueError, match="does not match"):
        score(PREDICTED, [gold])


def test_several_gold_emotions_are_all_accepted():
    gold = [GoldLabel(index=0, text_start="“There’s", speaker=["Alice"], emotion=["surprised", "angry"])]
    assert score(PREDICTED, gold).emotion_correct == 1
