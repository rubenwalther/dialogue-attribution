"""Data shapes shared by the pipeline, validated with pydantic."""

from typing import Literal, get_args

from pydantic import BaseModel, Field

# A small, fixed label set keeps the model's answers comparable and testable.
Emotion = Literal["neutral", "happy", "sad", "angry", "fearful", "surprised", "confused"]
EMOTIONS: tuple[str, ...] = get_args(Emotion)

NARRATOR = "Narrator"
UNKNOWN = "Unknown"


class Character(BaseModel):
    name: str
    description: str = ""


class CharacterList(BaseModel):
    """Input file: who can speak in this text. Written by hand for the demo."""
    characters: list[Character]

    def names(self) -> list[str]:
        return [c.name for c in self.characters]


class QuoteLabel(BaseModel):
    """One label as returned by the language model."""
    id: int
    speaker: str
    emotion: Emotion


class LabelResponse(BaseModel):
    """The JSON object the model must return."""
    labels: list[QuoteLabel]


class Line(BaseModel):
    """One segment of the chapter with its final speaker and emotion."""
    index: int
    kind: Literal["narration", "quote"]
    speaker: str
    emotion: Emotion
    text: str


class GoldLabel(BaseModel):
    """Hand-checked reference label for one quote.

    `speaker` may list several names when more than one character speaks
    ("they cried"); a prediction counts as correct if it names any of them.
    `emotion` may likewise list several emotions where careful readers
    disagreed; any of them counts as agreement.
    `text_start` guards against the gold file drifting out of sync with the
    segmenter.
    """
    index: int
    text_start: str
    speaker: list[str] = Field(min_length=1)
    emotion: list[Emotion] = Field(min_length=1)
