"""Compare predicted labels with hand-checked gold labels."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .schema import GoldLabel, Line


@dataclass
class Score:
    quotes: int = 0
    speaker_correct: int = 0
    emotion_correct: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def speaker_accuracy(self) -> float:
        return self.speaker_correct / self.quotes if self.quotes else 0.0

    @property
    def emotion_accuracy(self) -> float:
        return self.emotion_correct / self.quotes if self.quotes else 0.0


def load_gold(path: Path) -> list[GoldLabel]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [GoldLabel.model_validate_json(line) for line in lines if line.strip()]


def load_lines(path: Path) -> list[Line]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Line.model_validate_json(line) for line in lines if line.strip()]


def save_lines(lines: list[Line], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(line.model_dump(), ensure_ascii=False) + "\n" for line in lines)
    Path(path).write_text(text, encoding="utf-8")


def score(predicted: list[Line], gold: list[GoldLabel]) -> Score:
    """Speaker and emotion accuracy over all gold quotes.

    Raises ValueError if the gold file does not match the predicted segments,
    so a changed segmenter can never produce a silently wrong score.
    """
    by_index = {line.index: line for line in predicted}
    result = Score()
    for g in gold:
        line = by_index.get(g.index)
        if line is None or line.kind != "quote" or not line.text.startswith(g.text_start):
            raise ValueError(f"Gold label {g.index} does not match the segments "
                             f"(expected a quote starting with {g.text_start!r}).")
        result.quotes += 1
        if line.speaker in g.speaker:
            result.speaker_correct += 1
        else:
            result.errors.append(f"[{g.index}] {line.text[:60]!r}: "
                                 f"predicted {line.speaker}, gold {' or '.join(g.speaker)}")
        if line.emotion in g.emotion:
            result.emotion_correct += 1
    return result


def report(name: str, s: Score) -> str:
    """A short Markdown summary."""
    rows = [f"### {name}", "",
            "| Quotes | Speaker accuracy | Emotion agreement |",
            "|---:|---:|---:|",
            f"| {s.quotes} | {s.speaker_accuracy:.1%} ({s.speaker_correct}/{s.quotes}) "
            f"| {s.emotion_accuracy:.1%} ({s.emotion_correct}/{s.quotes}) |", ""]
    if s.errors:
        rows += ["Speaker errors:", ""] + [f"- {e}" for e in s.errors] + [""]
    return "\n".join(rows)
