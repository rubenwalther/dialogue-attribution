"""Ask a language model who speaks each quote and with which emotion."""

import json

from pydantic import ValidationError

from .llm import CachedLLM
from .schema import EMOTIONS, NARRATOR, UNKNOWN, CharacterList, LabelResponse, Line
from .segment import Segment

SYSTEM_PROMPT = """You label dialogue in fiction.
The user sends a passage in which every quote is preceded by a marker like [12].
For every marker, decide who speaks that quote and with which emotion.
Rules:
- speaker must be exactly one name from the character list, or "Unknown" if the text does not make it clear.
- A quote that a character only thinks still belongs to that character.
- emotion must be one of: {emotions}.
- Label every marker exactly once. Do not change or repeat the text.
Answer with JSON only: {{"labels": [{{"id": 12, "speaker": "Alice", "emotion": "neutral"}}]}}"""


def chunks(segments: list[Segment], max_chars: int) -> list[list[Segment]]:
    """Group segments into chunks of whole paragraphs, each at most max_chars long.

    A single paragraph longer than max_chars becomes its own chunk.
    """
    result: list[list[Segment]] = []
    current: list[Segment] = []
    size = 0
    for paragraph in _paragraphs(segments):
        length = sum(len(s.text) + 1 for s in paragraph)
        if current and size + length > max_chars:
            result.append(current)
            current, size = [], 0
        current.extend(paragraph)
        size += length
    if current:
        result.append(current)
    return result


def _paragraphs(segments: list[Segment]) -> list[list[Segment]]:
    groups: list[list[Segment]] = []
    for s in segments:
        if groups and groups[-1][-1].paragraph == s.paragraph:
            groups[-1].append(s)
        else:
            groups.append([s])
    return groups


def build_messages(chunk: list[Segment], characters: CharacterList) -> list[dict]:
    """The passage with [id] markers in front of each quote, one paragraph per line."""
    cast = "\n".join(f"- {c.name}: {c.description}" if c.description else f"- {c.name}"
                     for c in characters.characters)
    lines = []
    for paragraph in _paragraphs(chunk):
        parts = [f"[{s.index}]{s.text}" if s.kind == "quote" else s.text for s in paragraph]
        lines.append(" ".join(parts))
    user = f"Characters:\n{cast}\n\nPassage:\n" + "\n\n".join(lines)
    return [{"role": "system", "content": SYSTEM_PROMPT.format(emotions=", ".join(EMOTIONS))},
            {"role": "user", "content": user}]


def parse_answer(answer: str, expected_ids: set[int], names: set[str]) -> LabelResponse:
    """Validate the model's JSON. Raises ValueError with a readable reason."""
    text = answer.strip()
    if text.startswith("```"):  # some models wrap JSON in a code block
        text = text.strip("`").removeprefix("json").strip()
    try:
        response = LabelResponse.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as err:
        raise ValueError(f"Answer is not valid JSON in the required shape: {err}") from err
    got = [label.id for label in response.labels]
    missing = expected_ids - set(got)
    if missing:
        raise ValueError(f"Missing markers: {sorted(missing)}")
    extra = set(got) - expected_ids
    if extra or len(got) != len(set(got)):
        raise ValueError(f"Unknown or repeated markers: {sorted(extra) or 'duplicates'}")
    wrong = sorted({label.speaker for label in response.labels} - names - {UNKNOWN})
    if wrong:
        raise ValueError(f"Speakers not in the character list: {wrong}")
    return response


def label_chunk(chunk: list[Segment], characters: CharacterList, llm: CachedLLM) -> dict[int, tuple[str, str]]:
    """Return {segment index: (speaker, emotion)} for the quotes in one chunk.

    If the first answer is invalid, the model gets one more try with the
    reason attached. If that fails too, both answers are removed from the
    cache (so the next live run asks again) and the error is raised: a wrong
    label should never pass silently.
    """
    expected = {s.index for s in chunk if s.kind == "quote"}
    if not expected:
        return {}
    names = set(characters.names())
    messages = build_messages(chunk, characters)
    answer = llm.ask(messages)
    try:
        response = parse_answer(answer, expected, names)
    except ValueError as err:
        retry = messages + [{"role": "assistant", "content": answer},
                            {"role": "user", "content": f"{err}. Answer again with corrected JSON only."}]
        try:
            response = parse_answer(llm.ask(retry), expected, names)
        except ValueError:
            llm.forget(messages)
            llm.forget(retry)
            raise
    return {label.id: (label.speaker, label.emotion) for label in response.labels}


def attribute(segments: list[Segment], characters: CharacterList, llm: CachedLLM,
              max_chars: int = 20000) -> list[Line]:
    """Label all quotes; narration always goes to the narrator with a neutral emotion."""
    labels: dict[int, tuple[str, str]] = {}
    for chunk in chunks(segments, max_chars):
        labels.update(label_chunk(chunk, characters, llm))
    lines = []
    for s in segments:
        speaker, emotion = labels.get(s.index, (NARRATOR, "neutral"))
        lines.append(Line(index=s.index, kind=s.kind, speaker=speaker, emotion=emotion, text=s.text))
    return lines
