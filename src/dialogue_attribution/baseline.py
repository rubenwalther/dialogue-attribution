"""A simple rule-based baseline, so the LLM result has something to beat.

Rule: a quote belongs to the first character name in the narration right
after it ("…," said the Hatter), or else to the last name in the narration
right before it (The Hatter said: "…"), within the same paragraph. Quotes
still unresolved take the speaker of another quote in the same paragraph.
Emotion is always "neutral". No API key needed.
"""

from .schema import NARRATOR, UNKNOWN, Line
from .segment import Segment


def _names_in(text: str, names: list[str]) -> list[tuple[int, str]]:
    """(position, name) for every known name in the text, longest names first."""
    found = []
    taken = [False] * len(text)
    for name in sorted(names, key=len, reverse=True):
        start = text.find(name)
        while start != -1:
            if not any(taken[start:start + len(name)]):
                found.append((start, name))
                taken[start:start + len(name)] = [True] * len(name)
            start = text.find(name, start + 1)
    return sorted(found)


def baseline(segments: list[Segment], names: list[str]) -> list[Line]:
    speaker: dict[int, str] = {}
    for i, s in enumerate(segments):
        if s.kind != "quote":
            continue
        after = segments[i + 1] if i + 1 < len(segments) else None
        before = segments[i - 1] if i > 0 else None
        if after and after.kind == "narration" and after.paragraph == s.paragraph:
            hits = _names_in(after.text, names)
            if hits:
                speaker[s.index] = hits[0][1]
                continue
        if before and before.kind == "narration" and before.paragraph == s.paragraph:
            hits = _names_in(before.text, names)
            if hits:
                speaker[s.index] = hits[-1][1]

    # Second pass: borrow from another quote in the same paragraph.
    for s in segments:
        if s.kind == "quote" and s.index not in speaker:
            same = [q for q in segments if q.kind == "quote" and q.paragraph == s.paragraph
                    and q.index in speaker]
            if same:
                nearest = min(same, key=lambda q: abs(q.index - s.index))
                speaker[s.index] = speaker[nearest.index]

    return [Line(index=s.index, kind=s.kind,
                 speaker=speaker.get(s.index, UNKNOWN) if s.kind == "quote" else NARRATOR,
                 emotion="neutral", text=s.text)
            for s in segments]
