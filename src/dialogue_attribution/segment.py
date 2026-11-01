"""Split prose into narration and quoted speech.

This step is deterministic on purpose: code decides where a quote starts
and ends, the language model only labels the quotes afterwards. That way
the model can never change, drop or invent text.
"""

import re
from dataclasses import dataclass
from typing import Literal

# Curly double quotes as used in most English e-books, plus straight quotes.
OPENING = {"“": "”", '"': '"'}

# A paragraph break is a blank line (which may contain spaces or tabs).
PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


@dataclass(frozen=True)
class Segment:
    index: int                           # position in the chapter, 0-based
    kind: Literal["narration", "quote"]
    text: str                            # whitespace collapsed, quote marks kept
    start: int                           # offset in the text (inclusive), after \r\n -> \n
    end: int                             # offset in the text (exclusive)
    paragraph: int                       # paragraph in which the segment starts, 0-based


def _paragraph_starts(text: str) -> list[int]:
    """Offsets where a new paragraph begins."""
    return [0] + [m.end() for m in PARAGRAPH_BREAK.finditer(text)]


def _paragraph_of(offset: int, starts: list[int]) -> int:
    return sum(1 for start in starts[1:] if offset >= start)


def _starts_paragraph(text: str, pos: int) -> bool:
    """True if only a paragraph break (and spaces) come right before position pos."""
    window = text[max(0, pos - 100):pos]  # a short look-back keeps this fast on long books
    return re.search(r"\n[ \t]*\n[ \t]*\Z", window) is not None


def _spans(text: str) -> list[tuple[str, int, int]]:
    """Return (kind, start, end) spans that cover the whole text.

    A quote runs from an opening mark to its matching closing mark. Curly
    quotes may be nested (“… “much of a muchness” …”), so we count depth.
    Single quotes are ignored, because ’ is also an apostrophe; nested ‘…’
    simply stays inside the surrounding quote. Long speeches that span
    several paragraphs often open each paragraph with a quote mark and never
    close the previous one; a quote mark at the start of a paragraph while a
    quote is still open therefore ends that quote and starts a new one.
    """
    spans = []
    i = 0           # start of the current span
    closing = None  # expected closing mark while inside a quote
    depth = 0       # nesting level of curly quotes
    for pos, ch in enumerate(text):
        if closing is None:
            if ch in OPENING:
                if pos > i:
                    spans.append(("narration", i, pos))
                i, closing, depth = pos, OPENING[ch], 1
        elif ch in OPENING and _starts_paragraph(text, pos):
            spans.append(("quote", i, pos))  # unclosed speech continues in a new paragraph
            i, closing, depth = pos, OPENING[ch], 1
        elif closing == "”" and ch == "“":
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                spans.append(("quote", i, pos + 1))
                i, closing = pos + 1, None
    if i < len(text):
        spans.append(("quote" if closing else "narration", i, len(text)))
    return spans


def segment(text: str) -> list[Segment]:
    """Split a chapter into narration and quote segments, in reading order.

    Narration is also cut at paragraph boundaries. Pieces without any letter
    or digit (for example a lone "(" between two quotes) are dropped.
    """
    text = text.replace("\r\n", "\n")  # Windows line endings
    starts = _paragraph_starts(text)
    segments: list[Segment] = []
    for kind, start, end in _spans(text):
        pieces = [(start, end)]
        if kind == "narration":
            bounds = [start, *[s for s in starts if start < s < end], end]
            pieces = list(zip(bounds, bounds[1:], strict=False))
        for a, b in pieces:
            raw = text[a:b]
            if not any(c.isalnum() for c in raw):
                continue
            lead = len(raw) - len(raw.lstrip())  # point at the first visible character
            segments.append(Segment(
                index=len(segments),
                kind=kind,
                text=" ".join(raw.split()),
                start=a + lead,
                end=b,
                paragraph=_paragraph_of(a + lead, starts),
            ))
    return segments
