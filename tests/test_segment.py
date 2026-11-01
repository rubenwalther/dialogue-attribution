from dialogue_attribution.segment import segment


def kinds(text):
    return [(s.kind, s.text) for s in segment(text)]


def test_quote_and_narration():
    assert kinds("“Have some wine,” the March Hare said.") == [
        ("quote", "“Have some wine,”"),
        ("narration", "the March Hare said."),
    ]


def test_two_quotes_around_a_speech_tag():
    text = "“I didn’t know,” said Alice; “it’s laid for more.”"
    assert [k for k, _ in kinds(text)] == ["quote", "narration", "quote"]


def test_apostrophes_and_single_quotes_stay_inside():
    text = "“You might say that ‘I see what I eat’ is the same thing!” said the Hatter."
    segs = segment(text)
    assert segs[0].kind == "quote"
    assert "‘I see what I eat’" in segs[0].text


def test_nested_double_quotes():
    text = "“you say things are “much of a muchness”—did you ever see one?” said X."
    segs = segment(text)
    assert [s.kind for s in segs] == ["quote", "narration"]
    assert segs[0].text.endswith("ever see one?”")


def test_quote_spanning_paragraphs_when_closed():
    text = "“I had to sing\n\n‘Twinkle, twinkle!’\n\nYou know the song?” he asked."
    segs = segment(text)
    assert segs[0].kind == "quote"
    assert segs[0].text == "“I had to sing ‘Twinkle, twinkle!’ You know the song?”"


def test_multi_paragraph_speech_without_closing_mark():
    text = "“First paragraph of a long speech.\n\n“Second paragraph,” she said."
    segs = segment(text)
    assert [s.kind for s in segs] == ["quote", "quote", "narration"]
    assert segs[1].paragraph == 1


def test_straight_quotes():
    assert [k for k, _ in kinds('"Hello," said Alice. "Bye."')] == ["quote", "narration", "quote"]


def test_unclosed_quote_at_end_is_still_a_quote():
    assert kinds("He said “wait") == [("narration", "He said"), ("quote", "“wait")]


def test_narration_is_cut_at_paragraphs_and_punctuation_only_pieces_dropped():
    text = "First paragraph.\n\nSecond paragraph. (“Quiet,” she said.)"
    segs = segment(text)
    assert [s.text for s in segs] == ["First paragraph.", "Second paragraph. (",
                                      "“Quiet,”", "she said.)"]
    assert [s.paragraph for s in segs] == [0, 1, 1, 1]


def test_indices_are_sequential_and_offsets_point_into_the_text():
    text = "Alice sighed.\n\n“I think so,” she said, “but maybe not.”"
    segs = segment(text)
    assert [s.index for s in segs] == list(range(len(segs)))
    for s in segs:
        assert " ".join(text[s.start:s.end].split()) == s.text


def test_no_letter_is_lost(alice_text):
    """Every letter and digit of the chapter ends up in exactly one segment."""
    segs = segment(alice_text)
    covered = [False] * len(alice_text)
    for s in segs:
        for pos in range(s.start, s.end):
            assert not covered[pos]
            covered[pos] = True
    lost = [c for c, cov in zip(alice_text, covered, strict=True) if c.isalnum() and not cov]
    assert lost == []


def test_straight_quotes_spanning_paragraphs():
    segs = segment('"First paragraph.\n\n"Second," she said.')
    assert [(s.kind, s.text) for s in segs] == [
        ("quote", '"First paragraph.'), ("quote", '"Second,"'), ("narration", "she said.")]


def test_windows_line_endings_and_blank_lines_with_spaces():
    unix = segment("One.\n\nTwo. “Hi,” she said.\n \nThree.")
    windows = segment("One.\r\n\r\nTwo. “Hi,” she said.\r\n \r\nThree.")
    assert [(s.kind, s.text, s.paragraph) for s in windows] == \
           [(s.kind, s.text, s.paragraph) for s in unix]
    assert [s.paragraph for s in unix] == [0, 1, 1, 1, 2]
