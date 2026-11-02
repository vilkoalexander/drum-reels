import pytest

from reels.analysis.notes import Moment, NotesError, parse, slug

PEOPLE = ("guitar", "keys", "bass", "vocals", "vocal", "singer", "piano")

BRIEF_LINES = """\
1:12:30 fill
1:14:05 chorus
1:20:10 guitar solo
1:33:00 groove 60
"""


def test_brief_example_lines_parse():
    moments = parse(BRIEF_LINES, PEOPLE)
    assert [m.noted_s for m in moments] == [4350, 4445, 4810, 5580]
    assert [m.len_s for m in moments] == [None, None, None, 60]
    assert [m.person for m in moments] == [None, None, "guitar", None]
    assert [m.label for m in moments] == ["fill", "chorus", "guitar solo", "groove"]


def test_minutes_seconds_and_padded_hours_accepted():
    moments = parse("4:30 intro\n01:12:30 fill\n", PEOPLE)
    assert [m.noted_s for m in moments] == [270, 4350]


def test_blank_and_comment_lines_ignored():
    text = "\n  # set one\n1:12:30 fill\n\n# 1:00:00 skipped\n"
    assert [m.label for m in parse(text, PEOPLE)] == ["fill"]


def test_bad_line_raises_with_line_number_and_text():
    text = "1:12:30 fill\n\n# note\n1:14:5 chorus\n"
    with pytest.raises(NotesError, match="line 4") as info:
        parse(text, PEOPLE)
    assert "1:14:5 chorus" in str(info.value)


def test_time_without_label_is_rejected():
    with pytest.raises(NotesError, match="line 1"):
        parse("1:12:30\n", PEOPLE)


def test_seconds_out_of_range_is_rejected():
    with pytest.raises(NotesError, match="line 1"):
        parse("1:12:75 fill\n", PEOPLE)


def test_person_is_first_whole_word_match_case_insensitive():
    moments = parse("1:00:00 Guitar and keys\n1:00:10 guitarist solo\n1:00:20 drums\n", PEOPLE)
    assert [m.person for m in moments] == ["guitar", None, None]


def test_person_normalisation():
    moments = parse("1:00:00 vocal\n1:00:10 Singer\n1:00:20 vocals\n1:00:30 piano\n", PEOPLE)
    assert [m.person for m in moments] == ["vocals", "vocals", "vocals", "keys"]


def test_person_only_matches_configured_people():
    assert parse("1:00:00 guitar\n", ("drums",))[0].person is None


def test_slug_examples():
    assert slug(Moment(4350.0, "fill", None, None)) == "1-12-30_fill"
    assert slug(parse("0:02:30 Guitar Solo!!", PEOPLE)[0]) == "0-02-30_guitar-solo"


def test_short_or_zero_length_is_rejected_with_a_hint():
    with pytest.raises(NotesError, match=r"line 1: length 2 s is below the minimum 5 s.*read as a length"):
        parse("0:01:00 verse 2\n", [], min_len_s=5)
    with pytest.raises(NotesError, match="length must be positive"):
        parse("0:01:00 fill 0\n", [])
    assert parse("0:01:00 verse 2\n", [], min_len_s=2)[0].len_s == 2


def test_label_without_ascii_letters_keeps_the_note_with_a_time_only_slug():
    moment = parse("0:01:00 соло\n", [])[0]
    assert moment.label == "соло"
    assert slug(moment) == "0-01-00"
