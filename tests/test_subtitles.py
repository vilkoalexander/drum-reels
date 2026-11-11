import pytest

from reels.service import StageError
from reels.subtitles import Caption, check_free, clock, format_srt, parse_srt

CAPTIONS = [Caption(0.0, 2.5, "first line"), Caption(3.0, 4.0, "second, with comma"), Caption(3725.5, 3727.0, "late")]


def test_parse_reads_what_format_writes():
    assert parse_srt(format_srt(CAPTIONS)) == CAPTIONS


def test_parse_tolerates_hand_edits():
    text = ("﻿1\r\n00:00:01,000 --> 00:00:02,000\r\nfirst\r\nsecond half\r\n\r\n\r\n\r\n"
            "7\r\n00:00:03.5 --> 00:00:04.25\r\nthird\r\n\r\n")
    assert parse_srt(text) == [Caption(1.0, 2.0, "first second half"), Caption(3.5, 4.25, "third")]


def test_parse_without_index_lines():
    assert parse_srt("00:00:01,000 --> 00:00:02,000\nhello\n") == [Caption(1.0, 2.0, "hello")]


@pytest.mark.parametrize("text", [
    "", "1\nnot a timing\ntext\n",
    "1\n00:00:01,000 --> 00:00:02,000\nfirst\n2\n00:00:03,000 --> 00:00:04,000\nsecond\n",
])
def test_parse_rejects_unreadable_files(text):
    with pytest.raises(StageError):
        parse_srt(text)


def test_clock_is_minutes_and_seconds():
    assert clock(64.9) == "01:04"


def test_existing_files_are_refused_unless_forced(tmp_path):
    present = tmp_path / "a.srt"
    present.write_text("mine")
    with pytest.raises(StageError, match="a.srt already exists"):
        check_free([present, tmp_path / "b.srt"], force=False)
    check_free([present], force=True)
