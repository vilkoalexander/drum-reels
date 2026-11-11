from itertools import pairwise

import numpy as np
import pytest

from reels.config import Config
from reels.service import StageError, UsageError
from reels.subtitles import Caption, format_srt, format_txt, srt_time, write_all
from reels.tools import subs
from reels.tools.subs import (
    Options,
    Word,
    check_outputs,
    drop_isolated_captions,
    drop_junk_captions,
    drop_loops,
    group_captions,
    loudness_envelope,
    segment_words,
    shaky_captions,
    trim_words_to_sound,
)


def words(*spec: tuple[str, float, float], probability: float = 0.9) -> list[Word]:
    return [Word(text, start, end, probability) for text, start, end in spec]


def test_pause_starts_a_new_caption():
    captions = group_captions(words(("hold", 0.0, 0.4), ("on", 0.5, 0.8), ("tight", 2.0, 2.4)))
    assert [c.text for c in captions] == ["hold on", "tight"]


def test_caption_stays_within_42_characters():
    spec = [(f"word{i:02}", i * 0.3, i * 0.3 + 0.25) for i in range(10)]
    captions = group_captions(words(*spec))
    assert len(captions) > 1
    assert all(len(c.text) <= 42 for c in captions)


def test_caption_duration_is_capped():
    spec = [("la", i * 0.5, i * 0.5 + 0.4) for i in range(20)]
    captions = group_captions(words(*spec))
    assert all(c.end - c.start <= 6.0 for c in captions)


def test_captions_never_overlap_or_end_before_they_start():
    captions = group_captions(words(("a", 1.0, 1.0), ("b", 3.0, 2.0), ("c", 3.1, 3.2), ("d", 9.0, 9.1)))
    for caption in captions:
        assert caption.end > caption.start
    for earlier, later in pairwise(captions):
        assert later.start >= earlier.end


def test_silent_stretch_loses_its_words():
    sr = 16000
    vocals = np.zeros(sr * 10, dtype=np.float32)
    vocals[: sr * 2] = 0.5
    envelope = loudness_envelope(vocals, sr)
    kept = trim_words_to_sound(words(("sung", 0.5, 1.0), ("invented", 5.0, 5.5)), envelope)
    assert [w.text for w in kept] == ["sung"]


def test_srt_time_format():
    assert srt_time(0) == "00:00:00,000"
    assert srt_time(3725.5) == "01:02:05,500"
    assert srt_time(1.9996) == "00:00:02,000"


def test_srt_numbers_captions_from_one():
    srt = format_srt([Caption(1.0, 2.5, "first", 0.9), Caption(3.0, 4.0, "second", 0.9)])
    assert srt == "1\n00:00:01,000 --> 00:00:02,500\nfirst\n\n2\n00:00:03,000 --> 00:00:04,000\nsecond\n"


def test_txt_blank_line_only_after_a_long_gap():
    captions = [Caption(0, 2, "one", 1), Caption(3, 5, "two", 1), Caption(10, 12, "three", 1)]
    assert format_txt(captions) == "one\ntwo\n\nthree\n"


def test_shaky_lists_low_probability_captions_with_minutes_and_seconds():
    captions = [Caption(1, 2, "clear", 0.95), Caption(64.4, 66, "mumbled", 0.4)]
    assert shaky_captions(captions) == [{"at": "01:04", "text": "mumbled"}]


def test_stem_without_vocals_keeps_no_words():
    noise = np.random.default_rng(0).normal(0, 1e-5, 16000 * 10).astype(np.float32)
    assert trim_words_to_sound(words(("invented", 1.0, 2.0)), loudness_envelope(noise, 16000)) == []
    assert trim_words_to_sound(words(("invented", 1.0, 2.0)), np.zeros(200)) == []


def test_word_is_trimmed_to_its_loud_windows():
    sr = 16000
    vocals = np.zeros(sr * 40, dtype=np.float32)
    vocals[sr * 31 : sr * 32] = 0.5
    trimmed = trim_words_to_sound(words(("stretched", 31.0, 61.0)), loudness_envelope(vocals, sr))
    assert trimmed[0].start == pytest.approx(31.0, abs=0.06)
    assert trimmed[0].end == pytest.approx(32.0, abs=0.06)


def segment(text, no_speech=0.1, logprob=-0.3, probability=0.9):
    return {"no_speech_prob": no_speech, "avg_logprob": logprob,
            "words": [{"word": f" {t}", "start": 0.0, "end": 1.0, "probability": probability} for t in text.split()]}


def test_segments_that_whisper_doubts_are_dropped():
    segments = [segment("good"), segment("silence", no_speech=0.9), segment("guess", logprob=-1.5)]
    assert [w.text for w in segment_words(segments)] == ["good"]


def test_unlikely_and_letterless_words_are_dropped():
    segments = [segment("good"), segment("maybe", probability=0.05), segment(","), segment("שלום")]
    assert [w.text for w in segment_words(segments)] == ["good", "שלום"]


def test_captions_far_below_the_shaky_line_are_dropped_but_the_middle_band_stays():
    captions = [Caption(0, 1, "clear", 0.9), Caption(2, 3, "unsure", 0.5), Caption(4, 5, "junk", 0.2),
                Caption(6, 7, ",", 0.9)]
    assert [c.text for c in drop_junk_captions(captions)] == ["clear", "unsure"]


def test_one_stretched_word_still_respects_the_duration_cap():
    caption = group_captions(words(("hobbies", 98.0, 114.0)))[0]
    assert caption.end - caption.start <= 6.0


def test_line_breaks_after_punctuation_once_long_enough():
    spec = [("I", 0.0, 0.2), ("walk", 0.3, 0.5), ("along", 0.6, 0.8), ("the", 0.9, 1.0), ("river", 1.1, 1.3),
            ("tonight,", 1.4, 1.8), ("the", 1.9, 2.0), ("lights", 2.1, 2.4)]
    assert [c.text for c in group_captions(words(*spec))] == ["I walk along the river tonight,", "the lights"]
    short = group_captions(words(("Hey,", 0.0, 0.3), ("you", 0.4, 0.6)))
    assert [c.text for c in short] == ["Hey, you"]


def test_out_must_end_in_srt_before_any_model_work(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    with pytest.raises(StageError, match=r"\.srt"):
        check_outputs(Options(video=str(video), out=str(tmp_path / "x.txt")))


def test_existing_outputs_are_refused_without_force(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    (tmp_path / "v.en.txt").write_text("hand edited")
    with pytest.raises(StageError, match="v.en.txt already exists"):
        check_outputs(Options(video=str(video)))
    assert check_outputs(Options(video=str(video), force=True)) == {"en": tmp_path / "v.en.srt"}


def test_failed_second_write_leaves_neither_file(tmp_path, monkeypatch):
    real_replace = __import__("os").replace
    monkeypatch.setattr("reels.subtitles.os.replace",
                        lambda a, b: (_ for _ in ()).throw(OSError("disk")) if str(b).endswith(".txt") else real_replace(a, b))
    with pytest.raises(OSError):
        write_all({tmp_path / "a.srt": "srt", tmp_path / "a.txt": "txt"})
    assert list(tmp_path.iterdir()) == []


def run_of(text: str, start: float = 0.0, step: float = 0.3, held: dict[str, float] | None = None) -> list[Word]:
    """Words laid end to start like Whisper does on singing: no gaps inside a phrase."""
    out, t = [], start
    for token in text.split():
        length = (held or {}).get(token, step)
        out.append(Word(token, t, t + length, 0.9))
        t += length
    return out


def test_capitalised_word_after_an_unpunctuated_one_starts_a_new_line():
    captions = group_captions(run_of("Praise the Lord for he is good And his mercy is forever"))
    assert [c.text for c in captions] == ["Praise the Lord for he is good", "And his mercy is forever"]


def test_capital_after_a_short_word_does_not_split_a_name():
    assert [c.text for c in group_captions(run_of("Praise the Lord"))] == ["Praise the Lord"]


def test_overflow_breaks_at_a_phrase_boundary_not_at_the_limit():
    text = "How good and how pleasant it is for us to dwell together."
    captions = group_captions(run_of(text, held={"pleasant": 0.9, "it": 0.1, "is": 0.1}))
    assert [c.text for c in captions] == ["How good and how pleasant", "it is for us to dwell together."]


def test_overflow_prefers_the_gap_between_words():
    words = run_of("we sing and we dance for joy in the house of the Lord today")
    shifted = [w if i < 8 else Word(w.text, w.start + 0.5, w.end + 0.5, w.probability) for i, w in enumerate(words)]
    captions = group_captions(shifted)
    assert captions[0].text == "we sing and we dance for joy in"
    assert all(len(c.text) <= 42 for c in captions)


def test_a_breath_gap_shorter_than_the_pause_rule_ends_a_long_enough_line():
    words = run_of("lift up your hands") + run_of("and sing to him", start=1.2 + 0.5)
    assert [c.text for c in group_captions(words)] == ["lift up your hands", "and sing to him"]


def test_short_lone_word_joins_the_next_line_when_it_starts_within_a_second():
    words = [Word("And", 401.08, 401.45, 0.9), *run_of("his mercy is forever", start=402.35)]
    assert [c.text for c in group_captions(words)] == ["And his mercy is forever"]


def test_short_lone_word_far_from_the_next_line_stays():
    words = [Word("And", 100.0, 100.4, 0.9), *run_of("his mercy is forever", start=104.0)]
    assert next(c.text for c in group_captions(words)) == "And"


def caption_at(start: float, text: str) -> Caption:
    return Caption(start, start + 1.0, text, 0.9)


def test_lone_filler_far_from_every_other_caption_is_dropped():
    captions = [caption_at(4.7, "you."), caption_at(62.0, "How good and how pleasant"),
                caption_at(66.0, "dwell together"), caption_at(150.0, "Yo!")]
    assert [c.text for c in drop_isolated_captions(captions)] == ["How good and how pleasant", "dwell together"]


def test_short_line_close_to_others_stays():
    captions = [caption_at(10.0, "Praise the Lord for he is good"), caption_at(15.0, "Amen."),
                caption_at(20.0, "And his mercy is forever")]
    assert drop_isolated_captions(captions) == captions


def test_longer_line_in_silence_stays():
    captions = [caption_at(4.0, "Hallelujah to the Lord"), caption_at(60.0, "that is the end")]
    assert len(drop_isolated_captions(captions)) == 2


REFRAIN = "Praise the Lord for He is good And His mercy is forever"


def test_refrain_repeated_six_times_is_kept():
    refrain = segment(" ".join([REFRAIN] * 6))
    assert len(segment_words([refrain])) == 72


def test_single_token_loop_segment_is_dropped():
    assert segment_words([segment("Oh Oh Oh Oh Oh Oh Oh"), segment("I I I I")]) == []


def test_two_token_loop_segment_is_dropped():
    assert segment_words([segment("oh yo oh yo oh")]) == []


def test_run_of_one_word_captions_is_dropped():
    captions = [caption_at(10.0 + 3 * i, "I") for i in range(6)]
    assert drop_loops(captions) == []


def test_repeated_oh_inside_one_caption_is_dropped_but_neighbours_stay():
    captions = [caption_at(10.0, "Praise the Lord"), caption_at(40.0, "Oh Oh Oh Oh Oh Oh Oh"),
                caption_at(80.0, "And his mercy is forever")]
    assert [c.text for c in drop_loops(captions)] == ["Praise the Lord", "And his mercy is forever"]


def test_repeated_refrain_captions_are_kept():
    captions = [caption_at(10.0 + 5 * i, text) for i, text in enumerate(
        ["Praise the Lord", "for he is good", "Praise the Lord", "for he is good"])]
    assert drop_loops(captions) == captions


def test_one_word_sung_refrain_is_kept_as_segment_and_as_captions():
    assert len(segment_words([segment("Hallelujah, hallelujah, hallelujah, hallelujah")])) == 4
    assert len(segment_words([segment("Hosanna Hosanna Hosanna Hosanna Amen Amen")])) == 6
    captions = [caption_at(10.0, "Holy, holy, holy"), caption_at(14.0, "Holy, holy, holy")]
    assert drop_loops(captions) == captions


def test_short_word_loop_is_still_dropped_across_captions():
    captions = [caption_at(10.0, "Oh Oh"), caption_at(13.0, "Oh Oh")]
    assert drop_loops(captions) == []


def test_missing_library_or_unknown_language_fails_before_the_pipeline(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    monkeypatch.setattr(subs.media, "write_atomic", lambda *a: pytest.fail("audio extraction started"))
    monkeypatch.setattr(subs, "separate_vocals", lambda path: pytest.fail("listening started"))
    monkeypatch.setattr(subs.importlib.util, "find_spec", lambda name: None if name == "demucs" else object())
    with pytest.raises(StageError, match=r"demucs is not installed: pip install -e '\.\[subs\]'"):
        subs.run(subs.Options(video=str(video)), Config())
    monkeypatch.setattr(subs.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(subs, "whisper_languages", lambda: {"en", "he"})
    with pytest.raises(UsageError, match="'xx'"):
        subs.run(subs.Options(video=str(video), lang="xx"), Config())


def test_whisper_language_table_comes_from_mlx_whisper():
    pytest.importorskip("mlx_whisper")
    languages = subs.whisper_languages()
    assert {"en", "he", "uk"} <= languages and "xx" not in languages


def test_run_wires_listening_filters_and_output(tmp_path, monkeypatch):
    video = tmp_path / "song.mp4"
    video.write_bytes(b"")
    vocals = np.zeros(16000 * 70, dtype=np.float32)
    vocals[: 16000 * 6] = np.random.default_rng(0).normal(0, 0.3, 16000 * 6)
    vocals[16000 * 60 : 16000 * 61] = 0.3
    heard = [Word("Praise", 1.0, 1.8, 0.9), Word("the", 1.8, 2.2, 0.9), Word("Lord", 2.2, 3.0, 0.9),
             Word("invented", 20.0, 21.0, 0.9), Word("you.", 60.0, 61.0, 0.9)]
    monkeypatch.setattr(subs.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(subs, "whisper_languages", lambda: {"en"})
    monkeypatch.setattr(subs.media, "write_atomic", lambda *args: None)
    monkeypatch.setattr(subs, "separate_vocals", lambda path: vocals)
    monkeypatch.setattr(subs, "transcribe_words", lambda audio, lang: heard)
    result = subs.run(subs.Options(video=str(video)), Config())
    assert result == {"ok": True, "tracks": {"en": {"srt": str(tmp_path / "song.en.srt"),
                                                    "txt": str(tmp_path / "song.en.txt")}},
                      "captions": 1, "shaky": [], "cleaned": []}
    assert (tmp_path / "song.en.srt").read_text() == "1\n00:00:01,000 --> 00:00:03,000\nPraise the Lord\n"
    assert (tmp_path / "song.en.txt").read_text() == "Praise the Lord\n"
