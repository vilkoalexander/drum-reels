from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from reels.config import Config
from reels.service import StageError, UsageError
from reels.subtitles import Caption, format_srt
from reels.tools import subs, translate
from reels.tools.translate import (
    Options,
    cleaned_lines,
    lang_of_name,
    merge_tracks,
    plan_tracks,
    request_lines,
)

CAPTIONS = [Caption(1.0, 2.0, "Baruch haba"), Caption(3.0, 5.5, "we sing")]


def reply(count, **extra):
    return {"lines": [{"index": i, "source": f"s{i}", "uk": f"uk{i}", "he": f"he{i}", **extra} for i in range(1, count + 1)]}


def test_language_comes_from_the_file_name_or_from():
    assert lang_of_name(Path("song.uk.srt"), None) == "uk"
    assert lang_of_name(Path("song.srt"), "he") == "he"
    assert lang_of_name(Path("song.uk.srt"), "en") == "en"


@pytest.mark.parametrize("name, given", [("song.srt", None), ("song.fr.srt", None), ("song.srt", "fr")])
def test_unknown_or_missing_language_is_a_usage_error(name, given):
    with pytest.raises(UsageError):
        lang_of_name(Path(name), given)


def test_tracks_are_planned_beside_the_source_without_touching_it():
    paths = plan_tracks(Path("/m/song.en.srt"), "en")
    assert paths == {"en": Path("/m/song.en.srt"), "uk": Path("/m/song.uk.srt"), "he": Path("/m/song.he.srt")}
    assert plan_tracks(Path("/m/x.srt"), "he")["uk"] == Path("/m/x.uk.srt")


def test_merge_copies_timings_and_keeps_the_source_text_when_not_cleaning():
    tracks = merge_tracks(CAPTIONS, reply(2)["lines"], "en", clean=False)
    assert [c.text for c in tracks["en"]] == ["Baruch haba", "we sing"]
    assert [c.text for c in tracks["uk"]] == ["uk1", "uk2"]
    for lang in ("en", "uk", "he"):
        assert [(c.start, c.end) for c in tracks[lang]] == [(1.0, 2.0), (3.0, 5.5)]


def test_merge_takes_the_cleaned_source_when_cleaning():
    tracks = merge_tracks(CAPTIONS, reply(2)["lines"], "en", clean=True)
    assert [c.text for c in tracks["en"]] == ["s1", "s2"]


def test_changed_source_lines_are_reported_with_time():
    after = [Caption(1.0, 2.0, "Baruch haba (Blessed is he who comes)"), CAPTIONS[1]]
    assert cleaned_lines(CAPTIONS, after) == [
        {"at": "00:01", "was": "Baruch haba", "now": "Baruch haba (Blessed is he who comes)"}]


def test_wrong_count_is_retried_once_then_accepted():
    calls = []

    def ask(system, user, schema):
        calls.append(user)
        return reply(1) if len(calls) == 1 else reply(2)

    assert len(request_lines(["a", "b"], "en", False, ask)) == 2
    assert len(calls) == 2 and "exactly 2 lines" in calls[1]


def test_wrong_indices_twice_fail():
    calls = []

    def ask(system, user, schema):
        calls.append(user)
        return {"lines": [{"index": 2, "source": "a", "uk": "a", "he": "a"}, {"index": 1, "source": "b", "uk": "b", "he": "b"}]}

    with pytest.raises(StageError, match="twice"):
        request_lines(["a", "b"], "en", False, ask)
    assert len(calls) == 2


def test_prompt_states_the_convention_and_the_task_variant():
    seen = []

    def ask(system, user, schema):
        seen.append((system, schema))
        return reply(1)

    request_lines(["a"], "en", True, ask)
    request_lines(["a"], "en", False, ask)
    cleaning, keeping = seen[0][0], seen[1][0]
    assert "Baruch haba (Blessed is he who comes)" in cleaning and "Hallelujah" in cleaning
    assert "Do not rewrite" in cleaning and "exactly as given" in keeping and "Do not rewrite" not in keeping
    assert set(seen[0][1]["properties"]["lines"]["items"]["properties"]) == {"index", "source", "uk", "he"}
    assert '"uk" (Ukrainian), "he" (Hebrew)' in cleaning


def test_name_and_language_that_disagree_are_a_usage_error():
    with pytest.raises(UsageError, match="named as uk"):
        plan_tracks(Path("song.uk.srt"), "en")
    with pytest.raises(UsageError):
        lang_of_name(Path("x.srt"), "fr")
    assert plan_tracks(Path("song.uk.srt"), "uk")["en"] == Path("song.en.srt")


def test_subs_translate_rejects_out_name_for_another_language(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    with pytest.raises(UsageError):
        subs.check_outputs(subs.Options(video=str(video), out=str(tmp_path / "lyrics.en.srt"), lang="uk",
                                        translate=True))


class StubClient:
    created: ClassVar[list[dict]] = []
    outcome = None

    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        StubClient.created.append(kwargs)
        if isinstance(StubClient.outcome, Exception):
            raise StubClient.outcome
        return StubClient.outcome


def message(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


def stub_anthropic(monkeypatch, outcome):
    anthropic = pytest.importorskip("anthropic")
    StubClient.created, StubClient.outcome = [], outcome
    monkeypatch.setattr(anthropic, "Anthropic", StubClient)
    return anthropic


def test_ask_claude_reads_the_text_block_after_a_thinking_block(monkeypatch):
    stub_anthropic(monkeypatch, message("end_turn", SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text='{"lines": []}')))
    assert translate.ask_claude("m", "sys", "user", {"type": "object"}) == {"lines": []}
    sent = StubClient.created[0]
    assert sent["model"] == "m" and sent["output_config"]["effort"] == "low"
    assert sent["output_config"]["format"] == {"type": "json_schema", "schema": {"type": "object"}}


@pytest.mark.parametrize("stop", ["max_tokens", "refusal"])
def test_ask_claude_turns_a_cut_off_or_declined_reply_into_stage_error(monkeypatch, stop):
    stub_anthropic(monkeypatch, message(stop, SimpleNamespace(type="text", text="{")))
    with pytest.raises(StageError, match=stop):
        translate.ask_claude("m", "sys", "user", {})


def test_ask_claude_turns_api_errors_into_stage_error(monkeypatch):
    class Boom(pytest.importorskip("anthropic").APIError):
        def __init__(self):
            Exception.__init__(self, "overloaded")

    stub_anthropic(monkeypatch, Boom())
    with pytest.raises(StageError, match="overloaded"):
        translate.ask_claude("m", "sys", "user", {})


def test_missing_key_fails_before_the_pipeline(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(subs, "separate_vocals", lambda path: pytest.fail("listening started"))
    monkeypatch.setattr(subs.media, "write_atomic", lambda *a: pytest.fail("audio extraction started"))
    monkeypatch.setattr(translate.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(StageError, match="ANTHROPIC_API_KEY"):
        subs.run(subs.Options(video=str(video), translate=True), Config())


def test_missing_package_names_the_extra(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(translate.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(StageError, match=r"\.\[translate\]"):
        translate.check_ready()


def test_subs_translate_plans_three_tracks_and_refuses_any_existing_file(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    paths = subs.check_outputs(subs.Options(video=str(video), translate=True))
    assert paths == {lang: tmp_path / f"v.{lang}.srt" for lang in ("en", "uk", "he")}
    (tmp_path / "v.he.txt").write_text("mine")
    with pytest.raises(StageError, match="v.he.txt"):
        subs.check_outputs(subs.Options(video=str(video), translate=True))
    subs.check_outputs(subs.Options(video=str(video), translate=True, force=True))


def test_subs_translate_needs_a_supported_language(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    with pytest.raises(UsageError):
        subs.check_outputs(subs.Options(video=str(video), translate=True, lang="fr"))


def test_translate_writes_the_other_two_tracks_and_leaves_the_source(tmp_path, monkeypatch):
    source = tmp_path / "song.en.srt"
    original = format_srt(CAPTIONS)
    source.write_text(original)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(translate.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(translate, "ask_claude", lambda model, system, user, schema: reply(2))
    result = translate.run(Options(file=str(source)), Config())
    assert set(result["tracks"]) == {"uk", "he"}
    assert source.read_text() == original
    assert (tmp_path / "song.uk.srt").read_text().splitlines()[1] == "00:00:01,000 --> 00:00:02,000"
    assert (tmp_path / "song.he.txt").read_text() == "he1\nhe2\n"
    with pytest.raises(StageError, match="already exists"):
        translate.run(Options(file=str(source)), Config())
