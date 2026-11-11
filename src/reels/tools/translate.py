import importlib.util
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from reels.service import StageError, UsageError
from reels.subtitles import Caption, check_free, clock, format_srt, format_txt, parse_srt, write_all

log = logging.getLogger(__name__)

LANGS = {"en": "English", "uk": "Ukrainian", "he": "Hebrew"}
INSTALL_HINT = "pip install -e '.[translate]'"
MAX_TOKENS = 16000

SYSTEM_PROMPT = """You prepare subtitles for one song. You get its numbered caption lines, in order, as JSON.
Return every line with the same index: one output line per input line, none merged, split, added or dropped.

The song is in {source}. Each returned line has "index", "source" (the song's own line) and {targets}.

Translate by meaning, using familiar church wording for Bible phrases. Do not make singable verse. Keep each line short enough to read as a subtitle. Use the whole song as context.

Words sung in another language than the track's language (typically Hebrew words inside an English song) follow this convention:
- English track: Latin transliteration, with the meaning in brackets the first time it appears, for example: Baruch haba (Blessed is he who comes)
- Ukrainian track: Cyrillic transliteration, with the same bracket rule.
- Hebrew track: Hebrew script.
- Words everyone knows (Hallelujah, Hosanna, Amen, Shalom) stay as they are, with no brackets.

{source_rule}"""

CLEAN_RULE = """The source lines come from a speech recogniser. In "source" you may only fix foreign-language words it mangled and apply the convention above to them. Do not rewrite, improve, complete or correct the lyrics in any other way. A line that needs no change is returned exactly as given."""
KEEP_RULE = """The source lines are final. Return "source" exactly as given."""

Ask = Callable[[str, str, dict], dict]


@dataclass(frozen=True)
class Options:
    file: str
    source_lang: str | None = None
    force: bool = False


def lang_of_name(path: Path, given: str | None) -> str:
    if given is None:
        match = re.search(r"\.([a-z]{2})$", path.stem)
        given = match.group(1) if match else None
    if given is None:
        raise UsageError(f"cannot tell the language of {path.name}: name it <name>.<lang>.srt or pass --from")
    if given not in LANGS:
        raise UsageError(f"language {given!r} is not supported, use one of {', '.join(LANGS)}")
    return given


def plan_tracks(source: Path, source_lang: str) -> dict[str, Path]:
    named = re.search(r"\.([a-z]{2})$", source.stem)
    if named and named.group(1) != source_lang:
        raise UsageError(f"{source.name} is named as {named.group(1)} but the language is {source_lang}")
    base = source.stem.removesuffix(f".{source_lang}")
    return {lang: source if lang == source_lang else source.with_name(f"{base}.{lang}.srt") for lang in LANGS}


def output_files(srt_paths: list[Path]) -> list[Path]:
    return [p for srt in srt_paths for p in (srt, srt.with_suffix(".txt"))]


def check_ready() -> None:
    if importlib.util.find_spec("anthropic") is None:
        raise StageError(f"anthropic is not installed: {INSTALL_HINT}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise StageError("ANTHROPIC_API_KEY is not set: export your Claude API key before translating")


def reply_schema(targets: list[str]) -> dict:
    fields = {"index": {"type": "integer"}, "source": {"type": "string"}, **{t: {"type": "string"} for t in targets}}
    line = {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}
    return {"type": "object", "properties": {"lines": {"type": "array", "items": line}}, "required": ["lines"],
            "additionalProperties": False}


def ask_claude(model: str, system: str, user: str, schema: dict) -> dict:
    try:
        import anthropic
    except ImportError as error:
        raise StageError(f"anthropic is not installed: {INSTALL_HINT}") from error
    try:
        response = anthropic.Anthropic().messages.create(
            model=model, max_tokens=MAX_TOKENS, system=system, messages=[{"role": "user", "content": user}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
    except anthropic.APIError as error:
        raise StageError(f"Claude request failed: {error}") from error
    if response.stop_reason in ("max_tokens", "refusal"):
        raise StageError(f"Claude stopped with {response.stop_reason}; the song may be too long or was declined")
    try:
        return json.loads(next(b.text for b in response.content if b.type == "text"))
    except (StopIteration, ValueError) as error:
        raise StageError("Claude returned no readable JSON") from error


def matches_input(reply: dict, count: int) -> bool:
    return [line.get("index") for line in reply.get("lines", [])] == list(range(1, count + 1))


def request_lines(texts: list[str], source_lang: str, clean: bool, ask: Ask) -> list[dict]:
    targets = [lang for lang in LANGS if lang != source_lang]
    system = SYSTEM_PROMPT.format(source=LANGS[source_lang], targets=", ".join(f'"{t}" ({LANGS[t]})' for t in targets),
                                  source_rule=CLEAN_RULE if clean else KEEP_RULE)
    user = json.dumps({"lines": [{"index": i, "text": t} for i, t in enumerate(texts, 1)]}, ensure_ascii=False)
    schema = reply_schema(targets)
    reply = ask(system, user, schema)
    if not matches_input(reply, len(texts)):
        log.warning("Claude returned the wrong lines, asking once more")
        retry = f"{user}\n\nYour last answer did not hold exactly {len(texts)} lines with indices 1 to {len(texts)}."
        reply = ask(system, retry, schema)
        if not matches_input(reply, len(texts)):
            raise StageError(f"Claude did not return the {len(texts)} lines asked for, twice")
    return reply["lines"]


def merge_tracks(captions: list[Caption], lines: list[dict], source_lang: str, clean: bool) -> dict[str, list[Caption]]:
    tracks = {}
    for lang in LANGS:
        if lang == source_lang:
            texts = [line["source"] if clean else c.text for line, c in zip(lines, captions)]
        else:
            texts = [line[lang] for line in lines]
        tracks[lang] = [replace(c, text=text.strip()) for c, text in zip(captions, texts)]
    return tracks


def cleaned_lines(before: list[Caption], after: list[Caption]) -> list[dict]:
    return [{"at": clock(b.start), "was": b.text, "now": a.text} for b, a in zip(before, after) if b.text != a.text]


def translate_captions(captions: list[Caption], source_lang: str, clean: bool, model: str,
                       ask: Ask | None = None) -> dict[str, list[Caption]]:
    ask = ask or partial(ask_claude, model)
    lines = request_lines([c.text for c in captions], source_lang, clean, ask)
    return merge_tracks(captions, lines, source_lang, clean)


def track_files(tracks: dict[str, list[Caption]], paths: dict[str, Path]) -> dict[Path, str]:
    files = {}
    for lang, captions in tracks.items():
        files[paths[lang]] = format_srt(captions)
        files[paths[lang].with_suffix(".txt")] = format_txt(captions)
    return files


def track_report(paths: dict[str, Path]) -> dict:
    return {lang: {"srt": str(p), "txt": str(p.with_suffix(".txt"))} for lang, p in paths.items()}


def run(opts: Options, cfg) -> dict:
    source = Path(opts.file)
    if not source.is_file():
        raise StageError(f"{source} not found")
    source_lang = lang_of_name(source, opts.source_lang)
    paths = plan_tracks(source, source_lang)
    targets = {lang: p for lang, p in paths.items() if lang != source_lang}
    check_free(output_files(list(targets.values())), opts.force)
    check_ready()
    captions = parse_srt(source.read_text(encoding="utf-8-sig"))
    tracks = translate_captions(captions, source_lang, False, cfg.translate.model)
    write_all(track_files({lang: tracks[lang] for lang in targets}, targets))
    log.info("%d captions translated into %s", len(captions), ", ".join(targets))
    return {"ok": True, "tracks": track_report(targets), "captions": len(captions), "shaky": [], "cleaned": []}
