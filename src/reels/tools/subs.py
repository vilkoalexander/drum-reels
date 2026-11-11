import importlib.util
import logging
import string
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from reels import media
from reels.service import StageError, UsageError
from reels.subtitles import Caption, check_free, clock, write_all
from reels.tools.translate import (
    LANGS,
    check_ready,
    cleaned_lines,
    output_files,
    plan_tracks,
    track_files,
    track_report,
    translate_captions,
)

log = logging.getLogger(__name__)

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
DEMUCS_RATE = 44100
WHISPER_RATE = 16000
ENVELOPE_WINDOW_S = 0.05
# a word is kept when the vocal stem near it reaches this fraction of the stem's loud level (-20 dB)
SILENCE_FRACTION = 0.1
SILENCE_FLOOR_RMS = 1e-3
LOUD_PERCENTILE = 95
MAX_CHARS = 42
PAUSE_S = 0.8
MAX_CAPTION_S = 6.0
MIN_CAPTION_S = 0.3
SHAKY_BELOW = 0.6
JUNK_BELOW = 0.4
MIN_WORD_PROBABILITY = 0.2
MAX_NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0
HALLUCINATION_SILENCE_S = 2.0
PHRASE_END = ",.?!;:"
# Whisper word times run end to start, so a sung phrase end shows as punctuation, a capital or a held word more than a gap
BREATH_GAP_S = 0.4
MAX_GAP_SCORE = 1.5
PUNCTUATION_SCORE = 1.5
CAPITAL_SCORE = 1.0
HELD_WORD_S = 0.8
HELD_WORD_SCORE = 0.8
BREAK_SCORE = 1.0
MIN_LINE_CHARS = 8
SHORT_WORD_CHARS = 3
MIN_PIECE_CHARS = 8
LONG_WORD_CHARS = 5
LONG_WORD_SCORE = 0.3
BALANCE_WEIGHT = 0.5
ORPHAN_CHARS = 4
ORPHAN_JOIN_S = 1.0
# longest silence between two sung lines in the test song is 7 s
ISOLATED_S = 8.0
ISOLATED_MAX_WORDS = 2
LOOP_MAX_DISTINCT = 2
LOOP_MIN_WORDS = 4
LOOP_TOKEN_CHARS = 4
SEED = 0
INSTALL_HINT = "pip install -e '.[subs]'"


@dataclass(frozen=True)
class Options:
    video: str
    out: str | None = None
    lang: str = "en"
    force: bool = False
    translate: bool = False


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    probability: float


def loudness_envelope(vocals: np.ndarray, sample_rate: int) -> np.ndarray:
    hop = int(sample_rate * ENVELOPE_WINDOW_S)
    windows = len(vocals) // hop
    if windows == 0:
        return np.zeros(1)
    return np.sqrt((vocals[: windows * hop].reshape(windows, hop) ** 2).mean(axis=1))


def trim_words_to_sound(words: list[Word], envelope: np.ndarray) -> list[Word]:
    """Cut each word to its loud windows and drop words that sit entirely in silence."""
    threshold = max(float(np.percentile(envelope, LOUD_PERCENTILE)) * SILENCE_FRACTION, SILENCE_FLOOR_RMS)
    trimmed = []
    for word in words:
        first = int(word.start / ENVELOPE_WINDOW_S)
        last = int(word.end / ENVELOPE_WINDOW_S) + 1
        loud = np.flatnonzero(envelope[first:last] > threshold)
        if len(loud):
            start = (first + loud[0]) * ENVELOPE_WINDOW_S
            end = (first + loud[-1] + 1) * ENVELOPE_WINDOW_S
            trimmed.append(Word(word.text, max(start, word.start), min(end, word.end), word.probability))
        else:
            log.debug("drop word %.2f %r: vocal stem quiet (threshold %.4f)", word.start, word.text, threshold)
    return trimmed


def is_short_repeat(tokens: list[str]) -> bool:
    return len(set(tokens)) <= LOOP_MAX_DISTINCT and all(len(t) < LOOP_TOKEN_CHARS for t in tokens)


def is_loop(texts: list[str]) -> bool:
    """One or two short tokens repeated (`Oh Oh Oh Oh`): a recogniser loop, not a sung `Hallelujah` refrain."""
    tokens = [t.strip(string.punctuation).lower() for t in texts]
    return len(tokens) >= LOOP_MIN_WORDS and is_short_repeat(tokens)


def segment_problem(segment: dict) -> str | None:
    if segment["no_speech_prob"] > MAX_NO_SPEECH_PROB:
        return f"no_speech_prob {segment['no_speech_prob']:.2f} > {MAX_NO_SPEECH_PROB}"
    if segment["avg_logprob"] < MIN_AVG_LOGPROB:
        return f"avg_logprob {segment['avg_logprob']:.2f} < {MIN_AVG_LOGPROB}"
    if is_loop([w["word"] for w in segment.get("words", [])]):
        return "one or two short tokens repeated"
    return None


def is_usable_word(word: Word) -> bool:
    return word.probability >= MIN_WORD_PROBABILITY and any(c.isalpha() for c in word.text)


def segment_words(segments: list[dict]) -> list[Word]:
    words = []
    for segment in segments:
        problem = segment_problem(segment)
        if problem:
            log.debug("drop segment %.2f-%.2f %r: %s", segment.get("start", 0), segment.get("end", 0),
                      segment.get("text", "").strip(), problem)
            continue
        for w in segment.get("words", []):
            word = Word(w["word"].strip(), w["start"], w["end"], w["probability"])
            if is_usable_word(word):
                words.append(word)
            else:
                log.debug("drop word %.2f %r: probability %.2f or no letters", word.start, word.text, word.probability)
    return words


def line_text(line: list[Word]) -> str:
    return " ".join(w.text for w in line)


def too_long(line: list[Word]) -> bool:
    return len(line_text(line)) > MAX_CHARS or line[-1].end - line[0].start > MAX_CAPTION_S


def break_score(previous: Word, following: Word) -> float:
    score = min((following.start - previous.end) / BREATH_GAP_S, MAX_GAP_SCORE)
    if previous.text[-1] in PHRASE_END:
        score += PUNCTUATION_SCORE
    elif len(previous.text) > SHORT_WORD_CHARS and following.text[0].isupper():
        score += CAPITAL_SCORE
    if len(previous.text) > SHORT_WORD_CHARS and previous.end - previous.start >= HELD_WORD_S:
        score += HELD_WORD_SCORE
    return score


def best_cut(line: list[Word], incoming: Word) -> int:
    """Where to end a line that cannot take `incoming`: the cut with the clearest phrase boundary."""
    best, best_score = len(line), -1.0
    for cut in range(1, len(line)):
        left, right = line_text(line[:cut]), line_text([*line[cut:], incoming])
        if len(left) < MIN_PIECE_CHARS or len(right) < MIN_PIECE_CHARS:
            continue
        last = line[cut - 1].text
        score = break_score(line[cut - 1], line[cut])
        score += LONG_WORD_SCORE if len(last) >= LONG_WORD_CHARS else -LONG_WORD_SCORE if len(last) <= SHORT_WORD_CHARS else 0
        score -= BALANCE_WEIGHT * abs(len(left) - len(right)) / (len(left) + len(right))
        if score >= best_score:
            best, best_score = cut, score
    return best


def split_run(words: list[Word]) -> list[list[Word]]:
    lines: list[list[Word]] = []
    line: list[Word] = []
    for word in words:
        while line and too_long([*line, word]):
            cut = best_cut(line, word)
            lines.append(line[:cut])
            line = line[cut:]
        if line and len(line_text(line)) >= MIN_LINE_CHARS and break_score(line[-1], word) >= BREAK_SCORE:
            lines.append(line)
            line = []
        line.append(word)
    return [*lines, line] if line else lines


def join_orphans(lines: list[list[Word]]) -> list[list[Word]]:
    joined: list[list[Word]] = []
    carry: list[Word] = []
    for i, line in enumerate(lines):
        merged = [*carry, *line]
        carry = []
        following = lines[i + 1] if i + 1 < len(lines) else None
        if (following and len(merged) == 1 and len(merged[0].text) <= ORPHAN_CHARS
                and following[0].start - merged[0].end <= ORPHAN_JOIN_S and not too_long([*merged, *following])):
            carry = merged
        else:
            joined.append(merged)
    return joined


def group_captions(words: list[Word]) -> list[Caption]:
    ordered = sorted(words, key=lambda w: w.start)
    runs: list[list[Word]] = []
    for word in ordered:
        if runs and word.start - runs[-1][-1].end < PAUSE_S:
            runs[-1].append(word)
        else:
            runs.append([word])
    lines = join_orphans([line for run in runs for line in split_run(run)])
    captions: list[Caption] = []
    previous_end = 0.0
    for line in lines:
        start = max(line[0].start, previous_end)
        end = min(max(line[-1].end, start + MIN_CAPTION_S), start + MAX_CAPTION_S)
        captions.append(Caption(start, end, line_text(line), sum(w.probability for w in line) / len(line)))
        previous_end = end
    return captions


def drop_isolated_captions(captions: list[Caption]) -> list[Caption]:
    kept = []
    for i, caption in enumerate(captions):
        before = caption.start - captions[i - 1].end if i > 0 else float("inf")
        after = captions[i + 1].start - caption.end if i + 1 < len(captions) else float("inf")
        if len(caption.text.split()) <= ISOLATED_MAX_WORDS and min(before, after) >= ISOLATED_S:
            log.debug("drop caption %.2f %r: isolated by %.1f s of silence", caption.start, caption.text,
                      min(before, after))
            continue
        kept.append(caption)
    return kept


def tokens_of(caption: Caption) -> list[str]:
    return [t.strip(string.punctuation).lower() for t in caption.text.split()]


def drop_loops(captions: list[Caption]) -> list[Caption]:
    """Drop runs of nearby captions that only repeat one or two short tokens, like `Oh Oh Oh` or a stream of `I`."""
    dropped: set[int] = set()
    i = 0
    while i < len(captions):
        distinct: set[str] = set()
        total, j = 0, i
        while j < len(captions) and (j == i or captions[j].start - captions[j - 1].end < ISOLATED_S):
            merged = distinct | set(tokens_of(captions[j]))
            if not is_short_repeat(list(merged)):
                break
            distinct, total, j = merged, total + len(tokens_of(captions[j])), j + 1
        if total >= LOOP_MIN_WORDS:
            dropped.update(range(i, j))
        i = max(j, i + 1)
    for k in sorted(dropped):
        log.debug("drop caption %.2f %r: one or two tokens repeated", captions[k].start, captions[k].text)
    return [c for k, c in enumerate(captions) if k not in dropped]


def drop_junk_captions(captions: list[Caption]) -> list[Caption]:
    kept = []
    for c in captions:
        if c.probability >= JUNK_BELOW and any(ch.isalpha() for ch in c.text):
            kept.append(c)
        else:
            log.debug("drop caption %.2f %r: mean probability %.2f or no letters", c.start, c.text, c.probability)
    return kept


def shaky_captions(captions: list[Caption]) -> list[dict]:
    return [{"at": clock(c.start), "text": c.text} for c in captions if c.probability < SHAKY_BELOW]


def separate_vocals(audio_path: Path) -> np.ndarray:
    try:
        import torch
        from demucs.api import Separator
    except ImportError as error:
        raise StageError(f"{error.name} is not installed: {INSTALL_HINT}") from error
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("separating vocals on %s", device)
    try:
        _, stems = Separator(model="htdemucs", device=device, shifts=0).separate_audio_file(audio_path)
    except RuntimeError as error:
        raise StageError(f"vocal separation failed: {error}") from error
    mono = stems["vocals"].mean(dim=0).cpu().numpy().astype(np.float32)
    return librosa.resample(mono, orig_sr=DEMUCS_RATE, target_sr=WHISPER_RATE)


def transcribe_words(vocals: np.ndarray, lang: str) -> list[Word]:
    try:
        import mlx.core
        import mlx_whisper
    except ImportError as error:
        raise StageError(f"{error.name} is not installed: {INSTALL_HINT}") from error
    log.info("transcribing")
    mlx.core.random.seed(SEED)
    try:
        result = mlx_whisper.transcribe(vocals, path_or_hf_repo=WHISPER_MODEL, language=lang, word_timestamps=True,
                                        condition_on_previous_text=False,
                                        hallucination_silence_threshold=HALLUCINATION_SILENCE_S)
    except RuntimeError as error:
        raise StageError(f"transcription failed: {error}") from error
    return segment_words(result["segments"])


def whisper_languages() -> set[str]:
    from mlx_whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

    return set(LANGUAGES) | set(TO_LANGUAGE_CODE)


def check_listening_ready(lang: str) -> None:
    for name in ("torch", "demucs", "mlx_whisper"):
        if importlib.util.find_spec(name) is None:
            raise StageError(f"{name} is not installed: {INSTALL_HINT}")
    if lang not in whisper_languages():
        raise UsageError(f"--lang {lang!r} is not a language Whisper knows")


def check_outputs(opts: Options) -> dict[str, Path]:
    src = Path(opts.video)
    if not src.is_file():
        raise StageError(f"{src} not found")
    srt_path = Path(opts.out) if opts.out else src.with_name(f"{src.stem}.{opts.lang}.srt")
    if srt_path.suffix.lower() != ".srt":
        raise StageError(f"--out must end in .srt, got {srt_path.name}")
    if opts.translate:
        if opts.lang not in LANGS:
            raise UsageError(f"--translate needs --lang to be one of {', '.join(LANGS)}")
        paths = plan_tracks(srt_path, opts.lang)
    else:
        paths = {opts.lang: srt_path}
    check_free(output_files(list(paths.values())), opts.force)
    return paths


def run(opts: Options, cfg) -> dict:
    paths = check_outputs(opts)
    check_listening_ready(opts.lang)
    if opts.translate:
        check_ready()
    src = Path(opts.video)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        media.write_atomic(["-i", str(src)], ["-vn", "-ac", "2", "-ar", str(DEMUCS_RATE), "-c:a", "pcm_s16le"], wav)
        vocals = separate_vocals(wav)
    words = trim_words_to_sound(transcribe_words(vocals, opts.lang), loudness_envelope(vocals, WHISPER_RATE))
    captions = drop_isolated_captions(drop_loops(drop_junk_captions(group_captions(words))))
    if not captions:
        raise StageError(f"no sung words found in {src.name}")
    shaky = shaky_captions(captions)
    for line in shaky:
        log.warning("check %s  %s", line["at"], line["text"])
    cleaned = []
    tracks = {opts.lang: captions}
    if opts.translate:
        tracks = translate_captions(captions, opts.lang, True, cfg.translate.model)
        cleaned = cleaned_lines(captions, tracks[opts.lang])
        for line in cleaned:
            log.warning("cleaned %s  %s -> %s", line["at"], line["was"], line["now"])
    write_all(track_files(tracks, paths))
    log.info("%d captions written for %s", len(captions), ", ".join(tracks))
    return {"ok": True, "tracks": track_report(paths), "captions": len(captions), "shaky": shaky, "cleaned": cleaned}
