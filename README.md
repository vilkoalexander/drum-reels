# reels

reels takes one YouTube livestream and the camera recordings made at the same event.
It aligns each camera to the stream by its audio, then cuts the moments listed in
notes/moments.txt from every source that covers them. The result is out/timeline.fcpxml,
an FCPXML timeline that DaVinci Resolve imports. Each stage writes into one service
folder and skips work that is already up to date.

## Install

```
python3.13 -m venv .venv
.venv/bin/pip install -e '.[dev]'
brew install ffmpeg
```

## Agent skill

skills/drum-reels/SKILL.md teaches Claude Code (or any agent that reads skills) to drive this CLI.

```
/plugin install drum-reels --marketplace vilkoalexander/drum-reels   # Claude Code
npx skills add vilkoalexander/drum-reels                             # other agents
```

## Commands

```
reels fetch  <youtube-url> --out <service> [--force]  download the livestream
reels sync   <service> [--force]               align the camera recordings to it
reels cut    <service> [--notes FILE] [--force] [--dry-run]  cut the moments from the synced sources
reels render <service> --mode fcpxml [--force] [--dry-run]  write out/timeline.fcpxml
reels looks  <service> [slug] [--force] [--samples N]  find people, write cuts/<slug>/looks.toml
reels reel   <service> <slug> [--pattern NAME] [--skip-bars N] [--bars N]
             [--title T] [--caption C] [--look-text LOOK=TEXT]  9:16 reel via Remotion
reels run    <service> [--url URL] [--notes FILE] [--force]  run the stages in order

reels doctor [folder]                           check ffmpeg, yt-dlp, Resolve and disk space
reels synth  <folder> [--duration S] [--style drums|bursts]  write a synthetic service with truth.json
reels eval   <service> [--max-ms 5] [--max-ppm 10]  compare work/sync.json with truth.json
reels short  <video> [--seconds S] [--start S] [--out DIR] [--name N] [--grade G] [--title T] [--caption C]
             [--subs FILE.srt]                  9:16 short of the busiest bars, subtitles burned in with --subs
reels subs   <video> [--out FILE.srt] [--lang en] [--translate] [--force]  write <video>.<lang>.srt and .txt lyrics from the sung vocals
reels translate <file.srt> [--from LANG] [--force]  write the other two of en, uk, he beside it
```

Every command prints exactly one JSON line on stdout, `{"ok": true, "command": ..., ...}` with what
it wrote, or `{"ok": false, "error": ...}`. Progress goes to stderr and work/<stage>.log. Exit codes:
0 ok, 1 failed or partly failed, 2 bad arguments. Bad options fail before any file is written.

`--dry-run` plans the cut windows and prints the ffmpeg commands, or builds the timeline
and reports it, without writing anything.

`reels looks` decides how each moment can be framed, since cameras stand somewhere else every time.
It runs YOLO on a few frames of the moment: a 360 source gets one view per person seen in most frames
plus a `wide` view, a flat camera gets the rotation under which people stand upright. It writes
cuts/<slug>/looks.toml and cuts/<slug>/looks.jpg, the 360 frame with a degree scale and every view
marked. Rename the looks, fix a yaw from the picture, edit the pattern; `--force` overwrites your edits.
Needs `pip install -e '.[people]'` and models/yolo11n.pt.

`reels subs` listens to a music video and writes <video>.<lang>.srt and <video>.<lang>.txt beside it, ready to upload
to YouTube. Demucs isolates the vocals, mlx-whisper transcribes them, words over instrumental stretches
are dropped, and low-confidence lines are listed under `shaky` and logged with their MM:SS time so you can
fix them by hand. Captions Whisper itself doubts are dropped. It refuses to overwrite an existing .srt or
.txt without `--force`, since people edit them. It does not use a service folder. Needs `pip install -e '.[subs]'`.

`reels subs --translate` and `reels translate` write the three tracks en, uk and he with the Claude API
(one request per song, model from `[translate]` in reels.toml, key in `ANTHROPIC_API_KEY`). Timings are copied
from the source track, so all tracks switch lines together. `subs --translate` also fixes foreign words the
recogniser mangled (listed under `cleaned`); `translate` never touches your .srt. Needs
`pip install -e '.[translate]'`.

`reels reel` turns one cuts/<slug>/ folder into out/reels/<name>.mp4: it renders each look in
looks.toml into work/reels/<name>/, then Remotion in remotion/ cuts between them every two bars,
in the order of the chosen pattern, over the stream audio and adds the text. `master` (the stream
cropped to 9:16) always exists. Run `npm install` in remotion/ once.

## How it works

```
 fetch            sync                   cut                    render / reel
 ─────            ────                   ───                    ─────────────
 yt-dlp ──► sources/youtube.mp4 (master audio + timeline)
            sources/dji*.mp4 ─┐
            sources/x5*.mp4  ─┤ join per camera
                              ▼
                   audio xcorr vs master
                   cam_t = intercept + slope·yt_t
                   ──► work/sync.json
                              │
 notes/moments.txt ───────────┤ "1:12:30 fill"
                              ▼
                   beat grid, snap to bars,
                   cut every camera over the window
                   ──► cuts/<slug>/{youtube,dji,x5}.mov + moment.json
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
   render: out/timeline.fcpxml       looks: YOLO on a few frames
           (Resolve)                        360: a view per person + wide
                                            flat: rotation people stand up in
                                     ──► cuts/<slug>/looks.toml + looks.jpg
                                                │  (edit by hand if needed)
                                                ▼
                                     reel: per-look 9:16 clips (ffmpeg)
                                           flat: rotate + crop
                                           360: v360 virtual camera
                                           + grade, audio.wav, props.json
                                                │
                                                ▼
                                     Remotion: cut every 2 bars,
                                     zoom moves, title/caption/stickers/subtitles
                                     ──► out/reels/<name>.mp4

 short: any edited video ──► beat grid ──► busiest N bars ──► same Remotion path
```

## Service folder

```
services/<date>/
  sources/  youtube.mp4 youtube.info.json dji*.mp4 x5*.mp4
  notes/    moments.txt
  work/     audio/ dji.joined.mp4 sync.json beats.json <stage>.log
  cuts/     <slug>/ youtube.mov dji.mov x5.mov moment.json
  out/      timeline.fcpxml
```

Camera files are matched case-insensitively and sorted by name. Several files for one
camera are joined into work/<camera>.joined.mp4 before use.

To set a camera's sync by hand, edit that camera's entry in work/sync.json and set
`"manual": true`. Sync keeps the offset and drift values in that entry and only
refreshes the file details. The stage then skips the audio match for that camera.
After a hand edit run `reels cut` again: render refuses moments that were cut against an
older sync.json. A note's length may not be shorter than `cut.min_len_s` (5 s), so a label
ending in a number is caught instead of becoming a short cut.

Each stage appends its log to work/<stage>.log, for example work/sync.log.

## Code layout

```
src/reels/
  cli.py config.py service.py log.py
  stages/    fetch.py sync.py cut.py render.py   the commands; the only code that reads or writes a service folder
  media/     ffmpeg.py audio.py cache.py         everything that calls ffmpeg, ffprobe or soundfile
  analysis/  clock.py beats.py notes.py          pure maths and parsing, no files, no subprocess
  timeline/  fcpxml.py                           the Resolve/FCP output
  tools/     doctor.py synth.py eval.py          not part of a run
tests/       mirrors src/reels
```

## Tests

```
.venv/bin/python -m pytest -q
```
