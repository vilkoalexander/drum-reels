---
name: drum-reels
description: Turns a YouTube livestream plus multi-camera recordings (DJI action cam, Insta360 360°) of a live music event into an audio-synced DaVinci Resolve timeline and 9:16 Instagram/Shorts reels, using the `reels` CLI. Use when the user wants to sync camera footage to a livestream by audio, cut noted moments from a service or gig, reframe 360 footage, make vertical reels or shorts of drumming or band playing, or mentions reels.toml, moments.txt, sync.json or looks.toml.
---

# drum-reels

Drives the `reels` CLI from https://github.com/vilkoalexander/drum-reels.

## Setup (once)

```
git clone https://github.com/vilkoalexander/drum-reels && cd drum-reels
python3.13 -m venv .venv && .venv/bin/pip install -e '.[dev,people]'
brew install ffmpeg
(cd remotion && npm install)
.venv/bin/reels doctor
```

`reels looks` also needs `models/yolo11n.pt`. Config lives in `reels.toml` at the repo root.

## Contract

Every command prints one JSON line on stdout: `{"ok": true, "command": ..., ...}` or
`{"ok": false, "error": ...}`. Parse that, not stderr. Exit 0 ok, 1 failed or partly failed,
2 bad arguments. Progress goes to stderr and `work/<stage>.log`. Stages skip up-to-date work;
`--force` redoes it.

## Full service

1. Make `services/<date>/sources/` and drop camera files in (`dji*.mp4`, `x5*.mp4`;
   360 files must already be exported to equirectangular mp4).
2. Write `services/<date>/notes/moments.txt`, one moment per line in livestream time:
   ```
   1:12:30 fill
   1:20:10 guitar solo
   1:33:00 groove 60
   ```
   `H:MM:SS label [seconds]`. Default length 30 s, minimum 5 s.
3. `reels run services/<date> --url <youtube-url>` runs fetch, sync, cut, render.
4. Import `out/timeline.fcpxml` into Resolve, or go on to reels.

Single stages: `reels fetch <url> --out <service>`, `reels sync`, `reels cut [--dry-run]`,
`reels render <service> --mode fcpxml`.

## Reels from one moment

```
reels looks <service> <slug>        # writes cuts/<slug>/looks.toml + looks.jpg
reels reel  <service> <slug> --title "..." --caption "..."
```

Show the user `looks.jpg` before rendering: it marks every 360 view on a degree scale.
They may rename looks, fix a yaw or reorder the pattern in `looks.toml`; `looks --force`
overwrites those edits, so don't rerun it after hand edits. Output: `out/reels/<name>.mp4`.

## Short from any edited video

```
reels short <video> --seconds 20 --grade film --title "..."
```

Picks the busiest bars by beat grid. `--start S` overrides the pick. Grades: film, punchy,
mono, none. Log footage (DJI D-Log) needs the vendor LUT applied first.

## When sync looks wrong

Check `work/sync.json` confidence per camera. To fix one camera by hand, edit its entry
and set `"manual": true`, then rerun `reels cut`; render refuses cuts made against an
older sync.json. `reels synth` + `reels eval` test sync on generated data.
