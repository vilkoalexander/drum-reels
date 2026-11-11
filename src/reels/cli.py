import argparse
import dataclasses
import importlib
import json
import logging
from pathlib import Path

from reels import config, log
from reels.service import Service, StageError, UsageError

STAGES = ("fetch", "sync", "cut", "render")
TOOLS = ("doctor", "synth", "eval", "short", "subs", "translate")

logger = logging.getLogger(__name__)


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise UsageError(f"{self.prog}: {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = Parser(prog="reels")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str) -> argparse.ArgumentParser:
        # absent options stay out of the namespace so each tool's Options defaults apply
        return commands.add_parser(name, help=help, argument_default=argparse.SUPPRESS)

    fetch = command("fetch", "download the YouTube livestream")
    fetch.add_argument("url")
    fetch.add_argument("--out", dest="service", required=True)
    fetch.add_argument("--force", action="store_true")

    sync = command("sync", "align camera recordings to the livestream")
    sync.add_argument("service")
    sync.add_argument("--force", action="store_true")

    cut = command("cut", "cut moments from the synced sources")
    cut.add_argument("service")
    cut.add_argument("--notes", type=Path)
    cut.add_argument("--force", action="store_true")
    cut.add_argument("--dry-run", action="store_true", help="plan the windows and print the ffmpeg commands")

    render = command("render", "render the timeline")
    render.add_argument("service")
    render.add_argument("--mode", choices=["fcpxml"], required=True)
    render.add_argument("--force", action="store_true")
    render.add_argument("--dry-run", action="store_true", help="build the timeline and report it without writing")

    looks = command("looks", "find the people in each moment and write cuts/<slug>/looks.toml")
    looks.add_argument("service")
    looks.add_argument("slug", nargs="?", help="one folder under cuts/, default all")
    looks.add_argument("--force", action="store_true", help="rewrite an existing looks.toml")
    looks.add_argument("--samples", type=int, help="frames to look at per moment, default 5")

    reel = command("reel", "render a 9:16 reel from one cut moment with Remotion")
    reel.add_argument("service")
    reel.add_argument("slug", help="folder name under cuts/")
    reel.add_argument("--pattern", help="pattern name from cuts/<slug>/looks.toml, default \"default\"")
    reel.add_argument("--skip-bars", type=int, help="start this many bars into the moment")
    reel.add_argument("--bars", type=int, help="reel length in bars, 0 for the whole moment")
    reel.add_argument("--grade", help="colour look: film (default), punchy, mono or none")
    reel.add_argument("--title")
    reel.add_argument("--caption")
    reel.add_argument("--look-text", action="append", metavar="LOOK=TEXT",
                      help="sticker shown whenever that look is on screen, e.g. person2='...'")
    reel.add_argument("--name", help="output name under out/reels/, default <slug>_<pattern>")

    run = command("run", "fetch (with --url), sync, cut, render")
    run.add_argument("service")
    run.add_argument("--notes", type=Path)
    run.add_argument("--url")
    run.add_argument("--force", action="store_true")

    doctor = command("doctor", "check ffmpeg, yt-dlp, Resolve and disk space")
    doctor.add_argument("folder", nargs="?")

    synth = command("synth", "write a synthetic service with known sync truth")
    synth.add_argument("folder")
    synth.add_argument("--duration", type=float)
    synth.add_argument("--style", choices=["drums", "bursts"])

    evaluate = command("eval", "compare work/sync.json with truth.json")
    evaluate.add_argument("service")
    evaluate.add_argument("--truth", type=Path)
    evaluate.add_argument("--max-ms", type=float)
    evaluate.add_argument("--max-ppm", type=float)

    short = command("short", "cut an edited video down to its busiest bars as a 9:16 reel")
    short.add_argument("video")
    short.add_argument("--seconds", type=float, help="target length, default 20")
    short.add_argument("--start", type=float, help="start near this second instead of the busiest bars")
    short.add_argument("--out", help="folder for the result, default shorts/ next to the video")
    short.add_argument("--name")
    short.add_argument("--grade", help="colour look: film, punchy, mono or none (default)")
    short.add_argument("--title")
    short.add_argument("--caption")
    short.add_argument("--subs", metavar="FILE.srt", help="burn these subtitle lines into the short")

    subs = command("subs", "listen to a music video and write .srt subtitles and .txt lyrics")
    subs.add_argument("video")
    subs.add_argument("--out", help="path of the .srt (must end in .srt), default <video>.<lang>.srt; the .txt goes beside it")
    subs.add_argument("--lang", help="sung language, default en")
    subs.add_argument("--force", action="store_true", help="overwrite existing .srt and .txt files")
    subs.add_argument("--translate", action="store_true", help="also write the other two of en, uk, he with Claude")

    translate = command("translate", "translate an .srt into the other two of en, uk, he with Claude")
    translate.add_argument("file")
    translate.add_argument("--from", dest="source_lang", help="language of the file, default from <name>.<lang>.srt")
    translate.add_argument("--force", action="store_true", help="overwrite existing tracks")
    return parser


def run_stage(name: str, args: argparse.Namespace, cfg: config.Config) -> dict:
    package = "tools" if name in TOOLS else "stages"
    module = importlib.import_module(f"reels.{package}.{name}")
    given = vars(args)
    fields = dataclasses.fields(module.Options)
    return module.run(module.Options(**{f.name: given[f.name] for f in fields if f.name in given}), cfg)


def run_all(args: argparse.Namespace, cfg: config.Config) -> dict:
    stages = {}
    for name in STAGES:
        if name == "fetch" and "url" not in vars(args):
            continue
        log.setup(Service(args.service), name, args.verbose)
        stages[name] = run_stage(name, args, cfg)
        if not stages[name]["ok"]:
            return {"ok": False, "stages": stages}
    return {"ok": True, "stages": stages}


def _emit(command: str | None, result: dict) -> None:
    print(json.dumps({"ok": result["ok"], "command": command, **result}, default=str))


def main(argv: list[str] | None = None) -> int:
    command = None
    try:
        args = build_parser().parse_args(argv)
        command = args.command
        cfg = config.load(args.config)
        if command == "run":
            result = run_all(args, cfg)
        else:
            if command in TOOLS:
                log.setup_console(args.verbose)
            else:
                log.setup(Service(args.service), command, args.verbose)
            result = run_stage(command, args, cfg)
    except SystemExit as exit_:
        return int(exit_.code or 0)
    except UsageError as error:
        logger.error("%s", error)
        _emit(command, {"ok": False, "error": str(error), "usage": True})
        return 2
    except (StageError, ValueError, OSError) as error:
        logger.error("%s", error)
        _emit(command, {"ok": False, "error": str(error)})
        return 1
    _emit(command, result)
    return 0 if result["ok"] else 1

