import logging
import sys

from reels.service import Service

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_installed: list[logging.Handler] = []


def _install(handlers: list[logging.Handler], verbose: bool) -> None:
    root = logging.getLogger()
    for handler in _installed:
        root.removeHandler(handler)
        handler.close()
    for handler in handlers:
        handler.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(handler)
    _installed[:] = handlers
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


def setup(service: Service, stage: str, verbose: bool) -> None:
    service.work.mkdir(parents=True, exist_ok=True)
    _install([logging.StreamHandler(sys.stderr), logging.FileHandler(service.work / f"{stage}.log", mode="a")], verbose)


def setup_console(verbose: bool) -> None:
    _install([logging.StreamHandler(sys.stderr)], verbose)
