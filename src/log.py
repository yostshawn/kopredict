"""Copy stdout and stderr to a log file while still printing to the terminal."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd


class Tee:
    """Write each stream write to the terminal and a log file."""

    def __init__(self, stream, handle):
        self.stream = stream
        self.handle = handle

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.handle.write(data)
        self.handle.flush()
        return len(data)

    def flush(self) -> None:
        self.stream.flush()
        self.handle.flush()


@contextmanager
def log_to_file(path: Path) -> Iterator[Path]:
    """Send stdout and stderr to the terminal and overwrite ``path``.

    A SystemExit message is written once into the log, then raised again as
    SystemExit(1) so the process still fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    previous_stdout, previous_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(previous_stdout, handle)
    sys.stderr = Tee(previous_stderr, handle)
    try:
        print(f"Log file: {path}")
        yield path
    except SystemExit as exc:
        if str(exc):
            print(exc, file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        sys.stdout = previous_stdout
        sys.stderr = previous_stderr
        handle.close()


def shape(frame: pd.DataFrame | None) -> str:
    """Return a one-line row and column count for a log message."""
    if frame is None or frame.empty:
        return "empty"
    return f"{frame.shape[0]} rows x {frame.shape[1]} columns"
