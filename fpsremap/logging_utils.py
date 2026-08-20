"""
Printing log files per model.
"""

from __future__ import annotations

import sys
import time

_PREFIX = ""
_T0 = time.perf_counter()


def set_prefix(prefix: str) -> None:
    global _PREFIX
    _PREFIX = f"[{prefix}] " if prefix else ""


def get_prefix() -> str:
    return _PREFIX


def log(msg: str = "") -> None:
    
    if msg:
        print(f"{_PREFIX}{msg}", flush=True)
    else:
        print("", flush=True)


def log_header(title: str, char: str = "=", width: int = 60) -> None:
    log("")
    log(char * width)
    log(title)
    log(char * width)


def elapsed_min(t_start: float) -> float:
    """Minutes elapsed since a time.perf_counter() reading."""
    return (time.perf_counter() - t_start) / 60.0


def die(msg: str, code: int = 1):
    """Print an error to stderr and exit."""
    print(f"{_PREFIX}ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)
