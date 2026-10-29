"""Helpers for running experiment scripts from a notebook (Colab, Kaggle, Jupyter).

``!python run.py ... | grep ^RESULT`` hides a crash completely: grep prints nothing and the
cell looks fine. ``run`` streams the interesting stdout lines (RESULT, epoch logs, config
banners) while the script runs, keeps the rest, and on a non-zero exit prints the tail of
stdout + stderr and raises, so the cell stops instead of silently moving on.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

SHOW_PREFIXES = ("RESULT", "epoch", "[", "KD teacher", "classes=", "train=", "base=", "core thresholds", "wrote")


def run(cmd: str, show: Iterable[str] = SHOW_PREFIXES, tail: int = 30, cwd: Optional[str] = None,
        check: bool = True) -> str:
    """Run ``cmd`` (a shell-style string), echo selected stdout lines live, fail loudly.

    Returns the full stdout. Raises ``RuntimeError`` on a non-zero exit when ``check`` is set.
    """
    show = tuple(show)
    print(f"$ {cmd}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}   # otherwise a piped python child flushes only at exit
    proc = subprocess.Popen(shlex.split(cmd), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    out_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        out_lines.append(line)
        if line.startswith(show):
            print(line, end="", flush=True)
    err = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        log = "".join(out_lines) + err
        sys.stdout.write("\n".join(log.splitlines()[-tail:]) + "\n")
        sys.stdout.flush()
        if check:
            raise RuntimeError(f"command failed with exit code {rc}: {cmd[:160]}")
    return "".join(out_lines)


def done(name: str, out: str = "results", ckpt: bool = False) -> bool:
    """True when ``results/<name>.json`` (or ``results/<name>`` if ``name`` has a suffix) already exists,
    and, with ``ckpt``, the ``.pt`` checkpoint next to it (a base that later runs will load)."""
    base = Path(out) / name
    target = base if base.suffix else base.with_suffix(".json")
    if not target.exists():
        return False
    return (not ckpt) or base.with_suffix(".pt").exists()


def run_once(cmd: str, name: str, ckpt: bool = False, out: str = "results", **kw) -> Optional[str]:
    """``run`` unless the run named ``name`` is already in ``out`` (resume a session after a disconnect)."""
    if done(name, out, ckpt):
        print(f"skip {name}: already in {out}/", flush=True)
        return None
    return run(cmd, **kw)
