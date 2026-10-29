"""Push experiment artifacts (JSON / CSV / Markdown / PNG, plus checkpoints up to a size cap) to a
separate git repository so that a dying Colab / Kaggle session loses at most one run and the next
session resumes without retraining the bases.

Usage from a notebook or script::

    from bitladder.results_sync import ResultsSync
    sync = ResultsSync("Vzzzz/bitladder-results", token, label="colab-2026-10-27")
    ...                       # after every run:
    sync.push("results")      # copies new/changed files under <label>/ and pushes

The token is used only in the clone URL and is never printed. In Colab take it from
``google.colab.userdata.get("RESULTS_TOKEN")``; locally from the ``RESULTS_TOKEN``
environment variable. The repo is cloned once into ``work_dir`` (default ``/content``
on Colab, else ``~/.cache/bitladder``).

CLI (local): python -m bitladder.results_sync --repo Vzzzz/bitladder-results --label local-rtx3080 results
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

ARTIFACT_SUFFIXES = (".json", ".csv", ".md", ".png", ".txt")
CKPT_SUFFIX = ".pt"
CKPT_MAX_BYTES = 2_000_000   # our KWS checkpoints are 0.1-0.5 MB; anything bigger stays local


def _wanted(f: Path, suffixes: Iterable[str], ckpt_max_bytes: int) -> bool:
    if f.suffix in tuple(suffixes):
        return True
    return f.suffix == CKPT_SUFFIX and f.stat().st_size <= ckpt_max_bytes


class SyncError(RuntimeError):
    """A git step failed; the message never contains the token."""


def _run(args: list, cwd: Path, quiet: bool = True, mask: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run git; on failure raise SyncError with the command and stderr, the token replaced by ``<token>``."""
    try:
        return subprocess.run(args, cwd=str(cwd), check=True, capture_output=quiet, text=True)
    except subprocess.CalledProcessError as e:
        hide = (lambda t: t.replace(mask, "<token>")) if mask else (lambda t: t)
        cmd = " ".join(hide(str(a)) for a in args)
        err = hide(e.stderr or "").strip()[-600:]
        raise SyncError(f"`{cmd}` failed (exit {e.returncode}): {err}") from None


class ResultsSync:
    def __init__(self, repo: str, token: Optional[str], label: str, work_dir: Optional[str] = None,
                 user: str = "bitladder-sync", email: str = "sync@bitladder.local"):
        if not token:
            raise ValueError("no token: set RESULTS_TOKEN (Colab secret or environment variable)")
        self.repo = repo
        self.label = label
        base = Path(work_dir) if work_dir else (Path("/content") if Path("/content").is_dir()
                                                 else Path.home() / ".cache" / "bitladder")
        base.mkdir(parents=True, exist_ok=True)
        self.dir = base / repo.split("/")[-1]
        self._token = token
        self._url = f"https://x-access-token:{token}@github.com/{repo}.git"
        if not (self.dir / ".git").is_dir():
            self._git(["clone", "-q", self._url, str(self.dir)], base)
        self._git(["config", "user.name", user])
        self._git(["config", "user.email", email])
        self._git(["remote", "set-url", "origin", self._url])

    def _git(self, args: list, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        return _run(["git", *args], cwd or self.dir, mask=self._token)

    def _remote_has_main(self) -> bool:
        out = self._git(["ls-remote", "--heads", "origin", "main"]).stdout
        return bool(out.strip())

    def restore(self, results_dir: str = "results", label: Optional[str] = None,
                suffixes: Iterable[str] = ARTIFACT_SUFFIXES + (CKPT_SUFFIX,)) -> int:
        """Copy artifacts and checkpoints of ``label`` (default: this session's label) from the repo into
        ``results_dir`` when they are missing locally, so that a new runtime resumes where the last one died."""
        src = self.dir / (label or self.label)
        if not src.is_dir():
            return 0
        dst = Path(results_dir)
        n = 0
        for f in sorted(src.rglob("*")):
            if not f.is_file() or f.suffix not in tuple(suffixes):
                continue
            target = dst / f.relative_to(src)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n += 1
        return n

    def push(self, results_dir: str = "results", suffixes: Iterable[str] = ARTIFACT_SUFFIXES,
             message: Optional[str] = None, retries: int = 3, ckpt_max_bytes: int = CKPT_MAX_BYTES) -> int:
        """Copy artifacts (and checkpoints up to ``ckpt_max_bytes``) from ``results_dir`` into ``<label>/``
        and push. Returns files copied."""
        src = Path(results_dir)
        dst = self.dir / self.label
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(src.rglob("*")):
            if not f.is_file() or not _wanted(f, suffixes, ckpt_max_bytes):
                continue
            target = dst / f.relative_to(src)
            if target.exists() and target.stat().st_size == f.stat().st_size and target.read_bytes() == f.read_bytes():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n += 1
        if n == 0:
            return 0
        self._git(["add", "-A", self.label])
        msg = message or f"{self.label}: {n} file(s) {time.strftime('%Y-%m-%d %H:%M:%S')}"
        self._git(["commit", "-q", "-m", msg])
        for attempt in range(retries):
            try:
                if self._remote_has_main():   # a freshly created empty repo has no branch to pull yet
                    self._git(["pull", "-q", "--rebase", "origin", "main"])
                self._git(["push", "-q", "origin", "HEAD:main"])
                return n
            except SyncError as e:  # noqa: PERF203
                if attempt == retries - 1:
                    sys.stderr.write(f"results_sync: push failed: {e}\n")
                    raise
                time.sleep(2 * (attempt + 1))
        return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--repo", required=True, help="owner/name of the results repository")
    ap.add_argument("--label", required=True, help="subfolder in the results repo, e.g. colab-2026-10-27")
    ap.add_argument("--work-dir", default=None)
    a = ap.parse_args()
    sync = ResultsSync(a.repo, os.environ.get("RESULTS_TOKEN"), a.label, a.work_dir)
    n = sync.push(a.results_dir)
    print(f"pushed {n} file(s) to {a.repo}/{a.label}")


if __name__ == "__main__":
    main()
