"""Push experiment artifacts (JSON / CSV / Markdown / PNG, never checkpoints) to a separate
git repository so that a dying Colab session loses at most one run.

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


def _run(args: list, cwd: Path, quiet: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(cwd), check=True, capture_output=quiet, text=True)


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
        self._url = f"https://x-access-token:{token}@github.com/{repo}.git"
        if not (self.dir / ".git").is_dir():
            _run(["git", "clone", "-q", self._url, str(self.dir)], base)
        _run(["git", "config", "user.name", user], self.dir)
        _run(["git", "config", "user.email", email], self.dir)
        _run(["git", "remote", "set-url", "origin", self._url], self.dir)

    def push(self, results_dir: str = "results", suffixes: Iterable[str] = ARTIFACT_SUFFIXES,
             message: Optional[str] = None, retries: int = 3) -> int:
        """Copy artifacts from ``results_dir`` into ``<label>/`` and push. Returns files copied."""
        src = Path(results_dir)
        dst = self.dir / self.label
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(src.rglob("*")):
            if not f.is_file() or f.suffix not in suffixes:
                continue
            target = dst / f.relative_to(src)
            if target.exists() and target.stat().st_size == f.stat().st_size and target.read_bytes() == f.read_bytes():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n += 1
        if n == 0:
            return 0
        _run(["git", "add", "-A", self.label], self.dir)
        msg = message or f"{self.label}: {n} file(s) {time.strftime('%Y-%m-%d %H:%M:%S')}"
        _run(["git", "commit", "-q", "-m", msg], self.dir)
        for attempt in range(retries):
            try:
                _run(["git", "pull", "-q", "--rebase", "origin", "main"], self.dir)
                _run(["git", "push", "-q", "origin", "HEAD:main"], self.dir)
                return n
            except subprocess.CalledProcessError as e:  # noqa: PERF203
                if attempt == retries - 1:
                    sys.stderr.write(f"results_sync: push failed: {e.stderr[-400:] if e.stderr else e}\n")
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
