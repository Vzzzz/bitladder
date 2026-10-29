import subprocess

import pytest

from bitladder import results_sync
from bitladder.results_sync import ResultsSync


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A bare 'origin' with a main branch; ResultsSync is pointed at it via a file URL."""
    bare = tmp_path / "remote.git"
    _git(["init", "-q", "--bare", "-b", "main", str(bare)], tmp_path)
    seed = tmp_path / "seed"
    _git(["clone", "-q", str(bare), str(seed)], tmp_path)
    _git(["config", "user.email", "t@t"], seed)
    _git(["config", "user.name", "t"], seed)
    (seed / "README.md").write_text("results\n")
    _git(["add", "README.md"], seed)
    _git(["commit", "-q", "-m", "init"], seed)
    _git(["push", "-q", "origin", "HEAD:main"], seed)

    # Route the https URL to the local bare repo: patch the URL before cloning.
    orig_run = results_sync._run

    def patched_run(args, cwd, quiet=True, mask=None):
        args = [str(bare) if isinstance(a, str) and a.startswith("https://x-access-token:") else a for a in args]
        return orig_run(args, cwd, quiet, mask)

    monkeypatch.setattr(results_sync, "_run", patched_run)
    return bare


def test_push_copies_artifacts_and_skips_checkpoints(tmp_path, remote):
    results = tmp_path / "results"
    results.mkdir()
    (results / "a.json").write_text("{}")
    (results / "a.pt").write_bytes(b"\x00" * 10)
    (results / "sub").mkdir()
    (results / "sub" / "b.md").write_text("# b")
    sync = ResultsSync("owner/results", "secret-token", "sess1", work_dir=str(tmp_path / "work"))
    assert sync.push(str(results)) == 2
    assert sync.push(str(results)) == 0  # nothing new
    (results / "a.json").write_text('{"x": 1}')
    assert sync.push(str(results)) == 1
    log = subprocess.run(["git", "log", "--oneline", "main"], cwd=str(remote), capture_output=True, text=True).stdout
    assert log.count("\n") == 3  # init + 2 pushes
    tree = subprocess.run(["git", "ls-tree", "-r", "--name-only", "main"], cwd=str(remote),
                          capture_output=True, text=True).stdout.split()
    assert "sess1/a.json" in tree and "sess1/sub/b.md" in tree and "sess1/a.pt" not in tree


def test_missing_token_raises(tmp_path):
    with pytest.raises(ValueError):
        ResultsSync("owner/results", None, "x", work_dir=str(tmp_path))


def test_push_into_empty_remote(tmp_path, monkeypatch):
    """A freshly created GitHub repo has no branch yet: the first push must create main."""
    bare = tmp_path / "empty.git"
    _git(["init", "-q", "--bare", "-b", "main", str(bare)], tmp_path)
    orig_run = results_sync._run

    def patched_run(args, cwd, quiet=True, mask=None):
        args = [str(bare) if isinstance(a, str) and a.startswith("https://x-access-token:") else a for a in args]
        return orig_run(args, cwd, quiet, mask)

    monkeypatch.setattr(results_sync, "_run", patched_run)
    results = tmp_path / "results"
    results.mkdir()
    (results / "a.json").write_text("{}")
    sync = ResultsSync("owner/results", "secret-token", "first", work_dir=str(tmp_path / "work"))
    assert sync.push(str(results)) == 1
    tree = subprocess.run(["git", "ls-tree", "-r", "--name-only", "main"], cwd=str(bare),
                          capture_output=True, text=True).stdout.split()
    assert tree == ["first/a.json"]


def test_git_errors_never_contain_the_token(tmp_path):
    with pytest.raises(results_sync.SyncError) as ei:
        results_sync._run(["git", "clone", "-q", str(tmp_path / "missing-SECRET123.git"), str(tmp_path / "dst")],
                          tmp_path, mask="SECRET123")
    assert "SECRET123" not in str(ei.value) and "<token>" in str(ei.value)
    with pytest.raises(results_sync.SyncError) as ei:
        ResultsSync("owner/results", "SECRET123", "x", work_dir=str(tmp_path / "w2")) if False else \
            results_sync._run(["git", "rev-parse", "--verify", "nope-SECRET123"], tmp_path, mask="SECRET123")
    assert "SECRET123" not in str(ei.value)


def test_restore_brings_back_missing_artifacts(tmp_path, remote):
    results = tmp_path / "results"
    results.mkdir()
    (results / "a.json").write_text("{}")
    sync = ResultsSync("owner/results", "secret-token", "sess1", work_dir=str(tmp_path / "work"))
    assert sync.push(str(results)) == 1
    (results / "a.json").unlink()
    (results / "b.json").write_text("{}")
    assert sync.restore(str(results)) == 1
    assert (results / "a.json").read_text() == "{}" and (results / "b.json").exists()
    assert sync.restore(str(results)) == 0
    assert sync.restore(str(results), label="nope") == 0
