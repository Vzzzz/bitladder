import sys

import pytest

from bitladder.notebook import run


def test_run_streams_selected_lines(capsys):
    out = run(f'{sys.executable} -c "print(\'hello\'); print(\'RESULT x=1\')"')
    assert "RESULT x=1" in out and "hello" in out
    body = [ln for ln in capsys.readouterr().out.splitlines() if not ln.startswith("$ ")]
    assert "RESULT x=1" in body and "hello" not in body


def test_run_fails_loudly(capsys):
    with pytest.raises(RuntimeError):
        run(f'{sys.executable} -c "import sys; print(\'partial\'); sys.exit(3)"')
    assert "partial" in capsys.readouterr().out


def test_run_no_check_returns_output():
    out = run(f'{sys.executable} -c "import sys; print(\'bye\'); sys.exit(1)"', check=False)
    assert "bye" in out
