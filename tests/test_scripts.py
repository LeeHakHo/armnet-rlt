import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts" / "lib.sh"
PROTOCOL = {"CELL": "cell-03", "RECORD_REPO": "u/armnet"}


def lib(command: str, **env: str) -> subprocess.CompletedProcess:
    base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp")}
    return subprocess.run(
        ["bash", "-c", f'. "{LIB}" >/dev/null; {command}'],
        capture_output=True, text=True, env={**base, **env},
    )


def test_require_cell_accepts_only_benchmark_cells() -> None:
    assert lib("require_cell", CELL="cell-01").returncode == 0
    assert lib("require_cell", CELL="cell-03").returncode == 0
    empty = lib("require_cell")
    assert empty.returncode != 0 and "cell-01 or cell-03" in empty.stdout
    unknown = lib("require_cell", CELL="cell-02")
    assert unknown.returncode != 0 and "unknown cell cell-02" in unknown.stdout


def test_leaderboard_preflight_accepts_the_protocol_run() -> None:
    assert lib("leaderboard_preflight 400 100", **PROTOCOL).returncode == 0


def test_leaderboard_preflight_names_each_deviation() -> None:
    out = lib(
        "leaderboard_preflight 200 50",
        **{**PROTOCOL, "CELL": "cell-02", "EVAL_SEEDS": "1042", "EVAL_LIGHT": "30", "RECORD_PUBLIC": "0"},
    )
    assert out.returncode != 0
    for text in ("--episodes 200", "--eval-every 50", "CELL 'cell-02'", "EVAL_SEEDS '1042'",
                 "EVAL_LIGHT 30", "RECORD_PUBLIC=0"):
        assert text in out.stdout
    assert "5 problems" not in out.stdout and "6 problems" in out.stdout
    assert "RECORD_REPO unset" in lib("leaderboard_preflight 400 100", CELL="cell-01").stdout
    assert "RECORD 0" in lib("leaderboard_preflight 400 100", **PROTOCOL, RECORD="0").stdout


def test_leaderboard_reads_dotenv_without_overriding(tmp_path, monkeypatch) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("leaderboard", ROOT / "scripts/leaderboard.py")
    lb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lb)
    env = tmp_path / ".env"
    env.write_text("# comment\nRLT_T_A=from-file\nexport RLT_T_B='quoted'\nRLT_T_C=file\nnot a line\n")
    monkeypatch.setenv("RLT_T_C", "env")
    monkeypatch.delenv("RLT_T_A", raising=False)
    monkeypatch.delenv("RLT_T_B", raising=False)
    lb.load_dotenv(env)
    assert os.environ["RLT_T_A"] == "from-file"
    assert os.environ["RLT_T_B"] == "quoted"
    assert os.environ["RLT_T_C"] == "env"
    lb.load_dotenv(tmp_path / "missing")
    for key in ("RLT_T_A", "RLT_T_B"):
        monkeypatch.delenv(key)
