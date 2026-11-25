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
        **{**PROTOCOL, "CELL": "cell-02", "EVAL_LEVELS": "id", "RECORD_PUBLIC": "0"},
    )
    assert out.returncode != 0
    for text in ("--episodes 200", "--eval-every 50", "CELL 'cell-02'", "EVAL_LEVELS 'id'", "RECORD_PUBLIC=0"):
        assert text in out.stdout
    assert "5 problems" in out.stdout


def test_protocol_ignores_the_environment() -> None:
    out = lib('echo "$EVAL_SEEDS|$EVAL_ID_ROLLOUTS|$RLT_CELL_RAIL"', EVAL_SEEDS="1042", EVAL_ID_ROLLOUTS="3",
              RLT_CELL_RAIL="cell-01=0.3,0.9")
    assert "1042 2042 3042|10|cell-01=0.55,0.65" in out.stdout
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


FAKE_FRONT = """
import sys, time
argv = sys.argv[1:]
endpoint = argv[argv.index("--public-endpoint") + 1] if "--public-endpoint" in argv else "abc.ngrok-free.app:443"
print("[RLT] front " + endpoint + " (TLS) on 127.0.0.1:50050 -> learner, policy", flush=True)
print("argv " + " ".join(argv), flush=True)
time.sleep(60)
"""
TUNNELS = 'echo "L=$LEARNER_TUNNEL"; echo "P=$POLICY_TUNNEL"; echo "S=$STEERING"; echo "F=${FRONT[*]:1}"'


def _vars(out: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in out.splitlines() if line[:2] in {"L=", "P=", "S=", "F="})


def test_ngrok_gives_both_servers_the_front_endpoint(tmp_path) -> None:
    (tmp_path / "front.py").write_text(FAKE_FRONT)
    before = _vars(lib(TUNNELS).stdout)
    assert before["F"] == "front --port 50050 --learner-port 50051 --policy-port 50052 --tunnel ngrok"
    assert before["L"] == before["P"] == ""
    out = lib(f'FRONT=(python3 {tmp_path}/front.py "${{FRONT[@]:1}}"); front_start {tmp_path}/front.log; {TUNNELS}; front_stop')
    assert out.returncode == 0, out.stdout + out.stderr
    after = _vars(out.stdout)
    assert after["L"] == "--tunnel none --public-endpoint abc.ngrok-free.app:443 --public-endpoint-tls --port 50051"
    assert after["P"] == "--tunnel none --public-endpoint abc.ngrok-free.app:443 --public-endpoint-tls --port 50052"
    assert after["S"] == "--steering-from-learner 127.0.0.1:50051"
    assert "argv front --port 50050 --learner-port 50051 --policy-port 50052 --tunnel ngrok" in (tmp_path / "front.log").read_text()


def test_public_host_serves_one_port_unless_the_policy_port_is_set() -> None:
    host = {"PUBLIC_HOST": "h.example", "TLS_CERT": "/c.pem", "TLS_KEY": "/k.pem", "FRONT_PORT": "20051",
            "LEARNER_PORT": "20053", "POLICY_PORT": "20052"}
    one = _vars(lib(TUNNELS, **host).stdout)
    assert one["F"] == ("front --port 20051 --learner-port 20053 --policy-port 20052 --tunnel none "
                        "--public-endpoint h.example:443 --tls-cert /c.pem --tls-key /k.pem")
    assert one["L"] == "--tunnel none --public-endpoint h.example:443 --public-endpoint-tls --port 20053"
    assert one["P"] == "--tunnel none --public-endpoint h.example:443 --public-endpoint-tls --port 20052"
    assert one["S"] == "--steering-from-learner 127.0.0.1:20053"
    two = _vars(lib(TUNNELS + "; front_start /nonexistent/log", **host, PUBLIC_POLICY_PORT="8443").stdout)
    assert two["F"] == ""
    assert two["L"] == "--tunnel none --public-endpoint h.example:443 --port 20053 --tls-cert /c.pem --tls-key /k.pem"
    assert two["P"] == "--tunnel none --public-endpoint h.example:8443 --port 20052 --tls-cert /c.pem --tls-key /k.pem"
    assert two["S"] == "--steering-from-learner 127.0.0.1:20053 --steering-learner-tls-name h.example"
    assert lib("true", TUNNEL="frp").returncode != 0


FAKE_TAILSCALE = """#!/usr/bin/env bash
echo "$*" >> "$TS_LOG"
case "$1" in --socket=*) shift ;; esac
case "$*" in
  "status --json") echo '{"Self": {"DNSName": "node.tail1.ts.net."}}' ;;
  cert*) [ -z "$TS_DENY" ] || { echo "Access denied: cert access denied" >&2; exit 1; }
         openssl req -x509 -newkey rsa:2048 -nodes -days 30 -subj /CN=node -keyout "$5" -out "$3" 2>/dev/null ;;
  "funnel status --json") echo "${TS_FUNNEL:-{\\}}" ;;
esac
"""


def test_tailscale_takes_the_tailnet_name_a_certificate_and_one_tcp_funnel(tmp_path) -> None:
    (tmp_path / "front.py").write_text(FAKE_FRONT)
    ts = tmp_path / "tailscale"
    ts.write_text(FAKE_TAILSCALE.lstrip())
    ts.chmod(0o755)
    calls = tmp_path / "calls"
    env = {"TUNNEL": "tailscale", "TAILSCALE": f"{ts} --socket=/x.sock", "TS_LOG": str(calls),
           "RLT_DATA_ROOT": str(tmp_path / "data"), "FRONT_PORT": "20051"}
    run = f'FRONT=(python3 {tmp_path}/front.py "${{FRONT[@]:1}}"); front_start {tmp_path}/front.log; {TUNNELS}; front_stop'
    out = lib(run, **env)
    assert out.returncode == 0, out.stdout + out.stderr
    tls = tmp_path / "data/tls/node.tail1.ts.net"
    assert calls.read_text().splitlines() == [
        "--socket=/x.sock status --json",
        f"--socket=/x.sock cert --cert-file {tls}.crt --key-file {tls}.key node.tail1.ts.net",
        "--socket=/x.sock funnel status --json",
        "--socket=/x.sock funnel --bg --tcp 443 tcp://localhost:20051",
        "--socket=/x.sock funnel --tcp 443 tcp://localhost:20051 off",
    ]
    assert _vars(out.stdout)["L"] == "--tunnel none --public-endpoint node.tail1.ts.net:443 --public-endpoint-tls --port 50051"
    assert (f"--tunnel none --public-endpoint node.tail1.ts.net:443 --tls-cert {tls}.crt --tls-key {tls}.key"
            in (tmp_path / "front.log").read_text())
    # A current certificate and a funnel already in place are kept, and left on.
    calls.unlink()
    out = lib(run, **env, TS_FUNNEL='{"TCP": {"443": {"TCPForward": "localhost:20051"}}}')
    assert out.returncode == 0, out.stdout + out.stderr
    assert calls.read_text().splitlines() == ["--socket=/x.sock status --json", "--socket=/x.sock funnel status --json"]
    # A funnel to another port is not taken over.
    out = lib(run, **env, TS_FUNNEL='{"TCP": {"443": {"TCPForward": "localhost:20061"}}}')
    assert out.returncode != 0 and "goes to localhost:20061" in out.stdout
    # Without operator rights the commands to run are printed.
    (tls.parent / "node.tail1.ts.net.crt").unlink()
    out = lib(run, **env, TS_DENY="1")
    assert out.returncode != 0
    assert f"{ts} --socket=/x.sock funnel --bg --tcp 443 tcp://localhost:20051" in out.stdout
    assert "sudo tailscale set --operator=$USER" in out.stdout


def test_modal_start_puts_the_front_first(tmp_path) -> None:
    spec = tmp_path / "data/logs/r/modal-train.jsonl"
    spec.parent.mkdir(parents=True)
    spec.write_text('{"name": "learner"}\n')
    out = lib(f'modal_cli() {{ cat "$5"; }}; modal_start lerobot {spec} {tmp_path}/call',
              BACKEND="modal", RLT_DATA_ROOT=str(tmp_path / "data"))
    assert out.returncode == 0, out.stdout + out.stderr
    import json

    first, second = (json.loads(line) for line in out.stdout.splitlines()[-2:])
    assert first["name"] == "front" and first["log"] == str(spec.parent / "front.log")
    assert first["argv"] == ["envs/venv-lerobot/bin/rlt-local", "front", "--port", "50050", "--learner-port", "50051",
                             "--policy-port", "50052", "--tunnel", "none", "--public-endpoint-tls", "--bind-host", "0.0.0.0"]
    assert second == {"name": "learner"}
