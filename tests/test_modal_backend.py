import contextlib
import importlib
import json
import os
import shlex
import subprocess
import sys
import types
from pathlib import Path

import pytest

from armnet_rlt import local_app, modal_backend
from armnet_rlt.modal_backend import (
    container_argv,
    needs_tunnel,
    process_spec,
    remote_argv,
    supervise,
    tunnel_port,
    volume_path,
)

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts" / "lib.sh"
CONFIG = "pi05_rlt_busybox_multitask_singlearm_minmax"


def lib_vars(*names: str, **env: str) -> dict[str, str]:
    """Variables lib.sh sets for the given environment."""
    base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp")}
    script = "; ".join(f'echo "{n}=${{{n}}}"' for n in names)
    out = subprocess.run(
        ["bash", "-c", f'. "{LIB}" >/dev/null; {script}'],
        capture_output=True, text=True, check=True, env={**base, **env},
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines())


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("RLT_DATA_ROOT", str(tmp_path / "data"))
    return tmp_path / "data"


def test_local_paths_map_to_the_volume(data) -> None:
    checkpoints = ROOT / "checkpoints"
    assert volume_path(data / "tasks" / CONFIG / "demo" / "rlt_demo_cache.pt") == f"tasks/{CONFIG}/demo/rlt_demo_cache.pt"
    assert volume_path(checkpoints / "pi05_rltoken5k") == "checkpoints/pi05_rltoken5k"
    assert volume_path("/data/lerobot-sbc/run/checkpoints/000300") == "lerobot-sbc/run/checkpoints/000300"
    assert volume_path("/elsewhere/model") is None
    argv = ["envs/venv-openpi/bin/rlt-local", "policy-server", "--checkpoint", str(checkpoints / "pi05_rltoken5k"),
            "--actor-checkpoint", str(data / "tasks/c/frozen/r/step_000100.pt"),
            f"--output_dir={data}/dit/demo", "--prompt", "push the green button", "/elsewhere/x"]
    assert remote_argv(argv) == [
        "envs/venv-openpi/bin/rlt-local", "policy-server", "--checkpoint", "/data/checkpoints/pi05_rltoken5k",
        "--actor-checkpoint", "/data/tasks/c/frozen/r/step_000100.pt",
        "--output_dir=/data/dit/demo", "--prompt", "push the green button", "/elsewhere/x",
    ]


def test_process_spec_maps_the_log_and_forwards_the_policy_instance(data, monkeypatch) -> None:
    monkeypatch.setenv("RLT_POLICY_INSTANCE", "50052-eval")
    monkeypatch.setenv("ROBOMETER_URL", "http://r:8010")
    spec = process_spec({"name": "learner", "argv": ["x", str(data / "a")], "log": str(data / "logs/run/learner.log")})
    assert spec["argv"] == ["x", "/data/a"]
    assert spec["log"] == "/data/logs/run/learner.log"
    assert spec["env"]["RLT_POLICY_INSTANCE"] == "50052-eval"
    assert spec["env"]["ROBOMETER_URL"] == "http://r:8010"
    with pytest.raises(SystemExit):
        process_spec({"name": "x", "argv": [], "log": "/elsewhere/log"})


def test_modal_tunnel_flags_register_the_front_tunnel_for_both_servers() -> None:
    """On Modal the learner and the policy server register the container's one tunnel,
    the front's; the front serves plaintext on all interfaces behind the Modal edge."""
    names = ("LEARNER_TUNNEL", "POLICY_TUNNEL", "STEERING", "GPU_ARG", "FRONT[*]")
    local = lib_vars(*names, PUBLIC_HOST="h.example", TLS_CERT="/c.pem", TLS_KEY="/k.pem", GPU="1")
    modal = lib_vars(*names, BACKEND="modal", GPU="1")
    assert local["GPU_ARG"] == "--gpu 1" and modal["GPU_ARG"] == ""
    assert modal["STEERING"] == local["STEERING"] == "--steering-from-learner 127.0.0.1:50051"
    for name, port in (("LEARNER_TUNNEL", 50051), ("POLICY_TUNNEL", 50052)):
        flags = shlex.split(modal[name])
        assert needs_tunnel(flags) and tunnel_port(flags) == 50050
        flags = container_argv(flags, "r1.modal.host:443")
        assert flags == ["--tunnel", "none", "--public-endpoint", "r1.modal.host:443",
                         "--public-endpoint-tls", "--port", str(port)]
        assert container_argv(shlex.split(local[name]), "x") == ["--tunnel", "none", "--public-endpoint", "h.example:443",
                                                                 "--public-endpoint-tls", "--port", str(port)]
    front = shlex.split(modal["FRONT[*]"])
    assert front[1:] == ["front", "--port", "50050", "--learner-port", "50051", "--policy-port", "50052",
                         "--tunnel", "none", "--public-endpoint-tls", "--bind-host", "0.0.0.0"]
    assert not needs_tunnel(front)


def test_rlt_local_serves_plaintext_on_localhost_behind_the_modal_front(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RLT_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("RLT_RENDEZVOUS_DIR", str(tmp_path / "rv"))
    flags = container_argv(shlex.split(lib_vars("LEARNER_TUNNEL", BACKEND="modal")["LEARNER_TUNNEL"]), "r1.modal.host:443")
    args = local_app._parser().parse_args(["online", "--config-name", CONFIG, "--run-id", "run-1", *flags])
    cfg = types.SimpleNamespace(actor_learner=types.SimpleNamespace())
    with local_app._serve(args, cfg, "token"):
        entry = json.loads((tmp_path / "rv" / f"{CONFIG}__run-1.json").read_text())
    assert cfg.actor_learner.server_bind_host == "127.0.0.1"
    assert not getattr(cfg.actor_learner, "tls_server_cert_path", "")
    assert entry["key"] == f"{CONFIG}:run-1"
    assert (entry["host"], entry["port"], entry["use_tls"]) == ("r1.modal.host", 443, True)
    assert "tls_root_cert_pem" not in entry
    # --bind-host still sets the listen address.
    args = local_app._parser().parse_args(
        ["online", "--config-name", CONFIG, "--run-id", "run-2", "--tunnel", "none",
         "--public-endpoint", "h:443", "--public-endpoint-tls", "--bind-host", "0.0.0.0"])
    with local_app._serve(args, cfg, "token"):
        pass
    assert cfg.actor_learner.server_bind_host == "0.0.0.0"


CHILD = """
import json, os, sys, time
argv = sys.argv[1:]
key = argv[argv.index("--key") + 1]
path = os.path.join(os.environ["RLT_RENDEZVOUS_DIR"], key.replace(":", "__") + ".json")
endpoint = argv[argv.index("--public-endpoint") + 1] if "--public-endpoint" in argv else ""
open(path, "w").write(json.dumps({"key": key, "endpoint": endpoint, "instance": os.environ.get("RLT_POLICY_INSTANCE")}))
print("up", flush=True)
time.sleep(float(argv[argv.index("--sleep") + 1]))
"""


def test_supervise_publishes_endpoints_and_stops_with_the_first_process(tmp_path) -> None:
    child = tmp_path / "child.py"
    child.write_text(CHILD)
    tunnels, published, removed, commits = [], {}, [], []

    @contextlib.contextmanager
    def forward(port):
        tunnels.append(port)
        yield f"t{port}.modal.host", 443

    def publish(key, value):
        published[key] = value

    learner = [sys.executable, str(child), "--key", "cfg:run", "--sleep", "1.5", "--tunnel", "none",
               "--public-endpoint", "modal", "--port", "50051"]
    server = [sys.executable, str(child), "--key", "policy:cfg:1", "--sleep", "60",
              "--public-endpoint", "modal", "--port", "50052"]
    codes = supervise(
        [{"name": "learner", "argv": learner, "log": str(tmp_path / "logs/learner.log"), "env": {}},
         {"name": "policy-server", "argv": server, "log": str(tmp_path / "logs/server.log"),
          "env": {"RLT_POLICY_INSTANCE": "1"}}],
        forward=forward, publish=publish, unpublish=removed.append, commit=lambda: commits.append(1),
        call_key="call:fc-1", rendezvous_dir=tmp_path / "rv", cwd=str(tmp_path), poll_s=0.1,
        commit_every_s=0.5,
    )
    assert tunnels == [50051, 50052]
    assert codes["learner"] == 0 and codes["policy-server"] != 0
    assert published["cfg:run"]["endpoint"] == "t50051.modal.host:443"
    assert published["policy:cfg:1"] == {"key": "policy:cfg:1", "endpoint": "t50052.modal.host:443", "instance": "1"}
    assert published["call:fc-1"] == {"ready": True, "keys": ["cfg:run", "policy:cfg:1"]}
    assert set(removed) == {"cfg:run", "policy:cfg:1", "call:fc-1"}
    assert len(commits) >= 2
    assert (tmp_path / "logs/learner.log").read_text() == "up\n"


def test_supervise_opens_one_tunnel_for_the_front_port(tmp_path) -> None:
    child = tmp_path / "child.py"
    child.write_text(CHILD)
    tunnels, published = [], {}

    @contextlib.contextmanager
    def forward(port):
        tunnels.append(port)
        yield f"t{port}.modal.host", 443

    front = [sys.executable, "-c", "import time; time.sleep(60)", "--port", "50050"]
    learner = [sys.executable, str(child), "--key", "cfg:run", "--sleep", "1.5",
               "--public-endpoint", "modal:50050", "--public-endpoint-tls", "--port", "50051"]
    server = [sys.executable, str(child), "--key", "policy:cfg:1", "--sleep", "60",
              "--public-endpoint", "modal:50050", "--public-endpoint-tls", "--port", "50052"]
    supervise(
        [{"name": n, "argv": a, "log": str(tmp_path / f"logs/{n}.log"), "env": {}}
         for n, a in (("front", front), ("learner", learner), ("policy-server", server))],
        forward=forward, publish=published.__setitem__, unpublish=lambda key: None, commit=lambda: None,
        call_key="call:fc-2", rendezvous_dir=tmp_path / "rv", cwd=str(tmp_path), poll_s=0.1,
    )
    assert tunnels == [50050]
    assert published["cfg:run"]["endpoint"] == published["policy:cfg:1"]["endpoint"] == "t50050.modal.host:443"
    assert published["call:fc-2"] == {"ready": True, "keys": ["cfg:run", "policy:cfg:1"]}


def test_run_command_appends_to_its_log(tmp_path) -> None:
    commits = []
    log = tmp_path / "a/log"
    result = modal_backend.run_command(
        [sys.executable, "-c", "import os; print(os.environ['X']); raise SystemExit(3)"], str(log), {"X": "y"},
        commit=lambda: commits.append(1), cwd=str(tmp_path))
    assert result == {"returncode": 3, "tail": "y\n"} and commits == [1]


# A fake modal package for the local commands and the app definition.


class FakeDict(dict):
    def put(self, key, value):
        self[key] = value


class FakeCall:
    calls: dict = {}

    def __init__(self, object_id, outcome=None):
        self.object_id, self.outcome, self.cancelled = object_id, outcome, False
        FakeCall.calls[object_id] = self

    @classmethod
    def from_id(cls, object_id):
        return cls.calls[object_id]

    def get(self, timeout=None):
        if self.outcome is None:
            raise TimeoutError
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def cancel(self):
        self.cancelled = True


class FakeFunction:
    spawned: list = []

    def __init__(self, name):
        self.name = name

    def spawn(self, processes):
        FakeFunction.spawned.append((self.name, processes))
        call = FakeCall(f"fc-{len(FakeFunction.spawned)}")
        DICTS.setdefault("armnet-rlt-rendezvous", FakeDict())[f"call:{call.object_id}"] = {"ready": True, "keys": ["k"]}
        return call

    def remote(self, argv, log, env):
        FakeFunction.spawned.append((self.name, argv, log))
        return {"returncode": 0, "tail": "done\n"}


class FakeEntry:
    def __init__(self, path):
        self.path = path


class FakeVolume:
    def __init__(self):
        self.files: dict[str, bytes] = {}

    def listdir(self, path):
        prefix = path.rstrip("/") + "/" if path not in ("", ".", "/") else ""
        names = {p[len(prefix):].split("/")[0] for p in self.files if p.startswith(prefix)}
        if not names:
            raise FileNotFoundError(path)
        return [FakeEntry(prefix + n) for n in names]

    def read_file_into_fileobj(self, path, out):
        if path not in self.files:
            raise FileNotFoundError(path)
        out.write(self.files[path])

    @contextlib.contextmanager
    def batch_upload(self, force=False):
        volume = self

        class Batch:
            def put_file(self, local, remote):
                volume.files[remote] = Path(local).read_bytes()

        yield Batch()


DICTS: dict = {}
VOLUME = FakeVolume()


class _Chain:
    """Records Image calls."""

    def __init__(self, steps=()):
        self.steps = list(steps)

    def __getattr__(self, name):
        return lambda *a, **k: _Chain([*self.steps, (name, a, k)])


def fake_modal() -> types.ModuleType:
    modal = types.ModuleType("modal")
    modal.exception = types.SimpleNamespace(
        TimeoutError=type("TimeoutError", (Exception,), {}),
        ConnectionError=type("ConnectionError", (Exception,), {}),
        NotFoundError=type("NotFoundError", (Exception,), {}),
    )
    modal.Dict = types.SimpleNamespace(from_name=lambda name, **k: DICTS.setdefault(name, FakeDict()))
    modal.Volume = types.SimpleNamespace(from_name=lambda name, **k: VOLUME)
    modal.Secret = types.SimpleNamespace(from_name=lambda name: ("secret", name))
    modal.Function = types.SimpleNamespace(from_name=lambda app, name: FakeFunction(name))
    modal.FunctionCall = FakeCall
    modal.Image = types.SimpleNamespace(debian_slim=lambda **k: _Chain([("debian_slim", (), k)]))

    class App:
        def __init__(self, name, **kwargs):
            self.name, self.kwargs, self.functions = name, kwargs, {}

        def function(self, **kwargs):
            def register(fn):
                self.functions[fn.__name__] = kwargs
                return fn
            return register

    modal.App = App
    modal.forward = lambda port, **k: (port, k)
    modal.current_function_call_id = lambda: "fc-x"
    return modal


@pytest.fixture
def modal(monkeypatch):
    DICTS.clear()
    VOLUME.files.clear()
    FakeFunction.spawned.clear()
    module = fake_modal()
    monkeypatch.setitem(sys.modules, "modal", module)
    monkeypatch.setenv("RLT_BACKEND", "modal")
    return module


def test_submitters_read_endpoints_from_the_modal_dict(modal, monkeypatch, tmp_path) -> None:
    from armnet_rlt.resources import read_rendezvous, remote_policy_job_args
    from armnet_rlt.submit_actor import resolve_rendezvous

    monkeypatch.setenv("RLT_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("RLT_POLICY_INSTANCE", "50052-eval")
    rendezvous = DICTS.setdefault("armnet-rlt-rendezvous", FakeDict())
    rendezvous[f"{CONFIG}:run-1"] = {"key": f"{CONFIG}:run-1", "host": "r1.modal.host", "port": 443,
                                      "use_tls": True, "config_name": CONFIG, "run_id": "run-1"}
    rendezvous[f"policy:{CONFIG}:50052-eval"] = {"host": "r2.modal.host", "port": 443, "use_tls": True,
                                                 "config_name": CONFIG}
    assert resolve_rendezvous(f"{CONFIG}:run-1")["host"] == "r1.modal.host"
    job = remote_policy_job_args(CONFIG, timeout_s=5.0, image_codec="jpeg")
    assert (job["remote_policy_host"], job["remote_policy_port"], job["remote_policy_use_tls"]) == ("r2.modal.host", 443, True)
    with pytest.raises(RuntimeError, match="Modal Dict"):
        resolve_rendezvous(f"{CONFIG}:missing")
    # A stale local file is not read with RLT_BACKEND=modal.
    monkeypatch.setenv("RLT_BACKEND", "local")
    assert read_rendezvous(f"{CONFIG}:run-1") is None


def test_start_alive_stop(modal, data, tmp_path, capsys) -> None:
    specs = tmp_path / "specs.jsonl"
    specs.write_text(json.dumps({"name": "learner", "argv": ["a", str(data / "x")], "log": str(data / "logs/l.log")}) + "\n")
    call_file = tmp_path / "train.call"
    modal_backend.main(["start", "--image", "lerobot", "--specs", str(specs), "--call-file", str(call_file)])
    name, processes = FakeFunction.spawned[-1]
    assert name == "serve_lerobot" and processes[0]["argv"] == ["a", "/data/x"]
    call = FakeCall.from_id(call_file.read_text().strip())
    with pytest.raises(SystemExit) as alive:
        modal_backend.main(["alive", str(call_file)])
    assert alive.value.code == 0
    call.outcome = {"learner": 0}
    with pytest.raises(SystemExit) as alive:
        modal_backend.main(["alive", str(call_file)])
    assert alive.value.code == 1
    modal_backend.main(["stop", str(call_file)])
    assert call.cancelled and not call_file.exists()
    assert f"call:{call.object_id}" not in DICTS["armnet-rlt-rendezvous"]


def test_start_fails_when_the_call_ends_before_it_is_ready(modal, data, tmp_path, monkeypatch) -> None:
    def spawn(self, processes):
        return FakeCall("fc-dead", outcome=RuntimeError("image build failed"))

    monkeypatch.setattr(FakeFunction, "spawn", spawn)
    specs = tmp_path / "specs.jsonl"
    specs.write_text(json.dumps({"name": "s", "argv": [], "log": str(data / "l.log")}) + "\n")
    with pytest.raises(SystemExit, match="image build failed"):
        modal_backend.main(["start", "--specs", str(specs), "--call-file", str(tmp_path / "c")])


def test_volume_file_commands(modal, data) -> None:
    VOLUME.files["tasks/c/rlt_checkpoints/r/rollout_metrics.jsonl"] = b"{}\n{}\n"
    target = data / "tasks/c/rlt_checkpoints/r/rollout_metrics.jsonl"

    def code(*argv):
        with pytest.raises(SystemExit) as exit_:
            modal_backend.main(list(argv))
        return exit_.value.code

    assert code("exists", str(data / "tasks/c/rlt_checkpoints/r")) == 0
    assert code("exists", str(data / "tasks/c/rlt_checkpoints/q")) == 1
    assert code("exists", "relative-run-id/config.json") == 1
    modal_backend.main(["get", str(target)])
    assert target.read_bytes() == b"{}\n{}\n"
    assert code("get", str(data / "missing.jsonl")) == 1
    frozen = data / "tasks/c/frozen/r/step_000100.pt"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(b"pt")
    modal_backend.main(["put", str(frozen)])
    assert VOLUME.files["tasks/c/frozen/r/step_000100.pt"] == b"pt"
    assert code("run", "--image", "rl", "--log", str(data / "logs/x.log"), "--", "hf", "download") == 0
    assert FakeFunction.spawned[-1] == ("run_command", ["hf", "download"], "/data/logs/x.log")


def test_modal_app_serves_both_images_on_one_gpu_with_the_volume(modal) -> None:
    sys.modules.pop("armnet_rlt.modal_app", None)
    app_module = importlib.import_module("armnet_rlt.modal_app")
    try:
        functions = app_module.app.functions
        assert set(functions) == {"serve", "serve_lerobot", "run_command", "run_command_lerobot"}
        for options in functions.values():
            assert options["gpu"] and options["volumes"] == {"/data": VOLUME}
            assert options["timeout"] == 24 * 60 * 60
            assert ("secret", "armnet-rlt-auth") in options["secrets"]
        steps = functions["serve"]["image"].steps
        commands = " ".join(str(a) for name, a, _ in steps if name == "run_commands")
        assert "setup_local.sh learner" in commands and "setup_local.sh openpi" in commands
        assert "install_plugins.sh" in commands
        local_dirs = [(a[0], k) for name, a, k in steps if name == "add_local_dir"]
        assert local_dirs[-1][0] == ROOT / "src" and not local_dirs[-1][1].get("copy")
        assert any(path == ROOT / "plugins" and k.get("copy") for path, k in local_dirs)
        lerobot = " ".join(str(a) for name, a, _ in functions["serve_lerobot"]["image"].steps if name == "run_commands")
        assert "setup_local.sh lerobot" in lerobot and "openpi" not in lerobot
        assert app_module.app.kwargs == {"include_source": False}
    finally:
        sys.modules.pop("armnet_rlt.modal_app", None)


def test_modal_forward_is_tls_with_http2() -> None:
    source = (ROOT / "src/armnet_rlt/modal_app.py").read_text()
    assert "modal.forward(port, h2_enabled=True)" in source
    assert "unencrypted=True" not in source


def test_modal_proc_writes_a_spec_line(tmp_path) -> None:
    out = subprocess.run(
        ["bash", "-c", f'. "{LIB}" >/dev/null; modal_proc "{tmp_path}/s.jsonl" learner /d/l.log a "b c"; cat "{tmp_path}/s.jsonl"'],
        capture_output=True, text=True, check=True, env={"PATH": os.environ["PATH"], "HOME": "/tmp"},
    ).stdout
    assert json.loads(out) == {"name": "learner", "log": "/d/l.log", "argv": ["a", "b c"]}
