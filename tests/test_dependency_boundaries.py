"""Keep this project copyable outside alpha-robotics."""

from __future__ import annotations

import ast
import importlib.metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "armnet_rlt"
FORBIDDEN = ("hw_control", "missiontracker", "python", "armnet.")


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_source_imports_no_legacy_monorepo_package() -> None:
    bad: list[str] = []
    for path in SRC.rglob("*.py"):
        for name in _imports(path):
            if name == "python" or name.startswith(FORBIDDEN):
                bad.append(f"{path.relative_to(ROOT)}: {name}")
    assert bad == []


def test_actor_image_does_not_copy_parent_repository() -> None:
    dockerfile = (ROOT / "Dockerfile.actor").read_text()
    assert "COPY armnet/" not in dockerfile
    assert "COPY hw_control/" not in dockerfile
    assert "COPY missiontracker/" not in dockerfile
    assert "COPY python/" not in dockerfile
    assert "external/openpi" not in dockerfile
    assert "armnet-client==0.3.4" in dockerfile
    assert "armnet-runtime==0.3.2" in dockerfile
    assert "armnet-busybox==0.3.4" in dockerfile
    assert "armnet-core==0.3.2" in dockerfile
    assert "av>=14,<19" in dockerfile


def test_developer_sdk_is_the_published_armnet_distribution() -> None:
    distribution = importlib.metadata.distribution("armnet-client")
    assert distribution.version == "0.3.4"
    location = Path(distribution.locate_file("")).resolve()
    assert location.name == "site-packages"
    assert "alpha-robotics/armnet/client" not in str(location)
