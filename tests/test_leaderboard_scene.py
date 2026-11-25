import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import leaderboard as L  # noqa: E402


def scene(rail, light, **aim):
    return {"values": {"rail": rail}, "brightness": light, **aim}


def test_protocol_scenes_pass() -> None:
    assert L.scene_error("cell-03", "id", scene(0.70, 30)) is None
    assert L.scene_error("cell-03", "ood", scene(0.62, 40)) is None
    assert L.scene_error("cell-03", "ood", scene(0.78, 20, top_tilt=0.3019)) is None
    assert L.scene_error("cell-01", "id", scene(0.60, 30, top_pan=0.405, top_tilt=0.175)) is None
    assert L.scene_error("cell-01", "ood", scene(0.52, 20)) is None


def test_other_scenes_fail() -> None:
    assert "rail" in L.scene_error("cell-03", "id", scene(0.62, 30))
    assert "rail" in L.scene_error("cell-01", "ood", scene(0.60, 40))
    assert "light" in L.scene_error("cell-03", "id", scene(0.70, 40))
    assert "light" in L.scene_error("cell-03", "ood", scene(0.62, 30))
    assert "top_tilt" in L.scene_error("cell-03", "ood", scene(0.62, 40, top_tilt=0.1661))
    assert "top_pan" in L.scene_error("cell-01", "id", scene(0.60, 30, top_pan=0.3))
