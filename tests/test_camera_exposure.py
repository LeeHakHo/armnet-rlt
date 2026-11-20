import sys
import types
from types import SimpleNamespace

from armnet_rlt.so101_actor import apply_camera_exposure

FAKE_CV2 = types.SimpleNamespace(CAP_PROP_AUTO_EXPOSURE=21, CAP_PROP_EXPOSURE=15, CAP_PROP_GAIN=14, CAP_PROP_BRIGHTNESS=10)


class Capture:
    def __init__(self):
        self.props = {21: 3.0, 15: 156.0, 14: 0.0, 10: 0.0}

    def get(self, prop):
        return self.props[prop]

    def set(self, prop, value):
        self.props[prop] = value

    def getBackendName(self):
        return "V4L2"


def test_auto_exposure_is_turned_off_at_the_current_value(monkeypatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "cv2", FAKE_CV2)
    top, front = Capture(), Capture()
    front.props[15] = 90.0
    robot = SimpleNamespace(cameras={"top": SimpleNamespace(videocapture=top), "front": SimpleNamespace(videocapture=front)})
    apply_camera_exposure(SimpleNamespace(args={}), robot)
    assert top.props[21] == 1 and top.props[15] == 156.0
    assert front.props[21] == 1 and front.props[15] == 90.0
    assert capsys.readouterr().out.count("[rlt] camera exposure") == 2


def test_robots_without_opencv_cameras_are_skipped() -> None:
    apply_camera_exposure(SimpleNamespace(args={}), SimpleNamespace(cameras={}))
