from types import SimpleNamespace

import numpy as np

from armnet_rlt.rerun_view import (
    INDEX_PATH,
    KIND_FAIL,
    KIND_RUNNING,
    KIND_SUCCESS,
    KIND_TIMEOUT,
    RolloutRecordings,
    recording_id,
    recording_label,
    rerun_observation,
)


class _Entry:
    def __init__(self, path: str, scalar: float | None = None, image: object | None = None):
        self.entity_path = path
        self.scalar = scalar
        self.image = image

    def WhichOneof(self, _name: str) -> str:
        return "image" if self.image is not None else "scalar"


class _Packet:
    def __init__(self, entries: list[_Entry]) -> None:
        self.entries = entries


class _RecordingInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRerun:
    def __init__(self) -> None:
        self.recordings: list[str] = []
        self.connected: list[str] = []
        self.logs: list[tuple[str, object, str]] = []
        self.times: list[tuple[str, int]] = []
        self.names: list[str] = []
        self.blueprints = 0

    def RecordingStream(self, _application_id: str, *, recording_id: str) -> str:
        self.recordings.append(recording_id)
        return recording_id

    def connect_grpc(self, _url: str, *, recording: str) -> None:
        self.connected.append(recording)

    def set_time(self, timeline: str, *, sequence: int, recording: str) -> None:
        self.times.append((f"{recording}:{timeline}", sequence))

    def log(self, path: str, value: object, *, recording: str) -> None:
        self.logs.append((path, value, recording))

    def Scalars(self, value: float) -> tuple[str, float]:
        return ("scalar", value)

    def EncodedImage(self, **kwargs: object) -> tuple[str, object]:
        return ("jpeg", kwargs)

    def send_property(self, _name: str, values: _RecordingInfo, *, recording: str) -> None:
        self.names.append(f"{recording}:{values.name}")

    def send_blueprint(self, _blueprint: object, *, recording: str) -> None:
        self.blueprints += 1

    class archetypes:
        RecordingInfo = _RecordingInfo


def _identity(index: int, variant: int, kind: int | None = None) -> list[_Entry]:
    entries = [
        _Entry(INDEX_PATH, float(index)),
        _Entry("observation.rollout_variant", float(variant)),
    ]
    if kind is not None:
        entries.append(_Entry("observation.rollout_kind", float(kind)))
    return entries


def test_observation_namespaces_cameras_and_marks_the_rollout() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    payload = rerun_observation(
        {"front": image, "shoulder_pan.pos": 0.5},
        rollout_index=4,
        variant="base",
        kind=KIND_RUNNING,
    )

    assert payload["images.front"] is image
    assert payload["shoulder_pan.pos"] == 0.5
    assert payload[INDEX_PATH] == 4.0
    assert payload["observation.rollout_variant"] == 1.0
    assert payload["observation.rollout_kind"] == 0.0


def test_each_rollout_is_its_own_recording_and_outcome_renames_it() -> None:
    rr = _FakeRerun()
    session = RolloutRecordings(rr)
    first = _identity(1, 0, KIND_RUNNING) + [
        _Entry("observation.images.front", image=SimpleNamespace(
            encoding=1, data=b"jpeg-bytes"
        ))
    ]
    second = _identity(2, 0, KIND_RUNNING) + [
        _Entry("observation.shoulder_pan.pos", scalar=0.2)
    ]

    assert session.replay(_Packet(first)) is True
    assert session.replay(_Packet(second)) is True
    assert session.replay(_Packet(_identity(1, 0, KIND_SUCCESS))) is True
    assert session.replay(_Packet(_identity(2, 0, KIND_FAIL))) is True

    assert rr.recordings == ["rollout-0001-online_rlt", "rollout-0002-online_rlt"]
    assert rr.connected == rr.recordings
    assert [item[0] for item in rr.logs] == [
        "observation.images.front",
        "observation.shoulder_pan.pos",
        "outcome/success",
        "outcome/success",
    ]
    assert rr.logs[2][1] == ("scalar", 1.0)
    assert rr.logs[3][1] == ("scalar", 0.0)
    assert rr.names[-2:] == [
        "rollout-0001-online_rlt:" + recording_label(1, 0, KIND_SUCCESS),
        "rollout-0002-online_rlt:" + recording_label(2, 0, KIND_FAIL),
    ]
    assert recording_id(3, 1) == "rollout-0003-base"
    assert recording_label(3, 2, KIND_TIMEOUT) == "rollout 0003 frozen_rlt — timeout"
    # The second rollout starts its own step timeline at 0.
    assert ("rollout-0002-online_rlt:step", 0) in rr.times


def test_packets_without_a_rollout_index_stay_on_the_default_stream() -> None:
    rr = _FakeRerun()
    session = RolloutRecordings(rr)
    assert session.replay(_Packet([_Entry("observation.images.front", scalar=1.0)])) is False
    assert rr.recordings == []
