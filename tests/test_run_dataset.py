"""Recording every cell job of a run into one LeRobot dataset. The round trip needs
LeRobot (envs/venv-lerobot, the actor image) and is skipped without it."""

import json
import types

import numpy as np
import pytest

from armnet_rlt import dataset_recording
from fake_hub import FakeHub

REPO = "u/armnet_run"
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
    "observation.images.top": {"dtype": "video", "shape": (24, 32, 3), "names": ["height", "width", "channels"]},
}


class _Writer:
    def flush(self): pass
    def close(self): pass


class _Ctx:
    def __init__(self, job_id):
        self.job_id = job_id
        self.messages = []

    def report_progress(self, message):
        self.messages.append(message)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    hub = FakeHub(tmp_path / "hub")
    hub.install(monkeypatch)
    monkeypatch.setattr(dataset_recording, "RECORDING_ROOT", tmp_path / "cell")
    return hub


def _record(dataset, episodes, value):
    for e in range(episodes):
        for _ in range(4 + e):
            dataset.add_frame({
                "observation.state": np.full(6, value + e, np.float32),
                "action": np.zeros(6, np.float32),
                "observation.images.top": np.full((24, 32, 3), 10 * (value + e), np.uint8),
                "task": "push the green button",
            })
        dataset.save_episode()


def _job(job_id, episodes, value, labels, *, finish=True):
    dataset = dataset_recording.open_run_dataset(REPO, FEATURES, 20, public=True, vcodec="h264", streaming_encoding=False)
    _record(dataset, episodes, value)
    if not finish:
        return dataset
    rows = [{"success": e % 2 == 0, "outcome": "success" if e % 2 == 0 else "fail"} for e in range(episodes)]
    return dataset_recording.finish_dataset(
        _Ctx(job_id), dataset, _Writer(), repo_id=REPO, push_to_hub=True, public=True,
        episode_labels=rows, episode_frame_times=[[0.0, 0.05]] * episodes, job_labels=labels)


def test_jobs_append_to_one_lerobot_dataset(hub, tmp_path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    assert _job("job-a", 2, 0, {"phase": "train"}) == f"https://huggingface.co/datasets/{REPO}"
    # The second job downloads meta/ only.
    second = dataset_recording.open_run_dataset(REPO, FEATURES, 20, public=True, vcodec="h264", streaming_encoding=False)
    assert hub.downloads[-1] == (REPO, ["meta/"])
    assert sorted(p.name for p in second.root.iterdir()) == ["meta"]
    assert second.num_episodes == 2
    _record(second, 3, 2)
    rows = [{"success": True}] * 3
    dataset_recording.finish_dataset(_Ctx("job-b"), second, _Writer(), repo_id=REPO, push_to_hub=True, public=True,
                                     episode_labels=rows, episode_frame_times=[[0.0]] * 3,
                                     job_labels={"phase": "eval", "checkpoint_episodes": 100})
    assert hub.tags[REPO]["v3.0"] == hub.sha(REPO)

    whole = LeRobotDataset(REPO, root=hub.repo(REPO))
    assert whole.num_episodes == 5
    for e in range(5):
        meta = whole.meta.episodes[e]
        item = whole[int(meta["dataset_from_index"])]
        assert int(item["episode_index"]) == e and float(item["observation.state"][0]) == e
    labels = [json.loads(l) for l in (hub.repo(REPO) / "meta/armnet_rollouts.jsonl").read_text().splitlines()]
    assert [r["episode_index"] for r in labels] == [0, 1, 2, 3, 4]
    assert [r["job_id"] for r in labels] == ["job-a"] * 2 + ["job-b"] * 3
    assert [r["phase"] for r in labels] == ["train"] * 2 + ["eval"] * 3
    assert labels[2]["checkpoint_episodes"] == 100
    times = [json.loads(l)["episode_index"] for l in (hub.repo(REPO) / "meta/armnet_frame_times.jsonl").read_text().splitlines()]
    assert times == [0, 1, 2, 3, 4]


def test_a_second_writer_fails_without_touching_the_dataset(hub) -> None:
    _job("job-a", 1, 0, {})
    first = _job("job-b", 1, 1, {}, finish=False)
    _job("job-c", 2, 5, {})
    head = hub.sha(REPO)
    with pytest.raises(RuntimeError, match="another job"):
        dataset_recording.finish_dataset(_Ctx("job-b"), first, _Writer(), repo_id=REPO, push_to_hub=True,
                                         public=True, episode_labels=[{"success": True}])
    assert hub.sha(REPO) == head
    labels = (hub.repo(REPO) / "meta/armnet_rollouts.jsonl").read_text().splitlines()
    assert [json.loads(l)["job_id"] for l in labels] == ["job-a", "job-c", "job-c"]


def test_open_rejects_a_dataset_of_other_features(hub) -> None:
    _job("job-a", 1, 0, {})
    other = {**FEATURES, "observation.state": {"dtype": "float32", "shape": (12,), "names": None}}
    with pytest.raises(ValueError, match="other features"):
        dataset_recording.open_run_dataset(REPO, other, 20, public=True)


def test_finish_appends_labels_after_the_existing_ones(tmp_path, monkeypatch) -> None:
    """Without LeRobot: labels and frame times of an appended job continue the file."""
    pushed = []
    monkeypatch.setattr(dataset_recording, "push_appended",
                        lambda ctx, dataset, **kw: pushed.append(kw) or "url")
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/armnet_rollouts.jsonl").write_text('{"episode_index": 0}\n{"episode_index": 1}\n')
    dataset = types.SimpleNamespace(
        root=tmp_path, num_episodes=4, armnet_append={"commit": "c7", "first_episode": 2},
        has_pending_frames=lambda: False, finalize=lambda: None)
    url = dataset_recording.finish_dataset(
        _Ctx("j"), dataset, _Writer(), repo_id=REPO, push_to_hub=True,
        episode_labels=[{"success": True}, {"success": False, "episode_index": 3}],
        episode_frame_times=[[0.0], [0.0]], job_labels={"phase": "train"})
    assert url == "url" and pushed == [{"repo_id": REPO, "commit": "c7", "first": 2}]
    rows = [json.loads(l) for l in (tmp_path / "meta/armnet_rollouts.jsonl").read_text().splitlines()]
    assert rows[2:] == [{"job_id": "j", "phase": "train", "episode_index": 2, "success": True},
                        {"job_id": "j", "phase": "train", "episode_index": 3, "success": False}]
    assert [json.loads(l)["episode_index"] for l in (tmp_path / "meta/armnet_frame_times.jsonl").read_text().splitlines()] == [2, 3]
