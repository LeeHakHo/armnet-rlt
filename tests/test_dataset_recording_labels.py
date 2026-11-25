import json
import sys
import types
from pathlib import Path

from armnet_rlt.dataset_recording import finish_dataset


class _Writer:
    def flush(self): pass
    def close(self): pass


class _Dataset:
    def __init__(self, root: Path):
        self.root = root
        self.pushed = None
    def has_pending_frames(self): return False
    def finalize(self): pass
    def push_to_hub(self, private): self.pushed = private


class _Ctx:
    def report_progress(self, message): pass


def test_finish_dataset_writes_labels_and_pushes_public(tmp_path, monkeypatch) -> None:
    uploads = []

    class FakeApi:
        def upload_file(self, **kw): uploads.append(kw)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    dataset = _Dataset(tmp_path)
    labels = [{"success": True, "variation": {"condition": "id"}}, {"success": False}]
    url = finish_dataset(_Ctx(), dataset, _Writer(), repo_id="u/armnet_x", push_to_hub=True,
                         public=True, episode_labels=labels, episode_frame_times=[[0.0, 0.05], [0.0]])
    assert url == "https://huggingface.co/datasets/u/armnet_x"
    assert dataset.pushed is False
    rows = [json.loads(l) for l in (tmp_path / "meta/armnet_rollouts.jsonl").read_text().splitlines()]
    assert rows == [{"episode_index": i, **label} for i, label in enumerate(labels)]
    assert [u["path_in_repo"] for u in uploads] == ["meta/armnet_rollouts.jsonl", "meta/armnet_frame_times.jsonl"]
    times = [json.loads(l) for l in (tmp_path / "meta/armnet_frame_times.jsonl").read_text().splitlines()]
    assert times == [{"episode_index": 0, "frame_times_s": [0.0, 0.05]}, {"episode_index": 1, "frame_times_s": [0.0]}]


def test_finish_dataset_stays_private_by_default(tmp_path) -> None:
    dataset = _Dataset(tmp_path)
    finish_dataset(_Ctx(), dataset, _Writer(), repo_id="u/x", push_to_hub=False)
    assert dataset.pushed is None and not (tmp_path / "meta").exists()
