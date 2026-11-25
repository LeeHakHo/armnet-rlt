import json
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("leaderboard", Path(__file__).resolve().parents[1] / "scripts/leaderboard.py")
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)


def episodes(success_id=5, success_ood=1, seeds=(1042, 2042, 3042)):
    rows = []
    for seed in seeds:
        for i in range(10):
            rows.append({"success": i < success_id, "cell_id": "cell-01", "variation": {"condition": "id", "seed": seed, "values": {"rail": 0.7}, "brightness": 30}})
        for i in range(10):
            rows.append({"success": i < success_ood, "cell_id": "cell-01", "variation": {"condition": "ood-easy" if i < 5 else "ood-hard", "seed": seed, "values": {"rail": 0.62}, "brightness": 15}})
    return rows


def test_check_counts_a_valid_submission(monkeypatch) -> None:
    monkeypatch.setattr(lb, "labels", lambda repo: (episodes(), []))
    sub = {"cell": "cell-01", "evaluations": {str(c): [f"u/armnet_{c}"] for c in lb.CHECKPOINTS}}
    evals, rows = lb.check(sub)
    assert evals[400]["id"] == [15, 30] and evals[400]["ood-easy"][1] + evals[400]["ood-hard"][1] == 30
    assert len(rows) == 4 * 60


def test_check_rejects_a_short_evaluation(monkeypatch) -> None:
    monkeypatch.setattr(lb, "labels", lambda repo: (episodes(seeds=(1042, 2042)), []))
    sub = {"cell": "cell-01", "evaluations": {str(c): ["u/x"] for c in lb.CHECKPOINTS}}
    with pytest.raises(ValueError, match="seed 3042"):
        lb.check(sub)


def test_check_rejects_another_cell(monkeypatch) -> None:
    monkeypatch.setattr(lb, "labels", lambda repo: (episodes(), []))
    with pytest.raises(ValueError, match="cell must be"):
        lb.check({"cell": "cell-09", "evaluations": {}})


def test_build_lists_a_method_once_both_cells_are_in() -> None:
    def evals(i):
        return {c: {"id": [i + c // 100, 30], "ood-easy": [5 - c // 100, 15], "ood-hard": [0, 15], "trips": 0} for c in lb.CHECKPOINTS}
    one = [{"name": "My Method", "cell": "cell-01", "evals": evals(10), "rows": []}]
    assert lb.build(one)[0] == []
    board, _ = lb.build(one + [{"name": "My Method", "cell": "cell-03", "evals": evals(20), "rows": []}])
    assert board[0]["key"] == "my-method"
    assert board[0]["best"]["id"] == [10 + 4 + 20 + 4, 60, 400]
    assert board[0]["best"]["ood"] == [4 + 4, 60, 100]


def test_build_keeps_tagged_entries_of_one_method_apart() -> None:
    def evals(i):
        return {c: {"id": [i, 30], "ood-easy": [1, 15], "ood-hard": [0, 15], "trips": 0} for c in lb.CHECKPOINTS}
    subs = [{"name": "DSRL", "tag": tag, "date": d, "cell": c, "evals": evals(i), "rows": [{"cell": c}]}
            for tag, d, i in (("", "2026-11-18", 10), ("v2", "2026-11-20", 20)) for c in lb.CELLS]
    board, rollouts = lb.build(subs)
    assert sorted((e["key"], e["tag"], e["date"]) for e in board) == [("dsrl", "", "2026-11-18"), ("dsrl--v2", "v2", "2026-11-20")]
    assert {r["method"] for r in rollouts} == {"dsrl", "dsrl--v2"}


def test_a_resubmission_replaces_the_cell() -> None:
    def evals(i):
        return {c: {"id": [i, 30], "ood-easy": [0, 15], "ood-hard": [0, 15], "trips": 0} for c in lb.CHECKPOINTS}
    subs = [{"name": "X", "date": "2026-11-18", "cell": "cell-01", "evals": evals(1), "rows": []},
            {"name": "X", "date": "2026-11-19", "cell": "cell-01", "evals": evals(9), "rows": []},
            {"name": "X", "date": "2026-11-19", "cell": "cell-03", "evals": evals(9), "rows": []}]
    board, _ = lb.build(subs)
    assert board[0]["best"]["id"][0] == 18 and board[0]["date"] == "2026-11-19"


def test_eval_datasets_include_jobs_that_stopped_early(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lb, "DATA", tmp_path)
    out = tmp_path / "eval/run-a-ep100/all"
    out.mkdir(parents=True)
    (out / "job1.log").write_text("[rlt eval] task=x args={'record_dataset_repo_id': 'u/armnet_run-a-ep100_all_0'}\nJobStatus.FAILED\n")
    (out / "job2.log").write_text("[rlt eval] task=x args={'record_dataset_repo_id': 'u/armnet_run-a-ep100_all_13'}\n"
                                  '"eval_dataset_url": "https://huggingface.co/datasets/u/armnet_run-a-ep100_all_13"\n')
    assert lb.eval_datasets("run-a") == {100: ["u/armnet_run-a-ep100_all_0", "u/armnet_run-a-ep100_all_13"]}


def run_labels(train=5):
    """Labels of a run's dataset: training episodes, then each checkpoint's evaluation."""
    rows = [{"phase": "train", "success": True, "variation": {}} for _ in range(train)]
    for c in lb.CHECKPOINTS:
        rows += [{**r, "phase": "eval", "checkpoint_episodes": c, "job": "all"} for r in episodes(success_id=c // 100)]
    return [{**r, "episode_index": i} for i, r in enumerate(rows)]


def spans_of(n):
    key = "videos/observation.images.top"
    return [{"episode_index": i, key + "/chunk_index": 0, key + "/file_index": i // 100,
             key + "/from_timestamp": float(i % 100), key + "/to_timestamp": float(i % 100) + 0.5} for i in range(n)]


def test_check_counts_the_evaluations_of_a_run_dataset(monkeypatch) -> None:
    rows = run_labels()
    monkeypatch.setattr(lb, "labels", lambda repo: (rows, spans_of(len(rows))))
    sub = {"cell": "cell-01", "evaluations": {str(c): ["u/armnet_run-a"] for c in lb.CHECKPOINTS}}
    evals, out = lb.check(sub)
    assert [evals[c]["id"] for c in lb.CHECKPOINTS] == [[3, 30], [6, 30], [9, 30], [12, 30]]
    assert len(out) == 4 * 60
    first = next(r for r in out if r["episodes"] == 200)
    assert first["dataset_episode"] == 5 + 60 and first["dataset"] == "u/armnet_run-a"
    assert "file-000.mp4#t=65.00,65.50" in first["video"]


def test_check_rejects_a_run_dataset_without_a_checkpoint(monkeypatch) -> None:
    rows = [r for r in run_labels() if r.get("checkpoint_episodes") != 300]
    monkeypatch.setattr(lb, "labels", lambda repo: (rows, spans_of(len(rows) + 60)))
    sub = {"cell": "cell-01", "evaluations": {str(c): ["u/armnet_run-a"] for c in lb.CHECKPOINTS}}
    with pytest.raises(ValueError, match="after 300 episodes"):
        lb.check(sub)


@pytest.fixture
def hf(monkeypatch):
    import sys
    import types

    module = types.ModuleType("huggingface_hub")
    module.HfApi = object
    module.snapshot_download = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return module


def test_clips_cover_the_requested_episodes(monkeypatch, hf) -> None:
    old = [{"success": True}] * 3
    monkeypatch.setattr(lb, "labels", lambda repo: (old, spans_of(3)[:3]))
    names = {f"clips/v2/u__old/{i:03d}.mp4" for i in range(3)}
    assert sorted(lb.clips("u/old", names)) == [0, 1, 2]
    rows = run_labels()
    monkeypatch.setattr(lb, "labels", lambda repo: (rows, spans_of(len(rows))))
    names = {f"clips/v2/u__run/{i:03d}.mp4" for i in range(len(rows))}
    assert sorted(lb.clips("u/run", names, {7, 130})) == [7, 130]


def test_clips_download_only_the_video_files_of_missing_episodes(monkeypatch, hf) -> None:
    rows = run_labels()
    monkeypatch.setattr(lb, "labels", lambda repo: (rows, [
        {**s, **{f"videos/observation.images.{v}/{k}": s[f"videos/observation.images.top/{k}"]
                 for v in ("front", "wrist") for k in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")}}
        for s in spans_of(len(rows))]))
    seen = []

    def download(repo, **kw):
        seen.append(kw["allow_patterns"])
        raise StopIteration

    monkeypatch.setattr(hf, "snapshot_download", download)
    with pytest.raises(StopIteration):
        lb.clips("u/run", set(), {130})
    assert seen == [["meta/**", *sorted(f"videos/observation.images.{v}/chunk-000/file-001.mp4" for v in ("top", "front", "wrist"))]]


def test_eval_datasets_list_a_run_dataset_once(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lb, "DATA", tmp_path)
    for ep in (100, 200):
        out = tmp_path / f"eval/run-a-ep{ep}/all"
        out.mkdir(parents=True)
        (out / "job1.log").write_text("args={'record_dataset_repo_id': 'u/armnet_run-a', 'record_append': True}\n"
                                      '"eval_dataset_url": "https://huggingface.co/datasets/u/armnet_run-a"\n')
    assert lb.eval_datasets("run-a") == {100: ["u/armnet_run-a"], 200: ["u/armnet_run-a"]}


def test_labels_read_both_dataset_forms(tmp_path, monkeypatch) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("huggingface_hub")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from fake_hub import FakeHub

    hub = FakeHub(tmp_path / "hub")
    hub.install(monkeypatch)
    for repo, rows in (("u/armnet_run-a-ep100_all_0", episodes()), ("u/armnet_run-a", run_labels())):
        hub.create_repo(repo)
        meta = hub.repo(repo) / "meta"
        (meta / "episodes/chunk-000").mkdir(parents=True)
        (meta / "armnet_rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        pd.DataFrame(spans_of(len(rows))).to_parquet(meta / "episodes/chunk-000/file-000.parquet")
        (hub.repo(repo) / "videos").mkdir()
    old_rows, old_spans = lb.labels("u/armnet_run-a-ep100_all_0")
    new_rows, new_spans = lb.labels("u/armnet_run-a")
    assert len(old_rows) == len(old_spans) == 60 and len(new_rows) == len(new_spans) == 5 + 240
    assert [i for i, _ in lb.evaluation_episodes(old_rows, 100)] == list(range(60))
    assert [i for i, _ in lb.evaluation_episodes(new_rows, 400)] == list(range(5 + 180, 5 + 240))
