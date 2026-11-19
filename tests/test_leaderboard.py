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
