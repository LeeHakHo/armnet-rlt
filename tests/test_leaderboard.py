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
