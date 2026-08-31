from pathlib import Path

from armnet_rlt.jsonl_log import append_jsonl, read_jsonl


def test_jsonl_records_append_and_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "metrics" / "rollouts.jsonl"
    append_jsonl(path, {"rollout": 1, "success": False})
    append_jsonl(path, {"rollout": 2, "success": True}, sync=True)

    assert read_jsonl(path) == [
        {"rollout": 1, "success": False},
        {"rollout": 2, "success": True},
    ]


def test_missing_jsonl_is_empty(tmp_path: Path) -> None:
    assert read_jsonl(tmp_path / "missing.jsonl") == []
