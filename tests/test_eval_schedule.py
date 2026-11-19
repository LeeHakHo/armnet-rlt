from armnet_rlt.operator_control import eval_schedule_entry, tilt_steps

SCHEDULE = [
    {"condition": "id", "seed": 1042, "count": 10, "brightness": 30},
    {"condition": "id", "seed": 2042, "count": 10, "brightness": 30},
    {"condition": "ood-easy", "seed": 1042, "count": 5, "rail_ranges": [[0.60, 0.65], [0.75, 0.80]], "brightness": 15},
]


def test_eval_schedule_entry_walks_the_entries_back_to_back() -> None:
    assert eval_schedule_entry(SCHEDULE, 0) == (SCHEDULE[0], 0)
    assert eval_schedule_entry(SCHEDULE, 9) == (SCHEDULE[0], 9)
    assert eval_schedule_entry(SCHEDULE, 10) == (SCHEDULE[1], 0)
    assert eval_schedule_entry(SCHEDULE, 24) == (SCHEDULE[2], 4)
    assert eval_schedule_entry(SCHEDULE, 25) is None
    assert eval_schedule_entry(None, 0) is None


def test_tilt_steps_spans_the_mount_range() -> None:
    assert tilt_steps(0.0, [992, 3199]) == 992
    assert tilt_steps(1.0, [992, 3199]) == 3199
    assert abs(tilt_steps(0.2567, [992, 3199]) - (992 + 0.2567 * 2207)) < 1e-9
    assert tilt_steps(1.5, [0, 100]) == 100


def test_tally_counts_rollouts_per_condition_and_seed(tmp_path) -> None:
    import json
    import subprocess
    from pathlib import Path

    rows = [
        {"success": True, "variation": {"condition": "id", "seed": 1042}},
        {"success": False, "outcome": "safety_trip", "variation": {"condition": "id", "seed": 1042}},
        {"success": True, "variation": {"condition": "ood-hard", "seed": 2042}},
        {"success": True},
    ]
    (tmp_path / "job.log").write_text(
        "".join(f"[rlt rollout] {json.dumps(row)}\n" for row in rows)
    )
    lib = Path(__file__).resolve().parents[1] / "scripts" / "lib.sh"
    out = subprocess.run(
        ["bash", "-c", f'. "{lib}" >/dev/null; _tally "{tmp_path}" 10 all'],
        capture_output=True, text=True, check=True,
    ).stdout.split("\n")
    assert out[:4] == ["4", "id 1042 2 1 1", "ood-hard 2042 1 1 0", "all -1 1 1 0"]


def test_record_args_follow_record_settings() -> None:
    import subprocess
    from pathlib import Path

    lib = Path(__file__).resolve().parents[1] / "scripts" / "lib.sh"

    def args(env: str) -> str:
        return subprocess.run(
            ["bash", "-c", f'export {env}; . "{lib}" >/dev/null; record_args run_x; echo "${{RECORD_ARGS[*]}}"'],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    assert args("RECORD_REPO=u/armnet") == "--record-dataset --record-dataset-repo-id u/armnet_run_x --record-public"
    assert args("RECORD_REPO=u/armnet RECORD_PUBLIC=0") == "--record-dataset --record-dataset-repo-id u/armnet_run_x"
    assert args("RECORD_REPO=u/armnet RECORD=0") == ""
    assert args("RECORD_REPO=") == ""
