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
