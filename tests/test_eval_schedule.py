from armnet_rlt.operator_control import eval_schedule_entry

SCHEDULE = [
    {"condition": "id", "count": 30, "brightness": 30},
    {"condition": "rail-low", "count": 5, "rail_range": [0.60, 0.65], "brightness": 30},
    {"condition": "bright5", "count": 5, "brightness": 5},
]


def test_eval_schedule_entry_walks_the_entries_back_to_back() -> None:
    assert eval_schedule_entry(SCHEDULE, 0)["condition"] == "id"
    assert eval_schedule_entry(SCHEDULE, 29)["condition"] == "id"
    assert eval_schedule_entry(SCHEDULE, 30)["condition"] == "rail-low"
    assert eval_schedule_entry(SCHEDULE, 39)["condition"] == "bright5"
    assert eval_schedule_entry(SCHEDULE, 40) is None
    assert eval_schedule_entry(None, 0) is None
