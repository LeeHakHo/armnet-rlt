from armnet_rlt.metrics import EpisodeRecord, RollingMetrics


def test_rolling_metrics_use_last_ten_rollouts_and_successes() -> None:
    rolling = RollingMetrics(window=10)
    for index in range(12):
        rolling.add(
            EpisodeRecord(
                success=index % 2 == 0,
                duration_s=float(index),
                timeout=index == 11,
                action_deviation_mean=float(index) / 10,
            )
        )

    values = rolling.to_dict()
    assert values["rolling/success_rate_10"] == 0.5
    assert values["rolling/mean_success_duration_s_10"] == 5.0
    assert values["rolling/timeout_rate_10"] == 0.1
    assert values["rolling/action_deviation_mean_10"] == 0.65


def test_episode_record_round_trips_sequence_fields() -> None:
    record = EpisodeRecord(
        session_id="session",
        reset_problems=("goal already complete",),
        action_deviation_by_joint=(0.1, 0.2),
        variation={"seed": 42},
    )
    assert EpisodeRecord.from_dict(record.to_dict()) == record
