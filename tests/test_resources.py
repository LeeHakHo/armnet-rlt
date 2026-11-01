import pytest

from armnet_rlt.resources import rendezvous_key, task_paths


def test_task_paths_match_historical_volume_layout() -> None:
    cache, assets, output = task_paths("pi05_rlt_busybox_multitask")
    assert cache == (
        "/data/tasks/pi05_rlt_busybox_multitask/demo/rlt_demo_cache.pt"
    )
    assert assets == "/data/tasks/pi05_rlt_busybox_multitask/ckpt/assets"
    assert output == "/data/tasks/pi05_rlt_busybox_multitask/rlt_checkpoints"


def test_rendezvous_is_namespaced_by_config_and_run() -> None:
    assert rendezvous_key("config", "run-1") == "config:run-1"
    with pytest.raises(ValueError):
        rendezvous_key("config", "")
