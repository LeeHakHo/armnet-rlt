from types import SimpleNamespace

from armnet_rlt.dataset_recording import (
    finish_dataset,
    generate_rlt_eval_repo_id,
    save_episode,
)


class FakeCell:
    def __init__(self) -> None:
        self.calls = []

    def commit_episode(self, index):
        self.calls.append(("commit", index))

    def discard_episode(self):
        self.calls.append(("discard",))


class FakeDataset:
    def __init__(self, pending=True) -> None:
        self.pending = pending
        self.num_episodes = 2
        self.calls = []

    def has_pending_frames(self):
        return self.pending

    def save_episode(self):
        self.calls.append(("save",))
        self.pending = False

    def clear_episode_buffer(self):
        self.calls.append(("clear",))
        self.pending = False

    def finalize(self):
        self.calls.append(("finalize",))

    def push_to_hub(self, **kwargs):
        self.calls.append(("push", kwargs))


class FakeWriter:
    def __init__(self) -> None:
        self.calls = []

    def flush(self):
        self.calls.append(("flush",))

    def close(self):
        self.calls.append(("close",))


class FakeTelemetry:
    def __init__(self) -> None:
        self.calls = []

    def flush_episode(self, **kwargs):
        self.calls.append(("flush_episode", kwargs))

    def discard_episode(self, *args):
        self.calls.append(("discard_episode", args))

    def close(self):
        self.calls.append(("close",))


def test_generated_dataset_name_identifies_rlt() -> None:
    ctx = SimpleNamespace(
        task="push_green_button",
        args={"hf_user": "villekuosmanen"},
    )
    repo_id = generate_rlt_eval_repo_id(ctx)

    assert repo_id.startswith(
        "villekuosmanen/eval_rlt_push_green_button_"
    )


def test_completed_episode_is_saved_with_cell_sidecars() -> None:
    cell = FakeCell()
    ctx = SimpleNamespace(cell=cell)
    dataset = FakeDataset()
    writer = FakeWriter()
    telemetry = FakeTelemetry()

    assert save_episode(
        ctx,
        dataset,
        writer,
        telemetry=telemetry,
        ticks=123,
    )
    assert dataset.calls == [("save",)]
    assert cell.calls == [("commit", 2)]
    assert telemetry.calls == [
        ("flush_episode", {"expected_rows": 123})
    ]


def test_finish_discards_partial_episode_and_pushes_private() -> None:
    cell = FakeCell()
    ctx = SimpleNamespace(
        cell=cell, report_progress=lambda _message: None
    )
    dataset = FakeDataset()
    writer = FakeWriter()
    telemetry = FakeTelemetry()
    url = finish_dataset(
        ctx,
        dataset,
        writer,
        repo_id="villekuosmanen/eval_rlt_task_run",
        push_to_hub=True,
        telemetry=telemetry,
    )

    assert ("clear",) in dataset.calls
    assert ("finalize",) in dataset.calls
    assert ("push", {"private": True}) in dataset.calls
    assert cell.calls == [("discard",)]
    assert writer.calls == [("flush",), ("close",)]
    assert telemetry.calls == [
        ("discard_episode", ()),
        ("close",),
    ]
    assert url == (
        "https://huggingface.co/datasets/"
        "villekuosmanen/eval_rlt_task_run"
    )
