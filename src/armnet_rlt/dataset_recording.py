"""Record RLT actor rollouts as private LeRobot evaluation datasets."""

from __future__ import annotations

import logging
import queue
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STREAMING_ENCODING = True
DEFAULT_VCODEC = "libsvtav1"
DEFAULT_ENCODER_THREADS = 2


def generate_rlt_eval_repo_id(ctx: Any) -> str:
    """Return ``<user>/eval_rlt_<task>_<timestamp>`` with second precision."""
    timestamp = datetime.now(UTC).strftime("%Hh%Mm%Ss_%d-%b-%Y").lower()
    hf_user = ctx.args.get("hf_user")
    if not hf_user:
        try:
            from huggingface_hub import whoami

            hf_user = whoami()["name"]
        except Exception:  # noqa: BLE001 - still record locally without Hub auth
            hf_user = "armnet"
    return f"{hf_user}/eval_rlt_{ctx.task}_{timestamp}"


def dataset_features(robot: Any) -> dict[str, Any]:
    from lerobot.datasets.feature_utils import (
        combine_feature_dicts,
        hw_to_dataset_features,
    )
    from lerobot.utils.constants import ACTION, OBS_STR

    return combine_feature_dicts(
        hw_to_dataset_features(robot.action_features, ACTION, use_video=True),
        hw_to_dataset_features(
            robot.observation_features, OBS_STR, use_video=True
        ),
    )


def create_rlt_eval_dataset(
    repo_id: str,
    features: dict[str, Any],
    fps: int,
    *,
    streaming_encoding: bool = DEFAULT_STREAMING_ENCODING,
    vcodec: str = DEFAULT_VCODEC,
    encoder_threads: int = DEFAULT_ENCODER_THREADS,
    num_cameras: int = 1,
) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=str(Path("/tmp/eval_datasets") / repo_id),
        robot_type="so-101",
        features=features,
        use_videos=True,
        streaming_encoding=streaming_encoding,
        vcodec=vcodec,
        encoder_threads=encoder_threads,
        image_writer_processes=0,
        image_writer_threads=0 if streaming_encoding else 4 * max(num_cameras, 1),
    )


def save_episode(
    ctx: Any,
    dataset: Any,
    frame_writer: "DatasetFrameWriter",
    *,
    telemetry: Any = None,
    ticks: int = 0,
) -> bool:
    frame_writer.flush()
    if not dataset.has_pending_frames():
        ctx.cell.discard_episode()
        if telemetry is not None:
            telemetry.discard_episode()
        return False
    episode_index = dataset.num_episodes
    if telemetry is not None:
        telemetry.flush_episode(expected_rows=ticks)
    try:
        dataset.save_episode()
    except Exception:
        if telemetry is not None:
            telemetry.discard_episode(episode_index)
        raise
    ctx.cell.commit_episode(episode_index)
    return True


def finish_dataset(
    ctx: Any,
    dataset: Any,
    frame_writer: "DatasetFrameWriter",
    *,
    repo_id: str,
    push_to_hub: bool,
    telemetry: Any = None,
) -> str | None:
    """Discard an interrupted episode, finalize, and optionally upload."""
    frame_writer.flush()
    if dataset.has_pending_frames():
        dataset.clear_episode_buffer()
        ctx.cell.discard_episode()
    if telemetry is not None:
        telemetry.discard_episode()
    frame_writer.close()
    if telemetry is not None:
        telemetry.close()
    dataset.finalize()
    if not push_to_hub:
        return None
    ctx.report_progress(f"uploading RLT eval dataset to HuggingFace: {repo_id}")
    dataset.push_to_hub(private=True)
    url = f"https://huggingface.co/datasets/{repo_id}"
    ctx.report_progress(f"RLT eval dataset uploaded: {url}")
    return url


class DatasetFrameWriter:
    """Build and append dataset frames outside the control-loop thread."""

    def __init__(
        self,
        dataset: Any,
        *,
        features: dict[str, Any],
        task: str,
    ) -> None:
        self._dataset = dataset
        self._features = features
        self._task = task
        self._queue: queue.Queue = queue.Queue()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="rlt-dataset-frame-writer",
            daemon=True,
        )
        self._thread.start()

    def submit_observation(
        self,
        raw_obs: dict[str, Any],
        sent_action: dict[str, float],
        *,
        task: str | None = None,
    ) -> None:
        self._queue.put((raw_obs, sent_action, task or self._task))

    def flush(self) -> None:
        self._queue.join()
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error

    def _build(
        self,
        raw_obs: dict[str, Any],
        sent_action: dict[str, float],
        task: str,
    ) -> dict[str, Any]:
        from lerobot.datasets.feature_utils import build_dataset_frame
        from lerobot.utils.constants import ACTION, OBS_STR

        return {
            **build_dataset_frame(
                self._features, raw_obs, prefix=OBS_STR
            ),
            **build_dataset_frame(
                self._features, sent_action, prefix=ACTION
            ),
            "task": task,
        }

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self._error is None:
                    raw_obs, sent_action, task = item
                    self._dataset.add_frame(
                        self._build(raw_obs, sent_action, task)
                    )
            except Exception as exc:  # noqa: BLE001 - surfaced by flush()
                self._error = exc
                logger.exception("RLT dataset frame writer failed")
            finally:
                self._queue.task_done()
