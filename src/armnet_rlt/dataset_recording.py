"""Record RLT actor rollouts as private LeRobot evaluation datasets."""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STREAMING_ENCODING = True
DEFAULT_VCODEC = "libsvtav1"
DEFAULT_ENCODER_THREADS = 2
RECORDING_ROOT = Path("/tmp/eval_datasets")


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
        root=str(RECORDING_ROOT / repo_id),
        robot_type="so-101",
        features=features,
        use_videos=True,
        streaming_encoding=streaming_encoding,
        vcodec=vcodec,
        encoder_threads=encoder_threads,
        image_writer_processes=0,
        image_writer_threads=_image_writer_threads(streaming_encoding, num_cameras),
    )


def _image_writer_threads(streaming_encoding: bool, num_cameras: int) -> int:
    return 0 if streaming_encoding else 4 * max(num_cameras, 1)


def check_features(repo_id: str, recorded: dict[str, Any], features: dict[str, Any]) -> None:
    """Raises unless the dataset has the robot's features."""
    for key, feature in features.items():
        old = recorded.get(key)
        if old is None or old.get("dtype") != feature.get("dtype") or tuple(old.get("shape") or ()) != tuple(feature.get("shape") or ()):
            raise ValueError(
                f"{repo_id} holds other features than this robot ({key}); record to another dataset"
            )


def open_run_dataset(
    repo_id: str,
    features: dict[str, Any],
    fps: int,
    *,
    public: bool,
    streaming_encoding: bool = DEFAULT_STREAMING_ENCODING,
    vcodec: str = DEFAULT_VCODEC,
    encoder_threads: int = DEFAULT_ENCODER_THREADS,
    num_cameras: int = 1,
) -> Any:
    """The run's dataset on the Hub, resumed at its current commit (meta/ only is
    downloaded), or a new one when the repo has none. ``finish_dataset`` appends the new
    episodes in one commit on top of that commit."""
    from huggingface_hub import HfApi
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=not public, exist_ok=True)
    commit = api.repo_info(repo_id, repo_type="dataset").sha
    root = RECORDING_ROOT / repo_id
    shutil.rmtree(root, ignore_errors=True)
    if api.file_exists(repo_id, "meta/info.json", repo_type="dataset", revision=commit):
        dataset = LeRobotDataset.resume(
            repo_id,
            root=root,
            revision=commit,
            vcodec=vcodec,
            streaming_encoding=streaming_encoding,
            encoder_threads=encoder_threads,
            image_writer_threads=_image_writer_threads(streaming_encoding, num_cameras),
        )
        check_features(repo_id, dataset.meta.features, features)
    else:
        dataset = create_rlt_eval_dataset(
            repo_id,
            features,
            fps,
            streaming_encoding=streaming_encoding,
            vcodec=vcodec,
            encoder_threads=encoder_threads,
            num_cameras=num_cameras,
        )
    dataset.armnet_append = {"commit": commit, "first_episode": int(dataset.num_episodes)}
    return dataset


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
    public: bool = False,
    episode_labels: list[dict[str, Any]] | None = None,
    episode_frame_times: list[list[float]] | None = None,
    job_labels: dict[str, Any] | None = None,
) -> str | None:
    """Discard an interrupted episode, finalize, and optionally upload.

    ``episode_labels`` (one dict per saved episode, in episode order, with
    ``episode_index``, ``job_id`` and ``job_labels`` added where missing) is written to
    ``meta/armnet_rollouts.jsonl`` and ``episode_frame_times`` (per episode, each frame's
    time in seconds since the episode's first frame) to ``meta/armnet_frame_times.jsonl``,
    next to the dataset and uploaded with it. A dataset from ``open_run_dataset`` appends
    to both files and to the Hub repo.
    """
    append = getattr(dataset, "armnet_append", None)
    first = int(append["first_episode"]) if append else 0
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
    sidecars = []
    if episode_labels is not None:
        job_id = getattr(ctx, "job_id", None)
        job = {**({"job_id": str(job_id)} if job_id else {}), **(job_labels or {})}
        rows = [{**job, "episode_index": first + i, **row} for i, row in enumerate(episode_labels)]
        sidecars.append(("armnet_rollouts.jsonl", [json.dumps(row, default=str) for row in rows]))
    if episode_frame_times is not None:
        sidecars.append(("armnet_frame_times.jsonl", [
            json.dumps({"episode_index": first + i, "frame_times_s": times})
            for i, times in enumerate(episode_frame_times)
        ]))
    paths = []
    for name, lines in sidecars:
        path = Path(dataset.root) / "meta" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        old = path.read_text() if append and path.is_file() else ""
        if old and not old.endswith("\n"):
            old += "\n"
        path.write_text(old + "".join(line + "\n" for line in lines))
        paths.append(path)
    if not push_to_hub:
        return None
    if append:
        return push_appended(ctx, dataset, repo_id=repo_id, commit=append["commit"], first=first)
    ctx.report_progress(f"uploading RLT eval dataset to HuggingFace: {repo_id}")
    dataset.push_to_hub(private=not public)
    if paths:
        from huggingface_hub import HfApi

        for path in paths:
            HfApi().upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"meta/{path.name}",
                repo_id=repo_id,
                repo_type="dataset",
            )
    url = f"https://huggingface.co/datasets/{repo_id}"
    ctx.report_progress(f"RLT eval dataset uploaded: {url}")
    return url


def push_appended(ctx: Any, dataset: Any, *, repo_id: str, commit: str, first: int) -> str | None:
    """Upload the episodes recorded since ``first`` in one commit whose parent is
    ``commit``; fails without writing anything when the repo moved since."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    from lerobot.datasets.utils import create_lerobot_dataset_card

    url = f"https://huggingface.co/datasets/{repo_id}"
    last = int(dataset.num_episodes) - 1
    if last < first:
        ctx.report_progress(f"no new episodes for {repo_id}")
        return None
    root = Path(dataset.root)
    create_lerobot_dataset_card(
        dataset_info=dataset.meta.info, license="apache-2.0", repo_id=repo_id
    ).save(str(root / "README.md"))
    api = HfApi()
    ctx.report_progress(f"uploading episodes {first}-{last} to {repo_id}")
    try:
        info = api.upload_folder(
            repo_id=repo_id,
            folder_path=str(root),
            repo_type="dataset",
            ignore_patterns=["images/**", ".cache/**"],
            parent_commit=commit,
            commit_message=f"Add episodes {first}-{last}",
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not add episodes {first}-{last} to {repo_id} (another job may have "
            f"recorded to it since this one started); the local copy is at {root}: {exc}"
        ) from exc
    # LeRobotDataset(repo_id) reads the commit this tag points to.
    with contextlib.suppress(RevisionNotFoundError):
        api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=info.oid, repo_type="dataset")
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
