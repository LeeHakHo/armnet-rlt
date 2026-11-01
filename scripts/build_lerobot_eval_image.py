"""Build and push the armnet-lerobot-eval image with dict camera configs converted."""

from __future__ import annotations

import argparse
from pathlib import Path

from armnet_client import Image
from armnet_client.lerobot_eval import _IMAGE_NAME, _runtime_build_context


RUNTIME_FILE = "lerobot_eval_runtime.py"
HELPER_ANCHOR = "def _make_robot("
CALL_ANCHOR = (
    "    cameras = _override_camera_dimensions(ctx.camera_configs, policy_features)\n"
)
HELPER = '''_OPENCV_CAMERA_KEYS = {
    "index_or_path",
    "fps",
    "width",
    "height",
    "color_mode",
    "rotation",
    "warmup_s",
    "fourcc",
    "backend",
}


def _lerobot_camera_config(config: Any) -> Any:
    """Return an OpenCVCameraConfig for a dict camera config."""
    if not isinstance(config, dict):
        return config
    from lerobot.cameras.configs import ColorMode, Cv2Rotation
    from lerobot.cameras.opencv import OpenCVCameraConfig

    kwargs = {key: value for key, value in config.items() if key in _OPENCV_CAMERA_KEYS}
    if isinstance(kwargs.get("color_mode"), str):
        kwargs["color_mode"] = ColorMode(kwargs["color_mode"].lower())
    if isinstance(kwargs.get("rotation"), int):
        kwargs["rotation"] = Cv2Rotation(kwargs["rotation"])
    return OpenCVCameraConfig(**kwargs)


'''
CALL = '''    cameras = _override_camera_dimensions(
        {
            name: _lerobot_camera_config(config)
            for name, config in ctx.camera_configs.items()
        },
        policy_features,
    )
'''


def patch_runtime(context_dir: Path) -> None:
    """Insert the camera config conversion into the staged runtime file."""
    path = context_dir / RUNTIME_FILE
    source = path.read_text()
    if source.count(HELPER_ANCHOR) != 1 or source.count(CALL_ANCHOR) != 1:
        raise RuntimeError(f"{RUNTIME_FILE} does not match the expected source")
    source = source.replace(CALL_ANCHOR, CALL)
    source = source.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR)
    compile(source, str(path), "exec")
    path.write_text(source)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embodiment", default="lerobot/so-101")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    with _runtime_build_context(args.embodiment) as context_dir:
        patch_runtime(context_dir)
        dockerfile = context_dir / "Dockerfile"
        if args.dry_run:
            print(Image.planned_ref(dockerfile, _IMAGE_NAME, context_dir))
            return
        image = Image.build(
            dockerfile=dockerfile, context_dir=context_dir, name=_IMAGE_NAME
        ).push()
    print(image.ref)


if __name__ == "__main__":
    main()
