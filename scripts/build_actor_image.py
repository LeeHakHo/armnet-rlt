"""Build and push the RLT actor image and print its registry ref.

The build context is staged from Dockerfile.actor, pyproject.toml, README.md,
src/ and third_party/lerobot/, so the ref depends only on those files.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from armnet_client import Image
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_FILES = ("Dockerfile.actor", "pyproject.toml", "README.md")


def stage_context(context_dir: Path) -> Path:
    for name in CONTEXT_FILES:
        shutil.copy2(ROOT / name, context_dir / name)
    for name in ("src", "third_party/lerobot"):
        shutil.copytree(
            ROOT / name,
            context_dir / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
    return context_dir / "Dockerfile.actor"


def main(argv: list[str] | None = None) -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-name", default="armnet-rlt-actor")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ref a build and push would produce, without Docker",
    )
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        context_dir = Path(tmp)
        dockerfile = stage_context(context_dir)
        if args.dry_run:
            print(Image.planned_ref(dockerfile, args.image_name, context_dir))
            return
        image = Image.build(
            dockerfile=dockerfile, context_dir=context_dir, name=args.image_name
        ).push()
    print(image.ref)


if __name__ == "__main__":
    main()
