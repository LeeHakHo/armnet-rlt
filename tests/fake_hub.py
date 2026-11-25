"""A Hugging Face Hub stand-in for dataset repos in a local directory: create_repo,
repo_info, file_exists, upload_folder (with parent_commit), tags and snapshot_download."""

from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path
from types import SimpleNamespace


class FakeHub:
    def __init__(self, root: Path):
        self.root = root
        self.commits: dict[str, int] = {}
        self.tags: dict[str, dict[str, str]] = {}

    def repo(self, repo_id: str) -> Path:
        return self.root / repo_id

    def sha(self, repo_id: str) -> str:
        return f"c{self.commits[repo_id]}"

    # HfApi
    def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False, **_):
        if repo_id in self.commits:
            if not exist_ok:
                raise FileExistsError(repo_id)
            return
        self.repo(repo_id).mkdir(parents=True)
        self.commits[repo_id] = 0

    def repo_info(self, repo_id, repo_type=None, **_):
        return SimpleNamespace(sha=self.sha(repo_id))

    def repo_exists(self, repo_id, repo_type=None, **_):
        return repo_id in self.commits

    def file_exists(self, repo_id, filename, repo_type=None, revision=None, **_):
        return (self.repo(repo_id) / filename).is_file()

    def upload_folder(self, repo_id, folder_path, repo_type=None, parent_commit=None, ignore_patterns=(), **_):
        if parent_commit is not None and parent_commit != self.sha(repo_id):
            raise RuntimeError(f"412 parent commit {parent_commit} is not the head {self.sha(repo_id)}")
        folder = Path(folder_path)
        for path in folder.rglob("*"):
            rel = str(path.relative_to(folder))
            if path.is_dir() or any(fnmatch.fnmatch(rel, p) for p in ignore_patterns or ()):
                continue
            target = self.repo(repo_id) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        self.commits[repo_id] += 1
        return SimpleNamespace(oid=self.sha(repo_id))

    def delete_tag(self, repo_id, tag, repo_type=None, **_):
        self.tags.get(repo_id, {}).pop(tag, None)

    def create_tag(self, repo_id, tag, revision=None, repo_type=None, **_):
        self.tags.setdefault(repo_id, {})[tag] = revision

    # huggingface_hub.snapshot_download
    def snapshot_download(self, repo_id, repo_type=None, revision=None, local_dir=None, allow_patterns=None,
                          ignore_patterns=None, cache_dir=None, **_):
        patterns = [allow_patterns] if isinstance(allow_patterns, str) else list(allow_patterns or ["*"])
        source = self.repo(repo_id)
        target = Path(local_dir) if local_dir else self.root.parent / "cache" / repo_id
        for path in source.rglob("*"):
            rel = str(path.relative_to(source))
            if path.is_file() and any(rel.startswith(p) if p.endswith("/") else fnmatch.fnmatch(rel, p) for p in patterns):
                out = target / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)
        self.downloads = getattr(self, "downloads", []) + [(repo_id, patterns)]
        return str(target)

    def install(self, monkeypatch) -> None:
        """Route huggingface_hub.HfApi and snapshot_download (also LeRobot's) here."""
        import huggingface_hub

        hub = self
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: hub)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", self.snapshot_download)
        for name in ("lerobot.datasets.dataset_metadata", "lerobot.datasets.lerobot_dataset"):
            try:
                module = __import__(name, fromlist=["snapshot_download"])
            except ImportError:
                continue
            monkeypatch.setattr(module, "snapshot_download", self.snapshot_download)
