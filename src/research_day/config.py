from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .util import atomic_write_text, config_dir


DEFAULT_GLOBAL = {
    "version": 1,
    "timezone": "Asia/Shanghai",
    "vault_path": "~/Documents/Obsidian",
    "state_dir": "~/.local/share/research-day",
    "max_sync_mb": 50,
}


DEFAULT_TOPIC_C = {
    "project_id": "topic-c",
    "display_name": "课题C",
    "vault_folder": "课题C",
    "local_root": "~/research/topic-c",
    "remote": {
        "enabled": False,
        "host": "example.org",
        "port": 22,
        "user": "researcher",
        "root": "/srv/research/topic-c",
        "connect_timeout_seconds": 8,
    },
    "capture": {
        "local_globs": [
            "scripts/**/*.py",
            "birdsong/**/*.py",
            "tests/**/*.py",
            "outputs_topic_c/**/*",
            "*.py",
            "*.md",
        ],
        "remote_roots": ["scripts", "birdsong", "tests", "outputs_topic_c", "data/processed/topic_c_v1"],
        "allowed_extensions": [".py", ".sh", ".yaml", ".yml", ".toml", ".json", ".csv", ".parquet", ".log", ".txt", ".md", ".png", ".pdf"],
        "exclude_parts": [
            ".git",
            ".env",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".cache",
            "dataset",
            "datasets",
            "raw",
            "checkpoints",
        ],
        "exclude_extensions": [".pt", ".pth", ".ckpt", ".npy", ".npz", ".wav", ".flac", ".mp3", ".zip", ".tar", ".gz"],
        "sync_remote_small_files": True,
        "max_remote_files": 200,
    },
}


def _dump_yaml(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)


def initialize(project: str, force: bool = False) -> list[Path]:
    if project != "topic-c":
        raise ValueError("v1 currently initializes only the built-in topic-c profile")
    root = config_dir()
    global_path = root / "config.yaml"
    project_path = root / "projects" / f"{project}.yaml"
    created: list[Path] = []
    for path, payload in [(global_path, DEFAULT_GLOBAL), (project_path, DEFAULT_TOPIC_C)]:
        if path.exists() and not force:
            continue
        atomic_write_text(path, _dump_yaml(payload))
        created.append(path)
    return created


def load_global() -> dict[str, Any]:
    path = config_dir() / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}; run research-day init --project topic-c")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload


def load_project(project: str) -> dict[str, Any]:
    path = config_dir() / "projects" / f"{project}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing project profile: {path}; run research-day init --project {project}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = ["project_id", "display_name", "vault_folder", "local_root", "capture"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"project profile missing fields: {', '.join(missing)}")
    return payload


def load_config(project: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return load_global(), load_project(project)


def state_dir(global_config: dict[str, Any]) -> Path:
    return Path(global_config["state_dir"]).expanduser().resolve()


def vault_dir(global_config: dict[str, Any]) -> Path:
    return Path(global_config["vault_path"]).expanduser().resolve()
