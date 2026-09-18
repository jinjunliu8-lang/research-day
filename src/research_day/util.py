from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SENSITIVE_RE = re.compile(
    r"(?i)(password|passwd|token|api[_-]?key|secret|private[_-]?key)\s*[:=]\s*([^\s,;]+)"
)


def config_dir() -> Path:
    return Path(os.environ.get("RESEARCH_DAY_CONFIG_DIR", "~/.config/research-day")).expanduser()


def parse_day(value: str, timezone: str) -> date:
    if value == "today":
        return datetime.now(ZoneInfo(timezone)).date()
    if value == "yesterday":
        return datetime.now(ZoneInfo(timezone)).date() - timedelta(days=1)
    return date.fromisoformat(value)


def day_window(day: date, timezone: str) -> tuple[datetime, datetime]:
    tz = ZoneInfo(timezone)
    start = datetime.combine(day, time.min, tzinfo=tz)
    return start, start + timedelta(days=1)


def iso_now(timezone: str) -> str:
    return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact_text(value: str) -> str:
    return SENSITIVE_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", value)


def redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if re.search(r"(?i)(password|passwd|token|api[_-]?key|secret|private[_-]?key)", str(key)):
                cleaned[str(key)] = "[REDACTED]"
            else:
                cleaned[str(key)] = redact(item)
        return cleaned
    return value


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(redact(payload), ensure_ascii=False, indent=2) + "\n")


def run_command(args: list[str], cwd: Path | None = None, timeout: int = 20) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": redact_text(completed.stdout.strip()),
            "stderr": redact_text(completed.stderr.strip()),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": redact_text(str(exc))}


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{value} B"
