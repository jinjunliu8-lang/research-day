from __future__ import annotations

import csv
import json
import os
import re
import shlex
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from .config import state_dir
from .util import day_window, iso_now, redact_text, run_command, sha256_file, write_json


FAILURE_RE = re.compile(r"(?i)(traceback|\berror\b|failed|failure|out of memory|\boom\b|killed)")
FIGURE_RE = re.compile(r"(?i)^(fig(?:ure)?(?:\d+|_)|confusion|training|metrics|evaluation|pca)")


def _allowed(path: Path, capture: dict[str, Any], max_bytes: int) -> tuple[bool, str | None]:
    parts = set(path.parts)
    if any(part in parts for part in capture.get("exclude_parts", [])):
        return False, "excluded_path"
    if path.suffix.lower() in {x.lower() for x in capture.get("exclude_extensions", [])}:
        return False, "excluded_extension"
    allowed_extensions = {x.lower() for x in capture.get("allowed_extensions", [])}
    if allowed_extensions and path.suffix.lower() not in allowed_extensions:
        return False, "unsupported_extension"
    try:
        if path.stat().st_size > max_bytes:
            return False, "over_size_limit"
    except OSError:
        return False, "unreadable"
    return True, None


def _iter_local_candidates(root: Path, globs: Iterable[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in globs:
        for path in root.glob(pattern):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            yield path


def _file_record(path: Path, root: Path, timezone: str, source: str = "local") -> dict[str, Any]:
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, ZoneInfo(timezone))
    return {
        "source": source,
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "size_bytes": stat.st_size,
        "modified_at": modified.isoformat(timespec="seconds"),
        "sha256": sha256_file(path),
        "extension": path.suffix.lower(),
    }


def collect_local_files(
    root: Path,
    capture: dict[str, Any],
    start: datetime,
    end: datetime,
    max_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    excluded: dict[str, int] = {}
    for path in _iter_local_candidates(root, capture.get("local_globs", ["**/*"])):
        allowed, reason = _allowed(path, capture, max_bytes)
        if not allowed:
            excluded[reason or "excluded"] = excluded.get(reason or "excluded", 0) + 1
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, start.tzinfo)
        if not (start <= modified < end):
            continue
        records.append(_file_record(path, root, str(start.tzinfo)))
    records.sort(key=lambda item: (item["modified_at"], item["relative_path"]))
    return records, excluded


def _git_info(root: Path, start: datetime, end: datetime) -> dict[str, Any]:
    top = run_command(["git", "-C", str(root), "rev-parse", "--show-toplevel"])
    if not top["ok"]:
        return {"available": False}
    branch = run_command(["git", "-C", str(root), "branch", "--show-current"])
    head = run_command(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = run_command(["git", "-C", str(root), "status", "--short"])
    log = run_command(
        [
            "git",
            "-C",
            str(root),
            "log",
            f"--since={start.isoformat()}",
            f"--until={end.isoformat()}",
            "--pretty=format:%H|%cI|%s",
            "--stat",
        ]
    )
    return {
        "available": True,
        "root": top["stdout"],
        "branch": branch["stdout"] if branch["ok"] else "",
        "head": head["stdout"] if head["ok"] else "",
        "status_short": status["stdout"] if status["ok"] else "",
        "daily_log": log["stdout"] if log["ok"] else "",
    }


def _ssh_base(remote: dict[str, Any]) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={int(remote.get('connect_timeout_seconds', 8))}",
        "-p",
        str(remote["port"]),
        "-l",
        str(remote["user"]),
        str(remote["host"]),
    ]


def check_remote(remote: dict[str, Any]) -> dict[str, Any]:
    if not remote.get("enabled", False):
        return {"enabled": False, "available": False, "message": "remote collection disabled"}
    result = run_command(_ssh_base(remote) + ["true"], timeout=int(remote.get("connect_timeout_seconds", 8)) + 2)
    message = result["stderr"]
    if not result["ok"] and "permission denied" in message.lower():
        message = (
            "SSH key authentication is not configured for this profile. Install a public key for the configured "
            "host/user, then rerun doctor; password login is intentionally unsupported and never stored."
        )
    return {
        "enabled": True,
        "available": bool(result["ok"]),
        "message": "SSH key authentication succeeded" if result["ok"] else message,
    }


def _remote_git_info(remote: dict[str, Any], start: datetime, end: datetime) -> dict[str, Any]:
    root = shlex.quote(str(remote["root"]))
    command = (
        f"if git -C {root} rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
        f"printf 'BRANCH|'; git -C {root} branch --show-current; "
        f"printf 'HEAD|'; git -C {root} rev-parse HEAD; "
        f"printf 'STATUS_BEGIN\\n'; git -C {root} status --short; printf 'STATUS_END\\n'; "
        f"printf 'LOG_BEGIN\\n'; git -C {root} log --since=@{int(start.timestamp())} --until=@{int(end.timestamp())} "
        "--pretty='format:%H|%cI|%s' --stat; printf '\\nLOG_END\\n'; "
        "else printf 'NO_GIT\\n'; fi"
    )
    result = run_command(_ssh_base(remote) + [command], timeout=20)
    if not result["ok"] or result["stdout"].startswith("NO_GIT"):
        return {"available": False}
    text = result["stdout"]
    branch = re.search(r"^BRANCH\|(.*)$", text, re.M)
    head = re.search(r"^HEAD\|(.*)$", text, re.M)
    status = re.search(r"STATUS_BEGIN\n(.*?)\nSTATUS_END", text, re.S)
    log = re.search(r"LOG_BEGIN\n(.*?)\nLOG_END", text, re.S)
    return {
        "available": True,
        "branch": branch.group(1).strip() if branch else "",
        "head": head.group(1).strip() if head else "",
        "status_short": status.group(1).strip() if status else "",
        "daily_log": log.group(1).strip() if log else "",
    }


def _running_processes(local_root: Path, remote: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if remote:
        root = str(remote["root"])
        command = "ps -eo pid=,etime=,args= 2>/dev/null"
        result = run_command(_ssh_base(remote) + [command], timeout=12)
        source = "remote"
    else:
        root = str(local_root)
        result = run_command(["ps", "-axo", "pid=,etime=,command="], timeout=10)
        source = "local"
    if not result["ok"]:
        return []
    rows = []
    for line in result["stdout"].splitlines():
        if root not in line or re.search(r"(?i)(grep|research-day collect)", line):
            continue
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        rows.append({"source": source, "pid": parts[0], "elapsed": parts[1], "command": redact_text(parts[2])[:1000]})
    return rows


def _remote_file_records(
    remote: dict[str, Any], capture: dict[str, Any], start: datetime, end: datetime, max_bytes: int
) -> tuple[list[dict[str, Any]], str | None]:
    remote_root = str(remote["root"]).rstrip("/")
    roots = [f"{remote_root}/{part.strip('/')}" for part in capture.get("remote_roots", [])]
    if not roots:
        roots = [remote_root]
    quoted_roots = " ".join(shlex.quote(path) for path in roots)
    command = (
        f"find {quoted_roots} -type f -newermt '@{int(start.timestamp())}' "
        f"! -newermt '@{int(end.timestamp())}' -printf '%T@|%s|%p\\n' 2>/dev/null"
    )
    result = run_command(_ssh_base(remote) + [command], timeout=30)
    if not result["ok"]:
        return [], result["stderr"] or "remote find failed"
    allowed_extensions = {x.lower() for x in capture.get("allowed_extensions", [])}
    excluded_extensions = {x.lower() for x in capture.get("exclude_extensions", [])}
    excluded_parts = set(capture.get("exclude_parts", []))
    rows: list[dict[str, Any]] = []
    for line in result["stdout"].splitlines():
        try:
            epoch_text, size_text, path_text = line.split("|", 2)
            size = int(size_text)
            path = Path(path_text)
        except (ValueError, TypeError):
            continue
        if path.suffix.lower() in excluded_extensions or path.suffix.lower() not in allowed_extensions:
            continue
        if any(part in excluded_parts for part in path.parts) or size > max_bytes:
            continue
        rel = path_text[len(remote_root) :].lstrip("/") if path_text.startswith(remote_root) else path.name
        rows.append(
            {
                "source": "remote",
                "path": path_text,
                "relative_path": rel,
                "size_bytes": size,
                "modified_at": datetime.fromtimestamp(float(epoch_text), start.tzinfo).isoformat(timespec="seconds"),
                "sha256": None,
                "extension": path.suffix.lower(),
            }
        )
    rows.sort(key=lambda item: (item["modified_at"], item["relative_path"]))
    return rows[: int(capture.get("max_remote_files", 200))], None


def _sync_remote_files(
    records: list[dict[str, Any]], remote: dict[str, Any], destination: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    destination.mkdir(parents=True, exist_ok=True)
    synced: list[dict[str, Any]] = []
    errors: list[str] = []
    for record in records:
        target = destination / record["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        remote_spec = f"{remote['host']}:{record['path']}"
        command = [
            "scp",
            "-q",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={int(remote.get('connect_timeout_seconds', 8))}",
            "-o",
            f"User={remote['user']}",
            "-P",
            str(remote["port"]),
            remote_spec,
            str(target),
        ]
        result = run_command(command, timeout=30)
        if not result["ok"]:
            errors.append(f"{record['relative_path']}: {result['stderr']}")
            continue
        copied = dict(record)
        copied["local_copy"] = str(target)
        copied["sha256"] = sha256_file(target)
        synced.append(copied)
    return synced, errors


def _load_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for record in records:
        name = Path(record["relative_path"]).name
        if name not in {"metrics.json", "summary.json"}:
            continue
        path_text = record.get("local_copy") or record.get("path")
        if not path_text:
            continue
        path = Path(path_text)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            runs.append({"source_file": record["relative_path"], "status": "invalid", "error": redact_text(str(exc))})
            continue
        if name == "summary.json":
            primary = payload.get("primary_test_macro_f1")
            strict = payload.get("strict_test_macro_f1")
            if not isinstance(primary, dict) or not isinstance(strict, dict):
                continue
            runs.append(
                {
                    "source_file": record["relative_path"],
                    "status": "completed",
                    "experiment": payload.get("experiment", path.parent.name),
                    "seed": None,
                    "seeds": payload.get("seeds", []),
                    "aggregate": True,
                    "device": None,
                    "epochs_completed": None,
                    "best_val_macro_f1": None,
                    "parameter_count": None,
                    "inference_ms_per_clip": None,
                    "evaluations": {
                        "primary_test": {
                            "macro_f1": primary.get("mean"),
                            "macro_f1_std": primary.get("std"),
                            "macro_f1_min": primary.get("min"),
                            "macro_f1_max": primary.get("max"),
                            "n": primary.get("n"),
                        },
                        "strict_test": {
                            "macro_f1": strict.get("mean"),
                            "macro_f1_std": strict.get("std"),
                            "macro_f1_min": strict.get("min"),
                            "macro_f1_max": strict.get("max"),
                            "n": strict.get("n"),
                        },
                    },
                    "config": (payload.get("config_audit") or {}).get("normalized_config", {}),
                    "raw": payload,
                }
            )
            continue
        runs.append(
            {
                "source_file": record["relative_path"],
                "status": "completed",
                "experiment": payload.get("experiment", path.parent.name),
                "seed": payload.get("seed"),
                "device": payload.get("device"),
                "epochs_completed": payload.get("epochs_completed"),
                "best_val_macro_f1": payload.get("best_val_macro_f1"),
                "parameter_count": payload.get("head_parameter_count") or payload.get("parameter_count"),
                "inference_ms_per_clip": payload.get("head_inference_ms_per_clip") or payload.get("inference_ms_per_clip"),
                "evaluations": payload.get("evaluations", {}),
                "config": payload.get("config", {}),
                "raw": payload,
            }
        )
    runs.sort(key=lambda item: (not item.get("aggregate", False), item.get("source_file", "")))
    return runs


def _load_histories(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    histories: list[dict[str, Any]] = []
    for record in records:
        name = Path(record["relative_path"]).name.lower()
        if name not in {"history.csv", "training_history.csv", "fig07_training_history.csv"}:
            continue
        path_text = record.get("local_copy") or record.get("path")
        if not path_text:
            continue
        try:
            with Path(path_text).open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            continue
        if rows:
            histories.append({"source_file": record["relative_path"], "rows": rows})
    return histories


def _find_failures(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for record in records:
        if record["extension"] not in {".log", ".txt"}:
            continue
        path_text = record.get("local_copy") or record.get("path")
        if not path_text:
            continue
        try:
            lines = Path(path_text).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        matches = [redact_text(line.strip()) for line in lines if FAILURE_RE.search(line)]
        if matches:
            joined = "\n".join(matches).lower()
            status = "interrupted" if any(token in joined for token in ["keyboardinterrupt", "killed", "sigterm", "cancelled"]) else "failed"
            failures.append(
                {
                    "source_file": record["relative_path"],
                    "status": status,
                    "summary": matches[-1][:500],
                    "evidence": matches[-10:],
                    "cause": "需人工确认；CLI 仅记录日志证据，不自动臆测原因。",
                    "action": "检查完整日志、配置和资源状态后补充处理结论。",
                }
            )
    return failures


def _figure_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    figures: list[dict[str, Any]] = []
    for record in records:
        path = Path(record["relative_path"])
        if record["extension"] == ".png" and FIGURE_RE.search(path.name):
            item = dict(record)
            item["stem"] = path.stem
            figures.append(item)
    figures.sort(key=lambda item: item["relative_path"])
    return figures


def collect_evidence(
    global_config: dict[str, Any],
    project_config: dict[str, Any],
    day,
    *,
    local_only: bool = False,
) -> tuple[dict[str, Any], Path]:
    timezone = global_config.get("timezone", "Asia/Shanghai")
    start, end = day_window(day, timezone)
    local_root = Path(project_config["local_root"]).expanduser().resolve()
    capture = project_config["capture"]
    max_bytes = int(global_config.get("max_sync_mb", 50)) * 1024 * 1024
    local_files, excluded = collect_local_files(local_root, capture, start, end, max_bytes)
    remote_status = {"enabled": False, "available": False, "message": "local-only collection"}
    remote_files: list[dict[str, Any]] = []
    remote_git: dict[str, Any] = {"available": False}
    running = _running_processes(local_root)
    warnings: list[str] = []
    target_state = state_dir(global_config) / "evidence" / project_config["project_id"] / day.isoformat()
    if not local_only and project_config.get("remote", {}).get("enabled", False):
        remote = project_config["remote"]
        remote_status = check_remote(remote)
        if remote_status["available"]:
            remote_git = _remote_git_info(remote, start, end)
            running.extend(_running_processes(local_root, remote))
            remote_files, error = _remote_file_records(remote, capture, start, end, max_bytes)
            if error:
                warnings.append(error)
            elif capture.get("sync_remote_small_files", True):
                remote_files, sync_errors = _sync_remote_files(remote_files, remote, target_state / "remote_files")
                warnings.extend(sync_errors)
        else:
            warnings.append(f"remote unavailable: {remote_status['message']}")
    all_files = local_files + remote_files
    evidence = {
        "schema_version": 1,
        "project_id": project_config["project_id"],
        "project_name": project_config["display_name"],
        "date": day.isoformat(),
        "timezone": timezone,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "collected_at": iso_now(timezone),
        "collection_mode": "local-only" if local_only else "local-and-remote",
        "sources": {
            "local": {"root": str(local_root), "available": local_root.exists()},
            "remote": remote_status,
        },
        "git": {"local": _git_info(local_root, start, end), "remote": remote_git},
        "files": all_files,
        "metrics": _load_metrics(all_files),
        "histories": _load_histories(all_files),
        "failures": _find_failures(all_files),
        "running_processes": running,
        "figures": _figure_records(all_files),
        "warnings": warnings,
        "checks": {
            "selected_files": len(all_files),
            "local_files": len(local_files),
            "remote_files": len(remote_files),
            "excluded": excluded,
            "total_selected_bytes": sum(int(item["size_bytes"]) for item in all_files),
        },
    }
    evidence_path = target_state / "evidence.json"
    write_json(evidence_path, evidence)
    return evidence, evidence_path


def load_evidence(global_config: dict[str, Any], project: str, day) -> tuple[dict[str, Any], Path]:
    path = state_dir(global_config) / "evidence" / project / day.isoformat() / "evidence.json"
    if not path.exists():
        raise FileNotFoundError(f"missing evidence bundle: {path}; run research-day collect first")
    return json.loads(path.read_text(encoding="utf-8")), path
