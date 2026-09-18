from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .collect import check_remote, collect_evidence, load_evidence
from .config import initialize, load_config, vault_dir
from .publish import publish_journal
from .util import parse_day


app = typer.Typer(no_args_is_help=True, help="Generate evidence-backed daily research journals for Obsidian.")
console = Console()


def _hours(value: Optional[float]) -> float:
    if value is not None:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise typer.BadParameter("hours must be a number") from exc
        if result < 0:
            raise typer.BadParameter("hours must be non-negative")
        return result
    if sys.stdin.isatty():
        result = float(typer.prompt("Actual research hours"))
        if result < 0:
            raise typer.BadParameter("hours must be non-negative")
        return result
    raise typer.BadParameter("--hours is required in non-interactive mode; work time is never estimated")


def _show_result(payload: dict) -> None:
    console.print_json(json.dumps(payload, ensure_ascii=False))


def _annotations(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"invalid annotation file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("annotation file must contain a JSON object")
    return payload


def _check_publishable(evidence: dict, local_only: bool = False) -> None:
    remote = evidence.get("sources", {}).get("remote", {})
    if not local_only and remote.get("enabled") and not remote.get("available"):
        raise typer.BadParameter(
            "remote evidence is unavailable; refusing to publish a partial journal. Configure SSH keys or use --local-only explicitly."
        )


@app.command("init")
def init_command(
    project: str = typer.Option("topic-c", "--project", help="Project profile to initialize."),
    force: bool = typer.Option(False, "--force", help="Replace existing generated config files."),
) -> None:
    """Create global and project configuration without storing credentials."""
    try:
        created = initialize(project, force=force)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if created:
        for path in created:
            console.print(f"[green]created[/green] {path}")
    else:
        console.print("[yellow]configuration already exists; nothing changed[/yellow]")


@app.command("doctor")
def doctor_command(
    project: str = typer.Option("topic-c", "--project"),
    local_only: bool = typer.Option(False, "--local-only", help="Skip the SSH connectivity check."),
) -> None:
    """Run read-only environment, dependency, vault, and SSH checks."""
    try:
        global_config, project_config = load_config(project)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    checks: list[tuple[str, bool, str]] = []
    local_root = Path(project_config["local_root"]).expanduser()
    checks.append(("Local project", local_root.is_dir(), str(local_root)))
    vault = vault_dir(global_config)
    checks.append(("Obsidian vault", vault.is_dir() and (vault / ".obsidian").is_dir(), str(vault)))
    for module in ["yaml", "jinja2", "matplotlib", "pandas", "numpy", "typer", "rich"]:
        try:
            importlib.import_module(module)
            checks.append((f"Python: {module}", True, "available"))
        except ImportError as exc:
            checks.append((f"Python: {module}", False, str(exc)))
    profile_text = (Path.home() / ".config" / "research-day" / "projects" / f"{project}.yaml")
    if profile_text.exists():
        lowered = profile_text.read_text(encoding="utf-8").lower()
        has_secret_key = any(key in lowered for key in ["password:", "token:", "api_key:", "private_key:"])
        checks.append(("No stored secrets", not has_secret_key, "profile contains no password/token/private-key fields" if not has_secret_key else "remove secret fields"))
    if local_only:
        checks.append(("SSH key", True, "skipped by --local-only"))
    else:
        status = check_remote(project_config.get("remote", {}))
        checks.append(("SSH key", bool(status.get("available")), status.get("message", "")))
    table = Table(title=f"research-day doctor: {project}")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for name, ok, detail in checks:
        table.add_row(name, "[green]PASS[/green]" if ok else "[red]FAIL[/red]", detail)
    console.print(table)
    if not all(ok for _, ok, _ in checks):
        raise typer.Exit(1)


@app.command("collect")
def collect_command(
    project: str = typer.Option("topic-c", "--project"),
    date_value: str = typer.Option("today", "--date"),
    local_only: bool = typer.Option(False, "--local-only", help="Collect only already-synced local evidence."),
) -> None:
    """Collect read-only project evidence into a versioned JSON bundle."""
    try:
        global_config, project_config = load_config(project)
        day = parse_day(date_value, global_config.get("timezone", "Asia/Shanghai"))
        evidence, path = collect_evidence(global_config, project_config, day, local_only=local_only)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _show_result(
        {
            "status": "collected",
            "evidence": str(path),
            "files": evidence["checks"]["selected_files"],
            "metrics": len(evidence.get("metrics", [])),
            "figures": len(evidence.get("figures", [])),
            "remote": evidence["sources"]["remote"],
            "warnings": evidence.get("warnings", []),
        }
    )


@app.command("publish")
def publish_command(
    project: str = typer.Option("topic-c", "--project"),
    date_value: str = typer.Option("today", "--date"),
    hours: Optional[float] = typer.Option(None, "--hours"),
    goal: str = typer.Option("", "--goal"),
    decision: Optional[list[str]] = typer.Option(None, "--decision"),
    next_step: Optional[list[str]] = typer.Option(None, "--next-step"),
    notes: str = typer.Option("", "--notes"),
    annotation_file: Optional[Path] = typer.Option(None, "--annotation-file", exists=True, dir_okay=False),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Publish a collected evidence bundle to Obsidian."""
    try:
        global_config, project_config = load_config(project)
        day = parse_day(date_value, global_config.get("timezone", "Asia/Shanghai"))
        evidence, _ = load_evidence(global_config, project, day)
        _check_publishable(evidence, local_only=evidence.get("collection_mode") == "local-only")
        annotation = _annotations(annotation_file)
        result = publish_journal(
            global_config,
            project_config,
            evidence,
            hours=_hours(hours if hours is not None else annotation.get("hours")),
            goal=goal or str(annotation.get("goal", "")),
            decisions=decision or list(annotation.get("decisions", [])),
            next_steps=next_step or list(annotation.get("next_steps", [])),
            notes=notes or str(annotation.get("notes", "")),
            dry_run=dry_run,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _show_result(result)


@app.command("close")
def close_command(
    project: str = typer.Option("topic-c", "--project"),
    date_value: str = typer.Option("today", "--date"),
    hours: Optional[float] = typer.Option(None, "--hours"),
    goal: str = typer.Option("", "--goal"),
    decision: Optional[list[str]] = typer.Option(None, "--decision"),
    next_step: Optional[list[str]] = typer.Option(None, "--next-step"),
    notes: str = typer.Option("", "--notes"),
    annotation_file: Optional[Path] = typer.Option(None, "--annotation-file", exists=True, dir_okay=False),
    local_only: bool = typer.Option(False, "--local-only", help="Explicitly use already-synced local evidence."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Collect evidence, build figures, validate, and publish one daily journal."""
    try:
        global_config, project_config = load_config(project)
        day = parse_day(date_value, global_config.get("timezone", "Asia/Shanghai"))
        evidence, evidence_path = collect_evidence(global_config, project_config, day, local_only=local_only)
        _check_publishable(evidence, local_only=local_only)
        annotation = _annotations(annotation_file)
        result = publish_journal(
            global_config,
            project_config,
            evidence,
            hours=_hours(hours if hours is not None else annotation.get("hours")),
            goal=goal or str(annotation.get("goal", "")),
            decisions=decision or list(annotation.get("decisions", [])),
            next_steps=next_step or list(annotation.get("next_steps", [])),
            notes=notes or str(annotation.get("notes", "")),
            dry_run=dry_run,
        )
        result["evidence"] = str(evidence_path)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _show_result(result)


if __name__ == "__main__":
    app()
