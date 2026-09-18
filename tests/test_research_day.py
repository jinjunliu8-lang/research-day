from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from research_day.collect import collect_evidence
from research_day.cli import _annotations
from research_day.config import initialize, load_config
from research_day.plotting import prepare_figures
from research_day.publish import MANUAL_END, MANUAL_START, publish_journal
from research_day.util import redact, sha256_file


DAY = "2026-09-17"


def _metrics() -> dict:
    return {
        "experiment": "fixture_teacher",
        "device": "cpu",
        "seed": 42,
        "class_names": ["a", "b"],
        "epochs_completed": 2,
        "head_parameter_count": 10,
        "best_val_macro_f1": 0.8,
        "evaluations": {
            "primary_test": {
                "samples": 4,
                "loss": 0.2,
                "accuracy": 0.75,
                "macro_f1": 0.733,
                "per_class_f1": {"a": 0.8, "b": 0.666},
                "confusion_matrix": [[2, 0], [1, 1]],
            },
            "strict_test": {
                "samples": 4,
                "loss": 0.18,
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "per_class_f1": {"a": 1.0, "b": 1.0},
                "confusion_matrix": [[2, 0], [0, 2]],
            },
        },
    }


def _setup(tmp_path: Path) -> tuple[dict, dict, Path]:
    project_root = tmp_path / "project"
    output = project_root / "outputs" / "run1"
    output.mkdir(parents=True)
    metrics = output / "metrics.json"
    metrics.write_text(json.dumps(_metrics()), encoding="utf-8")
    history = output / "history.csv"
    history.write_text(
        "epoch,train_loss,val_loss,train_macro_f1,val_macro_f1\n1,0.8,0.6,0.6,0.7\n2,0.3,0.4,0.9,0.8\n",
        encoding="utf-8",
    )
    stamp = datetime(2026, 9, 17, 12, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
    os.utime(metrics, (stamp, stamp))
    os.utime(history, (stamp, stamp))
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    global_config = {
        "version": 1,
        "timezone": "Asia/Shanghai",
        "vault_path": str(vault),
        "state_dir": str(tmp_path / "state"),
        "max_sync_mb": 5,
    }
    project_config = {
        "project_id": "fixture",
        "display_name": "测试课题",
        "vault_folder": "测试课题",
        "local_root": str(project_root),
        "remote": {"enabled": False},
        "capture": {
            "local_globs": ["outputs/**/*"],
            "allowed_extensions": [".json", ".csv", ".log", ".png", ".pdf"],
            "exclude_parts": [".env", "raw"],
            "exclude_extensions": [".pt", ".npy", ".wav"],
        },
    }
    return global_config, project_config, vault


def test_init_has_no_secret_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_DAY_CONFIG_DIR", str(tmp_path / "config"))
    created = initialize("topic-c")
    assert len(created) == 2
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in created)
    assert "password:" not in text
    assert "token:" not in text
    global_config, project_config = load_config("topic-c")
    assert global_config["timezone"] == "Asia/Shanghai"
    assert project_config["project_id"] == "topic-c"


def test_collect_parses_metrics_and_history(tmp_path: Path) -> None:
    global_config, project_config, _ = _setup(tmp_path)
    evidence, path = collect_evidence(
        global_config,
        project_config,
        datetime.fromisoformat(DAY).date(),
        local_only=True,
    )
    assert path.exists()
    assert evidence["checks"]["selected_files"] == 2
    assert evidence["metrics"][0]["best_val_macro_f1"] == 0.8
    assert len(evidence["histories"][0]["rows"]) == 2
    assert evidence["sources"]["remote"]["message"] == "local-only collection"


def test_collect_parses_multiseed_summary_and_fig_prefix(tmp_path: Path) -> None:
    global_config, project_config, _ = _setup(tmp_path)
    output = Path(project_config["local_root"]) / "outputs" / "summary"
    output.mkdir()
    summary = output / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "experiment": "teacher_multiseed",
                "seeds": [42, 43, 44],
                "primary_test_macro_f1": {"n": 3, "mean": 0.9, "std": 0.01, "min": 0.89, "max": 0.91},
                "strict_test_macro_f1": {"n": 3, "mean": 0.88, "std": 0.02, "min": 0.86, "max": 0.9},
            }
        ),
        encoding="utf-8",
    )
    figure = output / "fig_teacher_stability.png"
    figure.write_bytes(b"png")
    stamp = datetime(2026, 9, 17, 12, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
    os.utime(summary, (stamp, stamp))
    os.utime(figure, (stamp, stamp))
    evidence, _ = collect_evidence(global_config, project_config, datetime.fromisoformat(DAY).date(), local_only=True)
    assert evidence["metrics"][0]["aggregate"] is True
    assert evidence["metrics"][0]["evaluations"]["primary_test"]["macro_f1"] == 0.9
    assert evidence["figures"][0]["stem"] == "fig_teacher_stability"


def test_collect_excludes_secrets_models_and_large_files(tmp_path: Path) -> None:
    global_config, project_config, _ = _setup(tmp_path)
    output = Path(project_config["local_root"]) / "outputs" / "run1"
    secret = output / ".env"
    model = output / "best.pt"
    large = output / "oversized.csv"
    secret.write_text("PASSWORD=do-not-read", encoding="utf-8")
    model.write_bytes(b"model")
    large.write_bytes(b"0" * (6 * 1024 * 1024))
    stamp = datetime(2026, 9, 17, 12, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
    for path in [secret, model, large]:
        os.utime(path, (stamp, stamp))
    evidence, _ = collect_evidence(global_config, project_config, datetime.fromisoformat(DAY).date(), local_only=True)
    selected = {item["relative_path"] for item in evidence["files"]}
    assert "outputs/run1/.env" not in selected
    assert "outputs/run1/best.pt" not in selected
    assert "outputs/run1/oversized.csv" not in selected
    assert evidence["checks"]["excluded"]["excluded_path"] == 1
    assert evidence["checks"]["excluded"]["excluded_extension"] == 1
    assert evidence["checks"]["excluded"]["over_size_limit"] == 1


def test_asia_shanghai_day_boundary(tmp_path: Path) -> None:
    global_config, project_config, _ = _setup(tmp_path)
    output = Path(project_config["local_root"]) / "outputs" / "run1"
    before_midnight = output / "before.json"
    at_midnight = output / "next_day.json"
    before_midnight.write_text("{}", encoding="utf-8")
    at_midnight.write_text("{}", encoding="utf-8")
    tz = ZoneInfo("Asia/Shanghai")
    included_stamp = datetime(2026, 9, 17, 23, 59, 59, tzinfo=tz).timestamp()
    excluded_stamp = datetime(2026, 9, 18, 0, 0, 0, tzinfo=tz).timestamp()
    os.utime(before_midnight, (included_stamp, included_stamp))
    os.utime(at_midnight, (excluded_stamp, excluded_stamp))
    evidence, _ = collect_evidence(global_config, project_config, datetime.fromisoformat(DAY).date(), local_only=True)
    selected = {item["relative_path"] for item in evidence["files"]}
    assert "outputs/run1/before.json" in selected
    assert "outputs/run1/next_day.json" not in selected


def test_plotting_exports_png_pdf_csv_and_manifest(tmp_path: Path) -> None:
    global_config, project_config, _ = _setup(tmp_path)
    evidence, _ = collect_evidence(global_config, project_config, datetime.fromisoformat(DAY).date(), local_only=True)
    figures, manifest = prepare_figures(evidence, tmp_path / "figures")
    assert {item["stem"] for item in figures} == {"daily_training_curves", "daily_evaluation", "daily_confusion_matrices"}
    assert manifest.exists()
    for item in figures:
        assert Path(item["png"]).exists()
        assert Path(item["pdf"]).exists()
    assert list((tmp_path / "figures" / "figure_data").glob("*.csv"))


def test_dry_run_does_not_write_vault(tmp_path: Path) -> None:
    global_config, project_config, vault = _setup(tmp_path)
    evidence, _ = collect_evidence(global_config, project_config, datetime.fromisoformat(DAY).date(), local_only=True)
    result = publish_journal(global_config, project_config, evidence, hours=3.5, dry_run=True)
    assert result["status"] == "dry-run"
    assert Path(result["preview"]).exists()
    assert not (vault / "工作日记").exists()
    assert not (vault / "附件").exists()


def test_publish_preserves_manual_area_and_creates_backup(tmp_path: Path) -> None:
    global_config, project_config, vault = _setup(tmp_path)
    evidence, _ = collect_evidence(global_config, project_config, datetime.fromisoformat(DAY).date(), local_only=True)
    first = publish_journal(global_config, project_config, evidence, hours=4.0, goal="完成回归实验")
    note = Path(first["note"])
    assert note.exists()
    text = note.read_text(encoding="utf-8")
    replacement = f"{MANUAL_START}\n这是我手写的判断。\n{MANUAL_END}"
    start = text.index(MANUAL_START)
    end = text.index(MANUAL_END) + len(MANUAL_END)
    note.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
    second = publish_journal(global_config, project_config, evidence, hours=5.0, next_steps=["补跑多个随机种子"])
    updated = note.read_text(encoding="utf-8")
    assert "这是我手写的判断。" in updated
    assert "实际投入时长 | 5 h" in updated
    assert second["backup"] is not None
    assert Path(second["backup"]).exists()
    assert Path(second["index"]).exists()
    assert not [line for line in updated.splitlines() if "![[]" in line]


def test_redaction_and_hash(tmp_path: Path) -> None:
    payload = {"password": "danger", "message": "token=abc123 safe", "nested": ["api_key:xyz"]}
    cleaned = redact(payload)
    assert cleaned["password"] == "[REDACTED]"
    assert "abc123" not in cleaned["message"]
    path = tmp_path / "x.txt"
    path.write_text("same", encoding="utf-8")
    assert sha256_file(path) == "0967115f2813a3541eaef77de9d9d5773f1c0c04314b0bbfe4ff3b3b1c55b5d5"


def test_annotation_file_interface(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps({"hours": 6.5, "goal": "完成基线", "decisions": ["固定划分"], "next_steps": ["补跑种子"]}),
        encoding="utf-8",
    )
    payload = _annotations(path)
    assert payload["hours"] == 6.5
    assert payload["decisions"] == ["固定划分"]
