from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import state_dir, vault_dir
from .plotting import prepare_figures
from .util import atomic_write_text, human_bytes, redact_text, sha256_file, write_json


AUTO_START = "<!-- research-day:auto:start -->"
AUTO_END = "<!-- research-day:auto:end -->"
MANUAL_START = "<!-- research-day:manual:start -->"
MANUAL_END = "<!-- research-day:manual:end -->"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _manual_content(existing: str | None) -> str:
    if not existing:
        return "\n在这里补充当天的灵感、感受、临时判断或尚未结构化的思考。\n"
    pattern = re.compile(re.escape(MANUAL_START) + r"(.*?)" + re.escape(MANUAL_END), re.S)
    match = pattern.search(existing)
    return match.group(1) if match else "\n在这里补充当天的灵感、感受、临时判断或尚未结构化的思考。\n"


def _summary(evidence: dict[str, Any]) -> str:
    completed = [item for item in evidence.get("metrics", []) if item.get("status") == "completed"]
    if not completed:
        return f"今日共收集 {evidence['checks']['selected_files']} 份可审计文件；没有发现足以形成新性能结论的完整指标文件。"
    run = completed[0]
    evaluations = run.get("evaluations") or {}
    primary = evaluations.get("primary_test") or next(iter(evaluations.values()), {})
    strict = evaluations.get("strict_test") or {}
    text = f"完成并记录 `{run.get('experiment', 'experiment')}`；主评估集 Macro-F1 = **{_fmt(primary.get('macro_f1'))}**"
    if strict:
        text += f"，严格评估集 Macro-F1 = **{_fmt(strict.get('macro_f1'))}**"
    return text + "。结论仅基于已采集的真实指标文件。"


def _completed_work(evidence: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    code_files = [f for f in evidence.get("files", []) if f.get("extension") in {".py", ".sh", ".yaml", ".yml", ".toml"}]
    if code_files:
        rows.append(f"修改或生成代码/配置文件 **{len(code_files)}** 份。")
    for run in evidence.get("metrics", []):
        if run.get("status") == "completed":
            detail = f"完成实验 `{run.get('experiment')}`"
            if run.get("seed") is not None:
                detail += f"（seed {run['seed']}）"
            if run.get("epochs_completed") is not None:
                detail += f"，训练 {run['epochs_completed']} 个 epoch"
            rows.append(detail + "。")
        else:
            rows.append(f"发现无法解析的指标文件 `{run.get('source_file')}`。")
    if evidence.get("figures"):
        canonical = {
            item.get("stem")
            for item in evidence["figures"]
            if re.match(r"^fig\d+", str(item.get("stem", "")), flags=re.I)
        }
        rows.append(f"发现并校验 **{len(canonical) if canonical else len(evidence['figures'])}** 张已有科研图。")
    if evidence.get("running_processes"):
        rows.append(f"采集时发现 **{len(evidence['running_processes'])}** 个仍在运行的项目进程，已记录到证据包。")
    if not rows:
        rows.append("完成当日项目文件与实验产物盘点。")
    return rows


def _interpretation(evidence: dict[str, Any]) -> list[str]:
    paragraphs: list[str] = []
    completed = next((x for x in evidence.get("metrics", []) if x.get("status") == "completed"), None)
    if not completed:
        return ["当前没有新的完整量化结果，不能据此判断模型性能变化。"]
    evaluations = completed.get("evaluations") or {}
    primary = evaluations.get("primary_test") or next(iter(evaluations.values()), {})
    strict = evaluations.get("strict_test") or {}
    if primary:
        paragraphs.append(
            f"主评估集 Accuracy = {_fmt(primary.get('accuracy'))}，Macro-F1 = {_fmt(primary.get('macro_f1'))}。"
        )
        per_class = primary.get("per_class_f1") or {}
        if per_class:
            easiest = max(per_class, key=per_class.get)
            hardest = min(per_class, key=per_class.get)
            paragraphs.append(
                f"分类 F1 中 `{hardest}` 最低（{_fmt(per_class[hardest])}），`{easiest}` 最高（{_fmt(per_class[easiest])}）；后续应优先检查困难类别的混淆来源。"
            )
    if strict:
        paragraphs.append(
            f"严格评估集 Macro-F1 = {_fmt(strict.get('macro_f1'))}。严格视图用于控制邻近窗口重叠，不等价于外部域测试，不能据此宣称跨地点或跨设备泛化。"
        )
    if completed.get("seed") is not None:
        paragraphs.append("当前若仅包含单一随机种子，应将结果视为 baseline，正式结论需补充多种子均值与离散程度。")
    return paragraphs


def _render_note(
    evidence: dict[str, Any],
    figures: list[dict[str, Any]],
    attachment_rel: Path,
    *,
    hours: float,
    goal: str,
    decisions: list[str],
    next_steps: list[str],
    notes: str,
    manual: str,
) -> str:
    goal = redact_text(goal)
    decisions = [redact_text(item) for item in decisions]
    next_steps = [redact_text(item) for item in next_steps]
    notes = redact_text(notes)
    day = evidence["date"]
    project = evidence["project_name"]
    lines = [
        "---",
        f"title: {day} {project}科研工作日记",
        f"date: {day}",
        f"project: {project}",
        f"work_hours: {hours:g}",
        "status: completed",
        "tags:",
        "  - 科研日记",
        f"  - {project}",
        "  - 实验记录",
        "  - 科研绘图",
        "---",
        "",
        f"# {day} {project}科研工作日记",
        "",
        AUTO_START,
        "",
        "## 1. 今日结论摘要",
        "",
        "> [!summary] 今日结论",
        f"> {_summary(evidence)}",
        "",
        "## 2. 工作时长和客观工作量",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        f"| 实际投入时长 | {hours:g} h |",
        f"| 纳入证据的文件 | {evidence['checks']['selected_files']} |",
        f"| 完整实验指标 | {sum(1 for x in evidence.get('metrics', []) if x.get('status') == 'completed')} |",
        f"| 失败/异常记录 | {len(evidence.get('failures', []))} |",
        f"| 采集时仍在运行的任务 | {len(evidence.get('running_processes', []))} |",
        f"| 科研图 | {len(figures)} |",
        f"| 纳入文件总量 | {human_bytes(int(evidence['checks']['total_selected_bytes']))} |",
        "",
        "## 3. 今日目标",
        "",
        goal or "整理并验证当日科研工作，保留可复现证据和量化结果。",
        "",
        "## 4. 已完成工作及证据",
        "",
    ]
    lines.extend([f"- {item}" for item in _completed_work(evidence)])
    lines.extend(["", "## 5. 科研数据与图表", ""])
    if figures:
        for index, figure in enumerate(figures, start=1):
            png_name = Path(figure["png"]).name
            link = (attachment_rel / png_name).as_posix()
            lines.extend(
                [
                    f"### 图 {index}　{figure['stem']}",
                    "",
                    f"![[{link}|1000]]",
                    "",
                    f"*图 {index}　{figure['caption']}*",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "> [!note] 今日无可合理绘制的新数据",
                "> 当前只有单点、配置值或不完整结果，未为了凑图生成缺乏科研含义的图表。",
                "",
            ]
        )
    lines.extend(["## 6. 结果解释和研究意义", ""])
    lines.extend(_interpretation(evidence))
    if decisions:
        lines.extend(["", "**今日研究决定：**", ""] + [f"- {item}" for item in decisions])
    if notes:
        lines.extend(["", "**补充说明：**", "", notes])
    lines.extend(["", "## 7. 失败实验与负结果", ""])
    if evidence.get("failures"):
        for failure in evidence["failures"]:
            lines.extend(
                [
                    f"- **证据：** `{failure['source_file']}`",
                    f"  - 错误摘要：{failure['summary']}",
                    f"  - 原因判断：{failure['cause']}",
                    f"  - 后续动作：{failure['action']}",
                ]
            )
    else:
        lines.append("- 当日纳入的日志中未检测到明确失败记录；这不代表未采集日志之外不存在失败。")
    lines.extend(["", "## 8. 代码、配置、数据版本和复现记录", ""])
    git = evidence.get("git", {}).get("local", {})
    if git.get("available"):
        lines.extend([f"- Git branch：`{git.get('branch') or 'detached'}`", f"- Git HEAD：`{git.get('head')}`"])
    else:
        lines.append("- 本地项目未检测到 Git 仓库；本日记使用文件时间、大小和 SHA-256 作为退化证据。")
    remote_git = evidence.get("git", {}).get("remote", {})
    if remote_git.get("available"):
        lines.extend(
            [
                f"- 服务器 Git branch：`{remote_git.get('branch') or 'detached'}`",
                f"- 服务器 Git HEAD：`{remote_git.get('head')}`",
            ]
        )
    lines.extend(
        [
            f"- 采集窗口：`{evidence['window']['start']}` 至 `{evidence['window']['end']}`。",
            f"- 采集模式：`{evidence['collection_mode']}`。",
            "",
            "<details>",
            "<summary><strong>展开查看文件级审计记录</strong></summary>",
            "",
            "| 来源 | 相对路径 | 大小 | 修改时间 | SHA-256 |",
            "|---|---|---:|---|---|",
        ]
    )
    for item in evidence.get("files", [])[:100]:
        digest = item.get("sha256") or "未同步"
        lines.append(f"| {item['source']} | `{item['relative_path']}` | {human_bytes(int(item['size_bytes']))} | {item['modified_at']} | `{digest}` |")
    lines.extend(["", "</details>", "", "## 9. 风险、局限和未解决问题", ""])
    risks: list[str] = []
    if evidence["collection_mode"] == "local-only":
        risks.append("本次使用本地只读采集，没有实时连接服务器；结论基于已经同步到本机的实验产物。")
    risks.extend(evidence.get("warnings", []))
    completed = [x for x in evidence.get("metrics", []) if x.get("status") == "completed"]
    if any(x.get("seed") is not None for x in completed) and len(completed) == 1:
        risks.append("当前仅发现一个完整实验结果，尚不能报告多随机种子均值和标准差。")
    if not risks:
        risks.append("未发现阻断性问题；仍需按后续实验计划继续验证稳定性与外部泛化。")
    lines.extend([f"- {risk}" for risk in risks])
    lines.extend(["", "## 10. 明日计划", ""])
    steps = next_steps or ["基于今天的结果继续下一项预注册实验，并保持数据划分和评估规则不变。"]
    lines.extend([f"- {item}" for item in steps])
    lines.extend(["", AUTO_END, "", "## 11. 手写思考区", "", MANUAL_START, manual, MANUAL_END, ""])
    return "\n".join(lines)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        shutil.copy2(source, tmp_name)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _backup_existing(note: Path, attachment_dir: Path, backup_root: Path) -> Path | None:
    if not note.exists() and not attachment_dir.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination = backup_root / stamp
    destination.mkdir(parents=True, exist_ok=False)
    if note.exists():
        shutil.copy2(note, destination / note.name)
    if attachment_dir.exists():
        shutil.copytree(attachment_dir, destination / "attachments")
    return destination


def _validate_wikilinks(note_text: str, vault: Path) -> list[str]:
    missing: list[str] = []
    for target in re.findall(r"!\[\[([^\]|#]+)(?:\|[^\]]+)?\]\]", note_text):
        if not (vault / target).exists():
            missing.append(target)
    return missing


def _update_index(vault: Path, project_folder: str) -> Path:
    journal_root = vault / "工作日记" / project_folder
    notes = sorted(journal_root.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].md"), reverse=True)
    lines = [f"# {project_folder}科研工作日记索引", "", AUTO_START, "", "| 日期 | 日记 |", "|---|---|"]
    for note in notes:
        rel = note.relative_to(vault).with_suffix("").as_posix()
        lines.append(f"| {note.stem} | [[{rel}|打开]] |")
    lines.extend(["", AUTO_END, "", "此索引由 `research-day` 自动维护。", ""])
    index_path = journal_root / "索引.md"
    atomic_write_text(index_path, "\n".join(lines))
    return index_path


def publish_journal(
    global_config: dict[str, Any],
    project_config: dict[str, Any],
    evidence: dict[str, Any],
    *,
    hours: float,
    goal: str = "",
    decisions: list[str] | None = None,
    next_steps: list[str] | None = None,
    notes: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    state = state_dir(global_config)
    project_id = project_config["project_id"]
    day = evidence["date"]
    rendered = state / "rendered" / project_id / day
    figures, manifest_path = prepare_figures(evidence, rendered / "figures")
    vault = vault_dir(global_config)
    project_folder = project_config["vault_folder"]
    year = day[:4]
    note = vault / "工作日记" / project_folder / year / f"{day}.md"
    attachment_rel = Path("附件") / "工作日记" / project_folder / day
    attachment_dir = vault / attachment_rel
    existing = note.read_text(encoding="utf-8") if note.exists() else None
    content = _render_note(
        evidence,
        figures,
        attachment_rel,
        hours=hours,
        goal=goal,
        decisions=decisions or [],
        next_steps=next_steps or [],
        notes=notes,
        manual=_manual_content(existing),
    )
    if dry_run:
        preview = rendered / "preview.md"
        atomic_write_text(preview, content)
        return {"status": "dry-run", "preview": str(preview), "figures": len(figures), "manifest": str(manifest_path)}
    backup = _backup_existing(note, attachment_dir, state / "backups" / project_id / day)
    for source in sorted((rendered / "figures").rglob("*")):
        if source.is_file():
            _atomic_copy(source, attachment_dir / source.relative_to(rendered / "figures"))
    missing = _validate_wikilinks(content, vault)
    if missing:
        raise RuntimeError(f"refusing to publish note with missing attachments: {missing}")
    atomic_write_text(note, content)
    index = _update_index(vault, project_folder)
    publication = {
        "status": "published",
        "note": str(note),
        "index": str(index),
        "attachments": str(attachment_dir),
        "figures": len(figures),
        "backup": str(backup) if backup else None,
        "note_sha256": sha256_file(note),
    }
    write_json(rendered / "publication.json", publication)
    return publication
