from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .util import atomic_write_text, sha256_file


INK = "#252525"
GRID = "#D9D9D9"
PRIMARY = "#4477AA"
SECONDARY = "#EE6677"
CLASS_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

CAPTIONS = {
    "fig01_dataset_inventory": "数据资产规模：HSN 监督、严格、无标签、背景池及 POW/UHH 保留测试集。",
    "fig02_class_balance": "HSN5 主视图与严格视图在训练、验证和测试划分中的类别平衡。",
    "fig03_source_partition": "源录音划分与交叉检查；非对角线为 0 表示 split 间无源录音交集。",
    "fig04_audio_quality": "音频 RMS、峰值和保留测试集削波比例。",
    "fig05_birdset_label_structure": "POW/UHH 空标签、单标签和多标签片段构成。",
    "fig06_perch_embedding_pca": "Perch 2.0 embedding 的 PCA 描述性投影；该图不单独证明类别可分性。",
    "fig07_teacher_training_curves": "冻结 Perch embedding 上线性教师头的训练与验证轨迹。",
    "fig08_teacher_evaluation": "教师模型在主测试集和严格测试集上的总体指标与分类 F1。",
    "fig09_teacher_confusion_matrices": "教师模型在主测试集和严格测试集上的混淆矩阵。",
    "daily_training_curves": "当日实验的训练和验证曲线。",
    "daily_evaluation": "当日实验的总体指标与分类 F1。",
    "daily_confusion_matrices": "当日实验的混淆矩阵。",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axes(ax: plt.Axes, axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> tuple[Path, Path]:
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return png, pdf


def _copy_existing_figures(evidence: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    candidates = []
    for item in evidence.get("figures", []):
        source_text = item.get("local_copy") or item.get("path")
        if not source_text:
            continue
        source = Path(source_text)
        if source.exists() and source.stem.lower().startswith("fig") and source.stem[3:5].isdigit():
            candidates.append((source.stem, source))
    selected: dict[str, Path] = {}
    for stem, source in sorted(candidates):
        selected.setdefault(stem, source)
    figures: list[dict[str, Any]] = []
    copied_data_dirs: set[Path] = set()
    data_output = output / "figure_data"
    for stem, source in selected.items():
        png_target = output / source.name
        shutil.copy2(source, png_target)
        pdf_source = source.with_suffix(".pdf")
        pdf_target = None
        if pdf_source.exists():
            pdf_target = output / pdf_source.name
            shutil.copy2(pdf_source, pdf_target)
        source_data = source.parent / "figure_data"
        if source_data.is_dir() and source_data not in copied_data_dirs:
            data_output.mkdir(parents=True, exist_ok=True)
            for csv_path in source_data.glob("*.csv"):
                shutil.copy2(csv_path, data_output / csv_path.name)
            copied_data_dirs.add(source_data)
        figures.append(
            {
                "stem": stem,
                "png": str(png_target),
                "pdf": str(pdf_target) if pdf_target else None,
                "caption": CAPTIONS.get(stem, stem),
                "origin": "verified-existing",
            }
        )
    return figures


def _plot_history(history: dict[str, Any], output: Path) -> dict[str, Any] | None:
    frame = pd.DataFrame(history.get("rows", []))
    if frame.empty or "epoch" not in frame:
        return None
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().all():
            frame[column] = converted
    data_path = output / "figure_data" / "daily_training_curves.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_path, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    epoch = pd.to_numeric(frame["epoch"])
    plotted = False
    for column, color, label in [("train_loss", PRIMARY, "Train"), ("val_loss", SECONDARY, "Validation")]:
        if column in frame:
            axes[0].plot(epoch, pd.to_numeric(frame[column]), color=color, linewidth=1.8, label=label)
            plotted = True
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    clean_axes(axes[0])
    if plotted:
        axes[0].legend(frameon=False)
    plotted = False
    for column, color, label in [("train_macro_f1", PRIMARY, "Train"), ("val_macro_f1", SECONDARY, "Validation")]:
        if column in frame:
            axes[1].plot(epoch, pd.to_numeric(frame[column]), color=color, linewidth=1.8, label=label)
            plotted = True
    axes[1].set_title("Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    clean_axes(axes[1])
    if plotted:
        axes[1].legend(frameon=False)
    fig.suptitle("Daily training trajectory", fontsize=13, fontweight="bold")
    fig.tight_layout()
    png, pdf = save_figure(fig, output, "daily_training_curves")
    return {"stem": "daily_training_curves", "png": str(png), "pdf": str(pdf), "caption": CAPTIONS["daily_training_curves"], "origin": "generated"}


def _plot_evaluation(run: dict[str, Any], output: Path) -> dict[str, Any] | None:
    evaluations = run.get("evaluations") or {}
    if not evaluations:
        return None
    overall_rows = []
    class_rows = []
    for split, payload in evaluations.items():
        if not isinstance(payload, dict):
            continue
        overall_rows.append({"split": split, "accuracy": payload.get("accuracy"), "macro_f1": payload.get("macro_f1"), "loss": payload.get("loss")})
        for label, value in (payload.get("per_class_f1") or {}).items():
            class_rows.append({"split": split, "class": label, "f1": value})
    overall = pd.DataFrame(overall_rows)
    per_class = pd.DataFrame(class_rows)
    if overall.empty:
        return None
    data_dir = output / "figure_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(data_dir / "daily_evaluation_overall.csv", index=False)
    per_class.to_csv(data_dir / "daily_evaluation_per_class.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), gridspec_kw={"width_ratios": [0.8, 1.35]})
    metrics = [m for m in ["accuracy", "macro_f1"] if m in overall and overall[m].notna().any()]
    x = np.arange(len(metrics))
    width = 0.7 / max(1, len(overall))
    for idx, row in overall.reset_index(drop=True).iterrows():
        vals = [float(row[m]) for m in metrics]
        split_label = str(row["split"]).replace("_", " ").title()
        axes[0].bar(x + (idx - (len(overall) - 1) / 2) * width, vals, width, label=split_label, color=[PRIMARY, SECONDARY, "#228833"][idx % 3])
    axes[0].set_xticks(x, [m.replace("_", " ").title() for m in metrics])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Overall performance")
    axes[0].legend(frameon=False)
    clean_axes(axes[0])
    if not per_class.empty:
        labels = list(dict.fromkeys(per_class["class"].tolist()))
        x2 = np.arange(len(labels))
        for idx, split in enumerate(dict.fromkeys(per_class["split"].tolist())):
            subset = per_class[per_class["split"] == split].set_index("class")
            vals = [float(subset.loc[label, "f1"]) if label in subset.index else np.nan for label in labels]
            split_label = str(split).replace("_", " ").title()
            axes[1].plot(x2, vals, marker="o", linewidth=1.5, label=split_label, color=[PRIMARY, SECONDARY, "#228833"][idx % 3])
        axes[1].set_xticks(x2, labels, rotation=25, ha="right")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel("F1 score")
        axes[1].set_title("Per-class F1")
        axes[1].legend(frameon=False)
        clean_axes(axes[1])
    fig.suptitle("Daily experiment evaluation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    png, pdf = save_figure(fig, output, "daily_evaluation")
    return {"stem": "daily_evaluation", "png": str(png), "pdf": str(pdf), "caption": CAPTIONS["daily_evaluation"], "origin": "generated"}


def _plot_confusions(run: dict[str, Any], output: Path) -> dict[str, Any] | None:
    matrices = []
    class_names = (run.get("raw") or {}).get("class_names") or []
    for split, payload in (run.get("evaluations") or {}).items():
        matrix = payload.get("confusion_matrix") if isinstance(payload, dict) else None
        if matrix:
            matrices.append((split, np.asarray(matrix, dtype=float)))
    if not matrices:
        return None
    data_dir = output / "figure_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.0 * len(matrices), 4.3), squeeze=False)
    image = None
    for ax, (split, matrix) in zip(axes[0], matrices):
        totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals != 0)
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        labels = class_names if len(class_names) == matrix.shape[0] else [str(i) for i in range(matrix.shape[0])]
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(str(split).replace("_", " ").title())
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{int(matrix[i, j])}\n{normalized[i, j]:.0%}", ha="center", va="center", fontsize=7, color="white" if normalized[i, j] > 0.55 else INK)
        pd.DataFrame(matrix.astype(int), index=labels, columns=labels).to_csv(data_dir / f"daily_confusion_{split}.csv", index_label="true_label")
    if image is not None:
        fig.colorbar(image, ax=list(axes[0]), fraction=0.025, pad=0.03, label="Row-normalized proportion")
    fig.suptitle("Daily confusion matrices", fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.22, wspace=0.34)
    png, pdf = save_figure(fig, output, "daily_confusion_matrices")
    return {"stem": "daily_confusion_matrices", "png": str(png), "pdf": str(pdf), "caption": CAPTIONS["daily_confusion_matrices"], "origin": "generated"}


def prepare_figures(evidence: dict[str, Any], output: Path) -> tuple[list[dict[str, Any]], Path]:
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    figures = _copy_existing_figures(evidence, output)
    if not figures:
        if evidence.get("histories"):
            item = _plot_history(evidence["histories"][0], output)
            if item:
                figures.append(item)
        completed = next((run for run in evidence.get("metrics", []) if run.get("status") == "completed"), None)
        if completed:
            for builder in (_plot_evaluation, _plot_confusions):
                item = builder(completed, output)
                if item:
                    figures.append(item)
    manifest_files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "research-day-manifest.json":
            manifest_files.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "status": "generated",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "figures": [{"stem": item["stem"], "caption": item["caption"], "origin": item["origin"]} for item in figures],
        "files": manifest_files,
    }
    manifest_path = output / "research-day-manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return figures, manifest_path
