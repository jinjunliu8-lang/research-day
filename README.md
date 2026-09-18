# research-day

Turn a day of experiments into an evidence-backed Obsidian research journal.

`research-day` collects verifiable local and remote experiment artifacts, parses common metrics and training histories, creates publication-ready scientific figures, and publishes an auditable daily note without copying raw datasets or model weights.

## Why it exists

Research logs are most useful when they contain more than a memory of what happened. This tool connects each daily claim to a file, metric, figure, Git change, or failure log, while keeping the narrative readable for both the researcher and an advisor.

## Features

- Collects local artifacts and optional read-only SSH evidence within one configured project root.
- Uses `Asia/Shanghai` day boundaries by default and normalizes timestamps across machines.
- Parses JSON metrics, CSV training histories, confusion matrices, failure logs, Git activity, and running experiments.
- Exports color-blind-friendly figures as 300 dpi PNG, vector PDF, source CSV, and a SHA-256 manifest.
- Publishes a fixed 11-section Markdown journal into Obsidian.
- Preserves a permanent handwritten section when the same day is regenerated.
- Creates recoverable backups and uses atomic writes for notes.
- Supports dry runs and explicitly labels local-only evidence.
- Excludes secrets, raw audio, datasets, checkpoints, embeddings, caches, and oversized files by default.
- Includes an optional Codex skill that turns requests such as "总结今天工作" into the validated CLI workflow.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- An Obsidian vault
- Optional: SSH public-key access to a research server

## Install

From GitHub:

```bash
uv tool install git+https://github.com/jinjunliu8-lang/research-day.git
```

For development:

```bash
git clone https://github.com/jinjunliu8-lang/research-day.git
cd research-day
uv sync --dev
uv run pytest -q
uv tool install --editable . --force
```

## Quick start

Create the built-in `topic-c` profile:

```bash
research-day init --project topic-c
```

Edit these generated files before the first collection:

```text
~/.config/research-day/config.yaml
~/.config/research-day/projects/topic-c.yaml
```

At minimum, set `vault_path`, `state_dir`, and `local_root`. To collect remote evidence, set `remote.enabled: true` and provide the SSH host, port, user, and project root. Credentials are deliberately not part of the configuration format.

Verify the environment:

```bash
research-day doctor --project topic-c
```

Preview a complete daily journal without modifying Obsidian:

```bash
research-day close \
  --project topic-c \
  --date today \
  --hours 6.5 \
  --goal "Complete the teacher-model baseline" \
  --dry-run
```

Publish after reviewing the preview:

```bash
research-day close --project topic-c --date today --hours 6.5
```

Actual work hours are always supplied by the user; the tool never estimates them from timestamps or runtime.

## Commands

| Command | Purpose |
|---|---|
| `init` | Create global and project configuration files. |
| `doctor` | Check paths, dependencies, secrets policy, and SSH access. |
| `collect` | Build a normalized `evidence.json` without publishing. |
| `publish` | Render an existing evidence bundle into Obsidian. |
| `close` | Collect, plot, validate, and publish in one command. |

All date-aware commands accept `--date today` or an ISO date such as `2026-09-17`. Use `--local-only` only when you intentionally accept a journal based on already-synchronized local artifacts.

Structured annotations can be supplied with `--annotation-file`:

```json
{
  "hours": 6.5,
  "goal": "Complete the baseline and audit data leakage",
  "decisions": ["Keep the source-recording split fixed"],
  "next_steps": ["Run five random seeds"],
  "notes": "Interpretations beyond direct evidence are marked as inferences."
}
```

## Output layout

```text
<vault>/
├── 工作日记/<project>/YYYY/YYYY-MM-DD.md
└── 附件/工作日记/<project>/YYYY-MM-DD/
    ├── *.png
    ├── *.pdf
    ├── figure_data/*.csv
    └── research-day-manifest.json
```

Evidence bundles, dry-run previews, and backups live under the configured `state_dir`.

## Codex skill

The reusable skill lives in [`codex-skill/write-daily-research-log`](codex-skill/write-daily-research-log). Install it into your personal Codex skills directory:

```bash
cp -R codex-skill/write-daily-research-log ~/.codex/skills/
```

After installation, requests such as "生成今天的科研日记" trigger the evidence-first workflow. The skill requires user-provided hours, separates facts from interpretation, and validates the final note and figure manifest.

## Safety model

- Remote access uses system SSH with public-key authentication and read-only commands.
- Passwords, tokens, private keys, shell history, and files outside the configured project root are never collected.
- Raw datasets, audio, model weights, checkpoints, and embeddings are excluded by default.
- Publishing stops when configured remote evidence is unavailable unless `--local-only` is explicitly selected.
- A dry run writes only to `state_dir`, never to the Obsidian vault.

Review generated profiles before use and keep personal configuration outside this repository.

## Test

```bash
uv run pytest -q
```

The test suite covers configuration safety, metric parsing, time-zone boundaries, exclusion rules, scientific figure exports, dry-run isolation, atomic republishing, handwritten-section preservation, redaction, and annotation input.

## License

[MIT](LICENSE)
