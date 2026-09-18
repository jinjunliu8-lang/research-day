# 工作流与接口

## CLI 使用

```bash
research-day init --project topic-c
research-day doctor --project topic-c
research-day collect --project topic-c --date today
research-day publish --project topic-c --date today --annotation-file annotations.json
research-day close --project topic-c --date today --annotation-file annotations.json
```

如果正在本地开发且命令未进入 PATH，使用：

```bash
uv run --project /path/to/research-day research-day --help
```

## Annotation JSON

```json
{
  "hours": 6.5,
  "goal": "完成学生模型 baseline 并检查数据泄漏",
  "decisions": ["固定现有源录音划分", "暂不使用保留测试集调参"],
  "next_steps": ["补跑 5 个随机种子", "比较 Student Only 与 Standard KD"],
  "notes": "事实之外的解释必须标明推断边界。"
}
```

`hours` 必须来自用户；其他字段可以根据证据整理，但不能加入不存在的实验或结论。

## 安全边界

- 远程命令固定为只读检查、`find` 和 Git 查询。
- SSH 使用密钥和 `BatchMode=yes`；不要索取、保存或回显服务器密码。
- 默认不同步原始数据、音频、权重、checkpoint、embedding、缓存和超过阈值的文件。
- SSH 不可用时，`close` 拒绝生成“远程完整”日记。用户明确接受本地证据后可加 `--local-only`，并在局限中披露。
- `--dry-run` 只在 state 目录生成预览，不写入 Obsidian。

## 输出位置

```text
<vault>/工作日记/<project>/YYYY/YYYY-MM-DD.md
<vault>/附件/工作日记/<project>/YYYY-MM-DD/
```

证据和预览位于项目配置的 `state_dir`；同日重跑前会备份既有日记和自动附件。

## 完成检查

1. 日记包含 11 个固定章节和手写区。
2. 所有嵌入图存在，图号连续，中文图注与数据一致。
3. 科研图包含 PNG、PDF、源 CSV 和 manifest。
4. Obsidian 审计没有新增失效链接。
5. 最终消息说明是否使用实时服务器证据或 `--local-only`。
