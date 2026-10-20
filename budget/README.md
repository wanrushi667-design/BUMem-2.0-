# 90% 两轮压缩预算实验

在项目根目录运行，默认使用第 1 个角色已有的全预算结果：

```powershell
.venv/Scripts/python.exe -m budget.run
```

只统计预算、不调用过滤或评测：

```powershell
.venv/Scripts/python.exe -m budget.run --inspect-only
```

默认读取 `eval-runs/amemgym-first-role`，实验写入独立目录
`eval-runs/amemgym-first-role-budget90-two-pass`。也可以通过 `--baseline-dir` 和
`--output-dir` 指定目录。`--stop-after-period 9` 表示完成 period_09 后停止。
使用同一命令继续运行时，已经完成的 period 会跳过；已保存的过滤批次不会重做。

第二角色使用独立的新目录，保留旧单轮实验结果：

```powershell
.venv/Scripts/python.exe -m budget.run --baseline-dir eval-runs/amemgym-second-role --output-dir eval-runs/amemgym-second-role-budget90-two-pass
```

## 统计与执行规则

- 基准是该角色最后一个全预算快照的累计 token 数，不累加各 period 的快照。
- 每条记忆分别统计 `raw_text` 和每个 `topics` 字符串，按 memory_id 各计一次，
  再汇总；相同 topic 若存于不同记忆中，分别计数。不统计组结构或 JSON 开销。
- 统一复用本地 Qwen GGUF tokenizer，不添加 BOS 或聊天模板。
- 固定上限为基准乘以 0.9 后向下取整，等于上限时已达标。
- 首次超限前复制已有快照及评测结果。首次超限 period 直接使用已有入库结果。
- 第一轮每批仅选最早仍有未尝试记忆的 period，按 raw_text token 数从多到少选最多
  10 条；不足 10 条不跨 period 补齐。过滤后重新统计整库占用。
- 第一轮候选全部尝试后仍超预算，启动第二轮：跨全部 period，按当前 raw_text
  token 数从多到少选最多 10 条，以当前保留的文本为输入进行紧凑笔记式压缩。
- 每批后重新计数和排序，达标即停止，不会在批次中途停止。
- 两轮独立标记：`filtered_ids` 记录第一轮尝试，`second_filtered_ids` 记录第二轮
  尝试。同一记忆每轮最多尝试一次，标记跨 period 保存。第一轮未缩短的记忆仍可
  进入第二轮；第二轮不会重做已经尝试过的条目。
- 后续 period 新增记忆先参加第一轮，需要时再继续全局第二轮。
- 仅接受非空且 token 更少的结果，否则保留原文，仍标记为已尝试。
- 修改 raw_text 后更新 embedding，topics、ID 和组关系不变。
  原文开头的 `[Current Time: ...]` 时间头由代码原样保留。
- 达标后保存快照并评测，再从压缩后的状态引入下一个 period 的原始对话。
- 两轮候选均耗尽仍超限时，记录当前 token 统计并停止报错，不继续重试或评测。

两份提示词均在 **`budget/text_filter.py`**：第一轮为 `TEXT_FILTER_SYSTEM_PROMPT`，
第二轮为 `SECOND_FILTER_SYSTEM_PROMPT`。两轮均保留事实与需求性质，第二轮更明确
要求短语和分号组织的紧凑笔记。修改提示词后使用新的输出目录，避免混用不同实验设置。
常规实验入口不直接续跑旧单轮目录，也不迁移其标记。

角色1和角色2已按相同的新提示词从全预算快照重新运行，覆盖原预算输出目录。
旧提示词的结果及第二角色 P10 的历史续跑记录归档在
`eval-runs/budget90-previous-prompts-backup/`。当前结果不包含旧第一轮续跑状态。
`validator.py` 只验证输出和 token 节省，不能保证语义无损；下游问答评测与原文抽查
用于评估信息保留效果。70% 和 50% 的操作尚未实现。

## 输出

- `baseline_tokens.json`：原全预算各 period 的逐记忆统计及预算判断。
- `tokens.json`：预算实验各 period 的最终统计，同一 period 覆盖更新。
- `agent_states/period_XX/memory.json`：记忆快照，以及 `budget` 字段中的两轮尝试
  ID、批次节省记录和评测就绪状态。批次的 `pass_index` 为 1 或 2；第一轮的
  `period_index` 为候选来源，第二轮为 null，各条 `changes` 单独记录来源 period。
- `overall_results.json`、`overall_metrics*.json`：原有格式的问答评测结果。
- `comparison.json`：各 period 的原始/实验 token 数和五项评测指标对比。

测试不加载模型、不调用 API：

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_budget*.py" -v
```
