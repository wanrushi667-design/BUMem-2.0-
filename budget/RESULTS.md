# 三个角色使用统一新版提示词的 90% 实验

三个角色均从原全预算快照出发，使用 `text_filter.py` 当前的第一轮和第二轮提示词。
预算是各角色最终全预算 token 数的 90% 向下取整。未触发压缩的 period 复用原结果。

| 角色 | 全预算 tokens | 上限 | 最终 P10 tokens | P10 原准确率 | P10 本次准确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 10,240 | 9,216 | 9,147 | 0.3 | 0.4 |
| 2 | 8,789 | 7,910 | 7,747 | 0.5 | 0.5 |
| 3 | 9,756 | 8,780 | 8,573 | 0.5 | 0.5 |

本次三个角色均在第一轮达标，第二轮未实际触发。第一轮累计尝试条数分别为
24、37、43。角色1和角色2首次在 P9 超限，角色3首次在 P10 超限。

最终输出目录：

- `eval-runs/amemgym-first-role-budget90/`
- `eval-runs/amemgym-second-role-budget90/`
- `eval-runs/amemgym-third-role-budget90-two-pass/`

三个角色均完成 11 个 period、110 次问答，最终所有 period 记忆占用均不超限。
已验证配置中的两份提示词相同、没有沿用旧压缩状态，压缩前各 period 的复用快照
及问答结果与原全预算数据一致。

完整五项指标、实际触发阶段指标、逐 period 对比及 token 节省比例见
`eval-runs/budget90-comparison.md` 和对应 JSON。差异为单次运行结果，不能直接
归因于压缩。

角色1、2旧提示词结果及角色2历史 P10 第二轮续跑结果保存在
`eval-runs/budget90-previous-prompts-backup/`。当前同名输出目录已被新版结果替换。

每个最终目录的 `run_summary.json` 为最新汇总，`logs/budget.log` 为真实运行日志。
