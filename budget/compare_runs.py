"""Summarize the three completed experiments without rerunning evaluation."""

from .run import ROOT, read_json
from assess.benchmark import write_json


METRICS = ("accuracy", "jaccard", "hamming", "accurate_retrieval", "selective_forgetting")
RUNS = (
    ("角色1", "amemgym-first-role", "amemgym-first-role-budget90", 9),
    ("角色2", "amemgym-second-role", "amemgym-second-role-budget90", 9),
    ("角色3", "amemgym-third-role", "amemgym-third-role-budget90-two-pass", 10),
)


def compare(before, after):
    result = {}
    for metric in METRICS:
        left = [row["scores"][metric] for period in before for row in period
                if row["scores"][metric] is not None]
        right = [row["scores"][metric] for period in after for row in period
                 if row["scores"][metric] is not None]
        assert len(left) == len(right)
        a = sum(left) / len(left) if left else None
        b = sum(right) / len(right) if right else None
        result[metric] = {"before": a, "after": b, "count": len(left),
                          "delta": b - a if a is not None else None}
    return result


def main():
    root = ROOT / "eval-runs"
    report = {}
    all_before, all_after = [], []
    active_before, active_after = [], []
    for label, source, target, first in RUNS:
        before = read_json(root / source / "overall_results.json")
        after = read_json(root / target / "overall_results.json")
        assert len(before) == len(after) == 11
        assert all(row is not None for periods in (before, after) for period in periods for row in period)
        config = read_json(root / target / "agent_config.json")
        tokens = read_json(root / target / "tokens.json")
        total_before = config["baseline_tokens"]
        total_after = tokens["period_10"]["total_tokens"]
        report[label] = {
            "baseline_dir": source, "budget_dir": target,
            "affected_periods": list(range(first, 11)),
            "tokens": {"before": total_before, "after": total_after,
                       "saved": total_before - total_after,
                       "saved_ratio": 1 - total_after / total_before},
            "overall": compare(before, after),
            "affected_only": compare(before[first:], after[first:]),
            "per_period": {f"period_{i:02d}": compare([a], [b])
                           for i, (a, b) in enumerate(zip(before, after))},
        }
        all_before.extend(before)
        all_after.extend(after)
        active_before.extend(before[first:])
        active_after.extend(after[first:])
    report["三个角色合并"] = {"overall": compare(all_before, all_after),
                              "affected_only": compare(active_before, active_after)}
    write_json(root / "budget90-comparison.json", report)

    lines = ["# 三个角色压缩前后对比", "",
             "单位为百分比，变化为百分点。所有五项指标均为越高越好；Hamming 为字段匹配率。",
             "整体按所有有效问答得分取平均，不把 selective_forgetting 的 null 当作 0。",
             "每角色 11 个 period、110 次问答；选择性遗忘的有效样本分别为 37、37、38。", "",
             "三个角色均从原全预算快照出发，采用相同的新版第一轮与第二轮提示词。",
             "未触发压缩的 period 复用原结果；触发后重新评测。单次运行的差异不能直接归因于压缩。", ""]
    for scope, title in (("overall", "全部 period"), ("affected_only", "仅实际触发压缩后的 period")):
        lines.extend([f"## {title}", "", "| 角色 | 指标 | 有效问答数 | 压缩前 | 压缩后 | 变化（百分点） |",
                      "| --- | --- | ---: | ---: | ---: | ---: |"])
        for label, data in report.items():
            for metric, values in data[scope].items():
                lines.append(f"| {label} | {metric} | {values['count']} | "
                             f"{values['before'] * 100:.2f}% | {values['after'] * 100:.2f}% | "
                             f"{values['delta'] * 100:+.2f} |")
        lines.append("")
    lines.extend(["## 最终记忆 token 占用", "", "| 角色 | 压缩前 | 压缩后 | 节省 | 节省比例 |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for label, _, _, _ in RUNS:
        t = report[label]["tokens"]
        lines.append(f"| {label} | {t['before']} | {t['after']} | {t['saved']} | {t['saved_ratio']:.2%} |")
    for label, _, _, first in RUNS:
        lines.extend(["", f"## {label}逐 period", "",
                      "表格单元格为压缩前 → 压缩后；— 表示该项无适用问答。", "",
                      "| Period | accuracy | jaccard | hamming | accurate_retrieval | selective_forgetting |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for period, metrics in report[label]["per_period"].items():
            cells = ["—" if v["before"] is None else f"{v['before']:.2%} → {v['after']:.2%}"
                     for v in metrics.values()]
            lines.append(f"| {period} | " + " | ".join(cells) + " |")
    (root / "budget90-comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "budget90-comparison.md")


if __name__ == "__main__":
    main()
