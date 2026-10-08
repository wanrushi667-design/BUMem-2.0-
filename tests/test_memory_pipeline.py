"""按分期运行真实记忆入库：python tests/test_memory_pipeline.py --period period00。"""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.model_config import DeepSeekConfig
from memory.match import MemoryMatcher
from memory.pipeline import add_memory
from memory.storage import load_memories, save_memories
from models import ModelManager
from assess.benchmark import run_evaluation, load_benchmark


def run_period(period: str, data_dir=ROOT / "data") -> None:
    data_dir = Path(data_dir)
    dialogues = json.loads((data_dir / "user_dialogues.json").read_text(encoding="utf-8"))
    periods = sorted({item["period"] for item in dialogues})
    if period not in periods:
        raise ValueError(f"未知分期 {period}，可选：{', '.join(periods)}")

    memory_file = data_dir / "memory.json"
    saved = json.loads(memory_file.read_text(encoding="utf-8")) if memory_file.exists() else {}
    completed = saved.get("completed_periods", [])
    score_file = data_dir / "overall_results_strict.json"
    period_index = periods.index(period)
    scores = json.loads(score_file.read_text(encoding="utf-8")) if score_file.exists() else []
    if period in completed and len(scores) > period_index and all(row is not None for row in scores[period_index]):
        print(f"{period} 入库和评测已完成，跳过。")
        return
    if any(name > period for name in completed):
        raise ValueError("已有后续分期记忆，不能用未来记忆评测早期分期。请使用新的 --data-dir，并放入 user_dialogues.json 后从 period00 开始。")
    missing = [name for name in periods[:periods.index(period)] if name not in completed]
    if missing:
        raise ValueError(f"请先按顺序运行：{', '.join(missing)}")
    unassessed = [name for i, name in enumerate(periods[:period_index])
                  if len(scores) <= i or not scores[i] or any(row is None for row in scores[i])]
    if unassessed:
        raise ValueError(f"请先完成前序分期评测：{', '.join(unassessed)}；已有后续记忆时需在新目录重新开始。")

    load_benchmark()  # 入库前确认评测数据和评分模块可用。
    records = sorted(
        (item for item in dialogues if item["period"] == period),
        key=lambda item: item["sequence"],
    )
    manager = ModelManager(deepseek_config=DeepSeekConfig(temperature=0.0))
    print(f"正在加载千问和词向量模型，准备处理 {period} 的 {len(records)} 条对话……", flush=True)
    qwen = manager.get_qwen()
    embedding = manager.get_embedding()
    nodes, groups = load_memories(memory_file, embedding)
    matcher = MemoryMatcher(manager.get_deepseek())

    for index, item in enumerate(records if period not in completed else [], start=1):
        result = add_memory(item["content"], nodes, groups, qwen, embedding, matcher)
        if result is None:
            print(f"[{index}/{len(records)}] 无有效主题，跳过。", flush=True)
            continue
        node, decision = result
        print(
            f"[{index}/{len(records)}] {decision.action} "
            f"{node.memory_id} -> {node.group_id}：{decision.reason}",
            flush=True,
        )

    # 完整分期成功后再保存；中途模型失败不会留下部分入库结果。
    save_memories(memory_file, nodes, groups, completed_periods=list(dict.fromkeys(completed + [period])))
    run_evaluation(memory_file, period, output_file=score_file,
                   manager=manager, nodes=nodes, groups=groups)
    print(f"{period} 完成：共 {len(nodes)} 条记忆、{len(groups)} 个小组，保存至 {memory_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行指定分期，并在已有记忆基础上继续入库")
    parser.add_argument("--period", required=True, help="本次处理的分期，例如 period00 或 period_00")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="输入与累积结果目录")
    args = parser.parse_args()
    period = args.period
    if period.startswith("period") and not period.startswith("period_"):
        period = "period_" + period[len("period"):]
    try:
        run_period(period, args.data_dir)
    except Exception as exc:
        parser.exit(1, f"运行失败：{exc}\n已完成的入库和问题缓存会保留，修复后可重试同一指令。\n")


if __name__ == "__main__":
    main()
