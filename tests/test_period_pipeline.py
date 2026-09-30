"""按轮抽取、合并并评测：python tests/test_period_pipeline.py --period period_00。

origin.json 累计保留合并前的抽取结构；period_XX.json 保存每轮完整记忆。
已有本轮快照时只重跑评测。中断后重跑会复用已保存的原始抽取，重新合并本轮。
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assess.benchmark import run_evaluation
from config.model_config import DeepSeekConfig
from memory.extractor import extract_memory
from memory.match import MemoryMatcher
from memory.memory_node import MemoryNode
from memory.merge import merge_memory
from models import ModelManager


def save_json(path, value):
    # 避免中断写入时损坏已有的原始记录或快照。
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def memory_dict(memory):
    return {key: value for key, value in memory.to_dict().items() if key != "embedding"}


def run_period(period, data_dir=ROOT / "data"):
    data_dir = Path(data_dir)
    dialogues = json.loads((data_dir / "user_dialogues.json").read_text(encoding="utf-8"))
    periods = list(dict.fromkeys(row["period"] for row in dialogues))
    if period not in periods:
        raise ValueError(f"不存在 {period}；可选：{', '.join(periods)}")

    output = data_dir / f"{period}.json"
    results_file = data_dir / "overall_results.json"
    if output.exists():
        print(f"{output.name} 已存在，保留快照，仅重新评测。", flush=True)
        return run_evaluation(output, period, output_file=results_file)

    period_index = periods.index(period)
    memories = []
    if period_index:
        previous_file = data_dir / f"{periods[period_index - 1]}.json"
        if not previous_file.exists():
            raise ValueError(f"请先运行 {periods[period_index - 1]}，缺少 {previous_file.name}")
        previous = json.loads(previous_file.read_text(encoding="utf-8"))
        memories = [MemoryNode(**row) for row in previous["memories"]]

    original_file = data_dir / "origin.json"
    originals = json.loads(original_file.read_text(encoding="utf-8")) if original_file.exists() else []
    cached = {
        record["source"]["sequence"]: record
        for record in originals if record["source"]["period"] == period
    }
    rows = [row for row in dialogues if row["period"] == period and row["role"] == "user"]
    manager = ModelManager(deepseek_config=DeepSeekConfig(max_tokens=2048, temperature=0.0))
    deepseek, embedding = manager.get_deepseek(), manager.get_embedding()
    matcher = MemoryMatcher(llm_client=deepseek)
    if memories:
        vectors = embedding.encode([memory.summary for memory in memories])
        for memory, vector in zip(memories, vectors):
            memory.embedding = vector.tolist()

    conversation_times = {}
    for number, row in enumerate(rows, start=1):
        print(f"[{period} {number}/{len(rows)}] sequence={row['sequence']}", flush=True)
        conversation = row["conversation_index"]
        timestamp = re.search(r"\[Current Time:\s*([^\]]+)\]", row["content"])
        if timestamp:
            conversation_times[conversation] = timestamp.group(1).strip().replace(" ", "T", 1)
        record = cached.get(row["sequence"])
        if record is None:
            new = extract_memory(
                row["content"], deepseek,
                current_time=conversation_times.get(conversation, period),
            )
            if new is not None:
                new.memory_id = f"m{row['sequence']:03d}"
            record = {
                "source": dict(row),
                "status": "ok" if new is not None else "skipped",
                "memory": memory_dict(new) if new is not None else None,
            }
            originals.append(record)
            # 在匹配和合并之前落盘，保留独立的原始结构。
            save_json(original_file, originals)
        if record["memory"] is None:
            continue
        new = MemoryNode(**record["memory"])
        new.embedding = embedding.encode(new.summary).tolist()
        decision = matcher.find_merge_target(new, memories)
        if decision.action == "MERGE":
            index = next(i for i, memory in enumerate(memories)
                         if memory.memory_id == decision.target_memory_id)
            memories[index] = merge_memory(memories[index], new, embedding, deepseek)
        else:
            memories.append(new)
        print(f"  {decision.action} -> {decision.target_memory_id}；共 {len(memories)} 条记忆", flush=True)

    save_json(output, {"period": period, "memories": [memory_dict(memory) for memory in memories]})
    print(f"已保存 {output}，开始评测。", flush=True)
    return run_evaluation(output, period, output_file=results_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", required=True, help="本次运行的阶段，例如 period_00")
    args = parser.parse_args()
    run_period(args.period)


if __name__ == "__main__":
    main()
