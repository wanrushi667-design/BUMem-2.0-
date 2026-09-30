"""完整顺序实验：python tests/test_match_merge.py。

data/test2.json 保存当前记忆库，data/orign.json 保存原始抽取及合并去向。
每次运行从空库开始；两个输出均不保存 embedding，向量留在内存中。
仅运行前两个 period：python tests/test_match_merge.py --periods 2。
"""

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.model_config import DeepSeekConfig
from memory.extractor import extract_memory
from memory.match import MemoryMatcher
from memory.merge import merge_memory
from models import ModelManager


def run(dialogues, output, original_output, deepseek, embedding):
    rows = [row for row in dialogues if row["role"] == "user"]
    matcher = MemoryMatcher(llm_client=deepseek)
    memories, originals = [], []
    conversation_times = {}

    def save():
        current = [
            {k: v for k, v in memory.to_dict().items() if k != "embedding"}
            for memory in memories
        ]
        output.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        original_output.write_text(json.dumps(originals, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    for number, row in enumerate(rows, start=1):
        print(f"[{number}/{len(rows)}] sequence={row['sequence']} 抽取、匹配、合并……", flush=True)
        conversation = (row["period"], row["conversation_index"])
        timestamp = re.search(r"\[Current Time:\s*([^\]]+)\]", row["content"])
        if timestamp:
            conversation_times[conversation] = timestamp.group(1).strip().replace(" ", "T", 1)
        current_time = conversation_times.get(conversation, row["period"])
        record = {"source": dict(row), "status": "ok"}
        try:
            new = extract_memory(row["content"], deepseek, current_time=current_time)
            if new is None:
                record.update(status="skipped", reason="未抽取到有效记忆")
            else:
                new.memory_id = f"m{row['sequence']:03d}"
                # to_dict 生成独立快照，后续合并不会覆盖原始抽取结构。
                record.update({k: v for k, v in new.to_dict().items() if k != "embedding"})
                new.embedding = embedding.encode(new.summary).tolist()
                decision = matcher.find_merge_target(new, memories)
                record["match"] = asdict(decision)
                if decision.action == "MERGE":
                    index = next(i for i, memory in enumerate(memories)
                                 if memory.memory_id == decision.target_memory_id)
                    memories[index] = merge_memory(memories[index], new, embedding, deepseek)
                else:
                    memories.append(new)
                print(f"  {decision.action} -> {decision.target_memory_id}，当前 {len(memories)} 条记忆", flush=True)
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            originals.append(record)
            save()
            raise RuntimeError(f"sequence={row['sequence']} 处理失败；已保存进度并停止") from exc
        originals.append(record)
        save()
    print(f"完成：{output}\n原始抽取及合并去向：{original_output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="从头运行并覆盖已有输出")
    parser.add_argument("--periods", type=int, help="按文件首次出现顺序，仅处理前 N 个 period；默认全部")
    args = parser.parse_args()
    if args.periods is not None and args.periods < 1:
        parser.error("--periods 必须大于 0")
    output = ROOT / "data" / "test2.json"
    original_output = ROOT / "data" / "orign.json"
    if not args.overwrite and (output.exists() or original_output.exists()):
        parser.error("test2.json 或 orign.json 已存在；重新运行请加 --overwrite")
    dialogues = json.loads((ROOT / "data" / "user_dialogues.json").read_text(encoding="utf-8"))
    if args.periods is not None:
        periods = list(dict.fromkeys(row["period"] for row in dialogues))[:args.periods]
        dialogues = [row for row in dialogues if row["period"] in periods]
        print(f"本次处理 period：{', '.join(periods)}", flush=True)
    manager = ModelManager(deepseek_config=DeepSeekConfig(max_tokens=2048, temperature=0.0))
    print("初始化 DeepSeek 和加载本地 embedding 模型……", flush=True)
    run(dialogues, output, original_output, manager.get_deepseek(), manager.get_embedding())


if __name__ == "__main__":
    main()
