"""分阶段比较：origin 用 DeepSeek 抽取；deepseek/qwen 读取同一份 origin 做 match、merge。"""

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.model_config import DeepSeekConfig, QwenConfig
from memory.extractor import extract_memory
from memory.match import MemoryMatcher
from memory.memory_node import MemoryNode
from memory.merge import merge_memory
from models import ModelManager


def visible(memory):
    return {k: v for k, v in memory.to_dict().items() if k != "embedding"}


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_origin(dialogues, output, client, period=None):
    periods = [period] if period else list(dict.fromkeys(row["period"] for row in dialogues))[:2]
    rows = [row for row in dialogues if row["period"] in periods and row["role"] == "user"]
    result = {"model": "deepseek", "periods": periods, "status": "running", "records": []}
    if period and output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        done = {record["source"]["sequence"] for record in result["records"]}
        if period not in result["periods"]:
            result["periods"].append(period)
        result["status"] = "running"
        result.pop("error", None)
    else:
        done = set()
    times = {}
    save(output, result)
    for number, row in enumerate(rows, 1):
        print(f"[{number}/{len(rows)}] 抽取 sequence={row['sequence']}", flush=True)
        conversation = (row["period"], row["conversation_index"])
        timestamp = re.search(r"\[Current Time:\s*([^\]]+)\]", row["content"])
        if timestamp:
            times[conversation] = timestamp.group(1).strip().replace(" ", "T", 1)
        if row["sequence"] in done:
            continue
        try:
            memory = extract_memory(row["content"], client, times.get(conversation, row["period"]))
            if memory is not None:
                memory.memory_id = f"m{row['sequence']:03d}"
            result["records"].append({
                "source": dict(row),
                "status": "ok" if memory is not None else "skipped",
                "memory": visible(memory) if memory is not None else None,
            })
        except Exception as exc:
            result.update(status="error", error=f"sequence={row['sequence']}: {type(exc).__name__}: {exc}")
            save(output, result)
            raise
        save(output, result)
    result["status"] = "complete"
    save(output, result)


def compare(origin, output, model, client, embedding, periods=None, base=None):
    if origin["status"] != "complete":
        raise ValueError("origin.json 尚未完整抽取成功，请先重新运行 origin 阶段")
    records = [record for record in origin["records"] if record["status"] == "ok"
               and (periods is None or record["source"]["period"] in periods)]
    memories = [MemoryNode(**memory) for memory in base["memories"]] if base else []
    for memory in memories:
        memory.embedding = embedding.encode(memory.summary).tolist()
    matcher = MemoryMatcher(llm_client=client)
    result = {"model": model, "status": "running",
              "memories": [visible(memory) for memory in memories], "decisions": []}
    save(output, result)
    for number, record in enumerate(records, 1):
        new = MemoryNode(**record["memory"])
        print(f"[{number}/{len(records)}] {model}: {new.memory_id}", flush=True)
        try:
            new.embedding = embedding.encode(new.summary).tolist()
            decision = matcher.find_merge_target(new, memories)
            if decision.action == "MERGE":
                index = next(i for i, memory in enumerate(memories)
                             if memory.memory_id == decision.target_memory_id)
                memories[index] = merge_memory(memories[index], new, embedding, llm_client=client)
            else:
                memories.append(new)
            result["decisions"].append({"memory_id": new.memory_id, **asdict(decision)})
            result["memories"] = [visible(memory) for memory in memories]
            print(f"  {decision.action} -> {decision.target_memory_id}", flush=True)
        except Exception as exc:
            result.update(status="error", error=f"{new.memory_id}: {type(exc).__name__}: {exc}")
            save(output, result)
            raise
        save(output, result)
    result["status"] = "complete"
    save(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["origin", "deepseek", "qwen"])
    parser.add_argument("--period", help="指定阶段；deepseek 默认处理 period_00、period_01，另支持 period_02、period_03；origin 追加抽取")
    parser.add_argument("--overwrite", action="store_true", help="覆盖当前阶段的输出，从头运行")
    args = parser.parse_args()
    periods = [args.period] if args.period else None
    base_path = None
    output = ROOT / "data" / f"{args.stage}.json"
    if args.stage == "deepseek":
        if args.period is None:
            periods = ["period_00", "period_01"]
            output = ROOT / "data" / "period_01.json"
        elif args.period in ("period_02", "period_03"):
            number = int(args.period[-2:])
            output = ROOT / "data" / f"period_{number:02d}.json"
            base_path = ROOT / "data" / f"period_{number - 1:02d}.json"
        else:
            parser.error("deepseek 不指定 --period 时处理 period_00、period_01；后续阶段请选择 period_02 或 period_03")
    if output.exists() and not args.overwrite and not (args.stage == "origin" and args.period):
        parser.error(f"{output.name} 已存在；重新运行请加 --overwrite")
    input_path = ROOT / "data" / ("user_dialogues.json" if args.stage == "origin" else "origin.json")
    if not input_path.exists():
        parser.error(f"缺少 {input_path.name}；请先运行 origin 阶段")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if args.stage != "origin" and data["status"] != "complete":
        parser.error("origin.json 尚未完整抽取成功，请先重新运行 origin 阶段")
    available_periods = {row["period"] for row in data} if args.stage == "origin" else set(data["periods"])
    if periods and set(periods) - available_periods:
        parser.error(f"输入中没有 {', '.join(sorted(set(periods) - available_periods))}")
    base = None
    if base_path:
        if not base_path.exists():
            parser.error(f"缺少 {base_path.name}；请先运行前一阶段")
        base = json.loads(base_path.read_text(encoding="utf-8"))
        if base["status"] != "complete":
            parser.error(f"{base_path.name} 尚未完整运行成功")
    manager = ModelManager(
        qwen_config=QwenConfig(n_ctx=16384, n_batch=128, max_tokens=2048, temperature=0.0),
        deepseek_config=DeepSeekConfig(max_tokens=2048, temperature=0.0),
    )
    if args.stage == "origin":
        extract_origin(data, output, manager.get_deepseek(), args.period)
    else:
        client = manager.get_deepseek() if args.stage == "deepseek" else manager.get_qwen()
        compare(data, output, args.stage, client, manager.get_embedding(), periods, base)
    print(f"已保存：{output}", flush=True)


if __name__ == "__main__":
    main()
