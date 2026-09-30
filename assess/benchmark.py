"""只评测已有记忆，不回放对话或更新记忆。

输出沿用 overall.py 的 period × question 二维列表及字段，新增 reason。
answer 是零起始标准答案索引；response 是一起始模型答案编号。
"""

import importlib.util
import json
import re
from pathlib import Path
from time import perf_counter

from memory.memory_node import MemoryNode
from models import ModelManager
from retrieval import process_query, retrieve_memories
from retrieval.reply import REPLY_SYSTEM_PROMPT


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = Path(r"D:\cd\project\amemgym_memory_bench")
ITEM_ID = "e8bbeec8-7398-4751-8ca4-dd65bbd2d31d"
OUTPUT_FILE = ROOT / "data" / "overall_results.json"

EVALUATION_PROMPT = REPLY_SYSTEM_PROMPT + """

For this evaluation, override the output-only requirements above.
Return valid JSON with exactly these fields:
{"answer": 1, "reason": "简短的证据说明"}
answer must be an integer option ID, starting at 1, from the supplied choices.
Write reason in Chinese: briefly cite relevant memory IDs and supporting facts.
When relevant evidence contains an update, cite the selected attribute and its
`time_at`, and identify the older value it supersedes or narrows. If temporal
applicability is unresolved, say so. Do not justify restoring an older state
using generic Query wording.
Mention uncertainty if evidence is insufficient. Provide a concise evidence
summary, not a step-by-step reasoning transcript.
"""


def evaluate_question(qa, period, previous_period, memories, manager, score_states):
    started = perf_counter()
    embedding = manager.get_embedding()
    query = process_query(qa["query"], embedding, manager.get_qwen())
    hits = retrieve_memories(query, memories, embedding, top_k=5)
    choices = qa["answer_choices"]
    payload = {
        "query": {k: v for k, v in query.items() if k != "embedding"},
        "answer_choices": [
            {"id": index + 1, "state": choice["state"]}
            for index, choice in enumerate(choices)
        ],
        "memories": [
            {k: v for k, v in hit.memory.to_dict().items() if k != "embedding"}
            for hit in hits
        ],
    }
    # 候选选项的 state 参与选择；用户真实 state 仅供评分。
    response = manager.get_deepseek().chat([
        {"role": "system", "content": EVALUATION_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ], json_mode=True)
    answer = json.loads(response)
    number = answer["answer"]
    if type(number) is not int or not 1 <= number <= len(choices):
        raise ValueError(f"模型答案编号必须是 1 到 {len(choices)} 的整数：{number!r}")
    if not isinstance(answer["reason"], str) or not answer["reason"].strip():
        raise ValueError("模型未返回选择依据")
    # 编号 n 对应索引 n - 1，绝不直接以编号访问列表。
    predicted = choices[number - 1]
    golden_state = [period["state"][key] for key in qa["required_info"]]
    golden_index = next(i for i, choice in enumerate(choices) if choice["state"] == golden_state)
    golden = choices[golden_index]
    previous_state = (
        [previous_period["state"][key] for key in qa["required_info"]]
        if previous_period is not None else None
    )
    return {
        "query": qa["query"],
        "answer": golden_index,
        "answer_state": golden["state"],
        "answer_choice": golden["answer"],
        "raw_response": response,
        "response": number,
        "response_state": predicted["state"],
        "response_choice": predicted["answer"],
        "json_error": False,
        "llm_usage_statistics": {
            "source": "BUMem",
            "time_elapsed": perf_counter() - started,
            "used_memory_ids": [hit.memory.memory_id for hit in hits],
        },
        "scores": score_states(golden["state"], predicted["state"], previous_state),
        "reason": answer["reason"],
    }


def run_evaluation(memory_file, period_name, *, output_file=OUTPUT_FILE):
    data = json.loads((BENCHMARK_ROOT / "data/v1.base/data.json").read_text(encoding="utf-8"))
    item = next(item for item in data if item["id"] == ITEM_ID)
    if not re.fullmatch(r"period_\d{2,}", period_name):
        raise ValueError("评测阶段格式应为 period_01")
    period_index = int(period_name.removeprefix("period_"))
    if not 0 <= period_index < len(item["periods"]):
        raise ValueError("评测阶段超出数据集范围")
    period = item["periods"][period_index]
    previous = item["periods"][period_index - 1] if period_index else None

    # 直接加载原评分文件，避免引入原主入口的交互回放及其他 agent 依赖。
    spec = importlib.util.spec_from_file_location(
        "amemgym_metric", BENCHMARK_ROOT / "src/amemgym/eval/metric.py"
    )
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)

    memory_data = json.loads(Path(memory_file).read_text(encoding="utf-8"))
    memories = [MemoryNode(**row) for row in memory_data["memories"]]
    print(f"加载 {len(memories)} 条记忆，评测 {period_name}（截至 {period['period_end']}）", flush=True)
    print("加载模型并生成缺失的记忆向量，仅保留在内存中……", flush=True)
    manager = ModelManager()
    missing = [memory for memory in memories if memory.embedding is None]
    if missing:
        vectors = manager.get_embedding().encode([memory.summary for memory in missing])
        for memory, vector in zip(missing, vectors):
            memory.embedding = vector.tolist()

    output_file = Path(output_file)
    results = (
        json.loads(output_file.read_text(encoding="utf-8")) if output_file.exists()
        else [[None] * len(item["qas"]) for _ in item["periods"]]
    )
    # 本次阶段重新评测；保留其他阶段，未评测位置保持 null。
    results[period_index] = [None] * len(item["qas"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for index, qa in enumerate(item["qas"]):
        print(f"[{index + 1}/{len(item['qas'])}] {qa['query']}", flush=True)
        result = evaluate_question(qa, period, previous, memories, manager, metrics.evaluate_state_metrics)
        results[period_index][index] = result
        output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"答案编号：{result['response']}；accuracy：{result['scores']['accuracy']}", flush=True)
    print(f"评测完成，结果已写入：{output_file}", flush=True)
    return results
