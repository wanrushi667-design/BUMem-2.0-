"""复用结构化问题，检索记忆组并调用 reply 回答和评分。"""

import importlib.util
import json
from pathlib import Path
from time import perf_counter

from memory.storage import load_memories
from models import ModelManager
from retrieval import process_query, retrieve_memories
from retrieval.reply import select_answer

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = Path(r"D:\cd\project\amemgym_memory_bench")
ITEM_ID = "e8bbeec8-7398-4751-8ca4-dd65bbd2d31d"
OUTPUT_FILE = ROOT / "data" / "overall_results_strict.json"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def load_benchmark():
    data = json.loads((BENCHMARK_ROOT / "data/v1.base/data.json").read_text(encoding="utf-8"))
    item = next(item for item in data if item["id"] == ITEM_ID)
    spec = importlib.util.spec_from_file_location(
        "amemgym_metric", BENCHMARK_ROOT / "src/amemgym/eval/metric.py"
    )
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    return item, metrics.evaluate_state_metrics


def run_evaluation(memory_file, period_name, *, output_file=OUTPUT_FILE,
                   manager=None, nodes=None, groups=None):
    item, score_states = load_benchmark()
    period_index = int(period_name.removeprefix("period_"))
    period = item["periods"][period_index]
    previous = item["periods"][period_index - 1] if period_index else None
    output_file = Path(output_file)
    query_file = output_file.parent / "query.json"
    result_file = output_file.parent / "result.json"
    cache = json.loads(query_file.read_text(encoding="utf-8")) if query_file.exists() else {}
    replies = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {}
    results = (json.loads(output_file.read_text(encoding="utf-8")) if output_file.exists()
               else [[None] * len(item["qas"]) for _ in item["periods"]])
    manager = manager or ModelManager()
    embedding = manager.get_embedding()
    if nodes is None:
        nodes, groups = load_memories(memory_file, embedding)
    period_replies, period_scores = [], []
    for index, qa in enumerate(item["qas"]):
        started = perf_counter()
        text = qa["query"].strip()
        if text in cache:
            query = {**cache[text], "embedding": embedding.encode(text).tolist()}
        else:
            query = process_query(text, embedding, manager.get_qwen())
            cache[text] = {key: value for key, value in query.items() if key != "embedding"}
            write_json(query_file, cache)
        memories = retrieve_memories(query, nodes, groups, manager.get_deepseek())
        choices = [{**choice, "id": str(i + 1)} for i, choice in enumerate(qa["answer_choices"])]
        reply = select_answer(query, choices, memories, manager.get_deepseek())
        period_replies.append({"query": text, "answer_choices": choices,
                               "memory_ids": [memory.memory_id for memory in memories],
                               "reply": reply})
        number = int(reply["answer_id"]) if reply["answer_id"] is not None else None
        predicted = choices[number - 1] if number is not None else None
        golden_state = [period["state"][key] for key in qa["required_info"]]
        golden_index = next(i for i, choice in enumerate(choices) if choice["state"] == golden_state)
        previous_state = ([previous["state"][key] for key in qa["required_info"]]
                          if previous is not None else None)
        predicted_state = predicted["state"] if predicted else [""] * len(golden_state)
        scores = score_states(golden_state, predicted_state, previous_state)
        if predicted is None:
            scores = {key: (None if value is None else 0.0) for key, value in scores.items()}
        period_scores.append({
            "query": text, "answer": golden_index, "answer_state": golden_state,
            "answer_choice": choices[golden_index]["answer"],
            "raw_response": json.dumps(reply, ensure_ascii=False), "response": number,
            "response_state": predicted["state"] if predicted else None,
            "response_choice": predicted["answer"] if predicted else None,
            "json_error": False, "answer_status": reply["answer_status"],
            "reason": reply["current_state"], "scores": scores,
            "llm_usage_statistics": {"source": "BUMem", "time_elapsed": perf_counter() - started,
                                     "used_memory_ids": [memory.memory_id for memory in memories]},
        })
        print(f"[{index + 1}/{len(item['qas'])}] 答案：{number}；accuracy：{scores['accuracy']}", flush=True)
    replies[period_name] = period_replies
    results[period_index] = period_scores
    write_json(result_file, replies)
    write_json(output_file, results)
    return results
