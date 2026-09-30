"""前两个 period 的顺序匹配实验：python tests/test_match_periods.py。

每条记忆均追加，MERGE 只记录判断，不执行内容合并。向量仅保留在内存中。
"""

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.model_config import DeepSeekConfig
from memory.extractor import extract_memory
from memory.match import MemoryMatcher
from models import ModelManager


def candidate_details(new_memory, history, matcher):
    """Report summary ranking and whether each memory was sent to the LLM."""
    if not history:
        return []
    query = np.asarray(new_memory.embedding, dtype=float)
    vectors = np.asarray([m.embedding for m in history], dtype=float)
    scores = vectors @ query / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
    checked = {c["memory_id"]: c for c in matcher.last_candidates}
    details = []
    for rank, index in enumerate(np.argsort(-scores, kind="stable"), start=1):
        old = history[index]
        evidence = checked.get(old.memory_id)
        item = {
            "memory_id": old.memory_id,
            "summary": old.summary,
            "topics": old.topics,
            "summary_rank": rank,
            "summary_score": float(scores[index]),
            "in_summary_top_k": rank <= matcher.top_k,
            "sent_to_llm": evidence is not None,
        }
        details.append(item)
    return details


def run(dialogues, output, deepseek, embedding):
    periods = list(dict.fromkeys(row["period"] for row in dialogues))[:2]
    rows = [row for row in dialogues if row["period"] in periods and row["role"] == "user"]
    matcher = MemoryMatcher(llm_client=deepseek)
    history, records = [], []
    conversation_times = {}
    print(f"测试 {periods}，共 {len(rows)} 条 user 消息。", flush=True)
    for number, row in enumerate(rows, start=1):
        print(f"[{number}/{len(rows)}] 抽取 sequence={row['sequence']}……", flush=True)
        conversation = (row["period"], row["conversation_index"])
        timestamp = re.search(r"\[Current Time:\s*([^\]]+)\]", row["content"])
        if timestamp:
            conversation_times[conversation] = timestamp.group(1).strip().replace(" ", "T", 1)
        current_time = conversation_times.get(conversation, row["period"])
        record = {"source": dict(row), "status": "ok"}
        try:
            new = extract_memory(row["content"], deepseek, current_time=current_time)
            if new is None:
                record.update(status="skipped", reason="未抽取到有效记忆，未进行匹配")
            else:
                new.memory_id = f"m{row['sequence']:03d}"
                new.embedding = embedding.encode(new.summary).tolist()
                record.update({k: v for k, v in new.to_dict().items() if k != "embedding"})
                print(f"  匹配已有 {len(history)} 条记忆……", flush=True)
                decision = matcher.find_merge_target(new, history)
                candidates = candidate_details(new, history, matcher)
                record["match"] = {
                    **asdict(decision),
                    "decision_by": "deepseek" if any(c["sent_to_llm"] for c in candidates) else "rule",
                    "history_count": len(history),
                    "top_k": matcher.top_k,
                    "summary_filter": "按余弦相似度取 Top K；summary 没有阈值",
                    "history_policy": "所有成功处理的新记忆均追加；MERGE 仅记录，不改写旧记忆",
                    "candidates": candidates,
                }
                history.append(new)
                print(f"  {decision.action} -> {decision.target_memory_id}: {decision.reason}", flush=True)
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            records.append(record)
            output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(f"sequence={row['sequence']} 处理失败，已保存记录到 {output}；停止以免影响后续历史") from exc
        records.append(record)
        output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成，结果已保存：{output}（不含 embedding）", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="重新运行并覆盖已有 data/test1.json")
    args = parser.parse_args()
    output = ROOT / "data" / "test1.json"
    if output.exists() and not args.overwrite:
        parser.error("data/test1.json 已存在；如需重新测试，请加 --overwrite")
    dialogues = json.loads((ROOT / "data" / "user_dialogues.json").read_text(encoding="utf-8"))
    manager = ModelManager(deepseek_config=DeepSeekConfig(max_tokens=2048, temperature=0.0))
    print("初始化 DeepSeek 和加载本地 embedding 模型……", flush=True)
    deepseek = manager.get_deepseek()
    embedding = manager.get_embedding()
    run(dialogues, output, deepseek, embedding)


if __name__ == "__main__":
    main()
