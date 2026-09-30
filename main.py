"""使用 DeepSeek 抽取并保存记忆：python main.py。"""

import argparse
import json
from pathlib import Path

from config.model_config import DeepSeekConfig
from memory.extractor import extract_memory
from models import ModelManager


MEMORY_FILE = Path(__file__).resolve().parent / "data" / "memory.json"


def main():
    parser = argparse.ArgumentParser(description="抽取主题记忆、生成向量并保存")
    parser.add_argument("--model", choices=["deepseek"], default="deepseek")
    args = parser.parse_args()
    manager = ModelManager(
        deepseek_config=DeepSeekConfig(temperature=0.0, max_tokens=2048),
    )
    print(f"正在加载 {args.model} 和词嵌入模型……")
    try:
        client = manager.get_deepseek()
        embedding_client = manager.get_embedding()
    except Exception as exc:
        print(f"模型加载失败：{exc}")
        return

    while True:
        try:
            dialogue = input("请输入一段记忆（输入 /exit 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if dialogue == "/exit":
            break
        if not dialogue:
            continue
        try:
            memory = extract_memory(dialogue, client)
            if memory is None:
                print("没有可保存的记忆。")
                continue
            memory.embedding = embedding_client.encode(memory.summary).tolist()
            memories = json.loads(MEMORY_FILE.read_text(encoding="utf-8")) if MEMORY_FILE.exists() else []
            memories.append(memory.to_dict())
            MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            MEMORY_FILE.write_text(json.dumps(memories, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"处理失败：{exc}")
            continue
        print(f"记忆已保存至 {MEMORY_FILE}（共 {len(memories)} 条）。")


if __name__ == "__main__":
    main()
