"""输入 2，使用已有记忆运行 AMemGym 当前 period 的全部评测题。

默认：python tests/test_interactive_retrieval.py
切换：python tests/test_interactive_retrieval.py --memory data/period_02.json --period period_02
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assess import run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", type=Path, default=Path("data/period_01.json"))
    parser.add_argument("--period", default="period_01", help="数据集阶段，如 period_01（第二阶段）")
    args = parser.parse_args()
    memory_file = ROOT / args.memory
    print(f"记忆文件：{memory_file}\n评测阶段：{args.period}")
    while True:
        try:
            action = input("\n输入 2 开始评测，输入 /exit 退出：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        if action == "/exit":
            return
        if action == "2":
            run_evaluation(memory_file, args.period)
        else:
            print("请输入 2 或 /exit。")


if __name__ == "__main__":
    main()
