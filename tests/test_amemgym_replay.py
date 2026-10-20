"""Replay an AMemGym character, one period per invocation."""

import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assess.benchmark import BENCHMARK_ROOT, load_benchmark, run_evaluation, write_json
from config.model_config import DeepSeekConfig
from memory.match import MemoryMatcher
from memory.pipeline import add_memory
from memory.storage import load_memories, save_memories
from models import ModelManager

ITEM_ID = "875a8a65-5b3e-4cfb-b5f1-635d840cdeaf"
SOURCE = BENCHMARK_ROOT / "eval-output/test-native/deepseek-v4-flash-mini" / ITEM_ID
OUTPUT = ROOT / "eval-runs/amemgym-second-role"


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()


def run_period(period_index, output_dir=OUTPUT, source_dir=SOURCE,
               benchmark_root=BENCHMARK_ROOT, manager=None, item_id=ITEM_ID):
    output_dir, source_dir = Path(output_dir), Path(source_dir)
    config_file = output_dir / "agent_config.json"
    if config_file.exists() and json.loads(config_file.read_text(encoding="utf-8"))["item_id"] != item_id:
        raise ValueError("This output directory belongs to another character. Use a different --output-dir.")
    item, _ = load_benchmark(benchmark_root, item_id)
    if not 0 <= period_index < len(item["periods"]):
        raise ValueError(f"period must be between 0 and {len(item['periods']) - 1}")
    period = f"period_{period_index:02d}"
    state_root = output_dir / "agent_states"
    states = sorted(state_root.glob("period_*/memory.json"))
    memory_file = states[-1] if states else state_root / period / "memory.json"
    saved = json.loads(memory_file.read_text(encoding="utf-8")) if memory_file.exists() else {}
    completed = saved.get("completed_periods", [])
    prior = [f"period_{i:02d}" for i in range(period_index)]
    score_file = output_dir / "overall_results.json"
    scores = json.loads(score_file.read_text(encoding="utf-8")) if score_file.exists() else []
    if any(len(scores) <= i or any(row is None for row in scores[i])
           for i in range(period_index)):
        raise ValueError("Complete evaluation of previous periods first.")
    if period in completed and len(scores) > period_index and all(
        row is not None for row in scores[period_index]
    ):
        print(f"{period} already completed: {output_dir}", flush=True)
        return
    if completed not in (prior, prior + [period]):
        raise ValueError("Run periods in order from 0; use a new output directory to restart.")

    interactions = json.loads((source_dir / "interactions" / f"{period}.json").read_text(encoding="utf-8"))
    records = [msg for session in interactions for msg in session if msg["role"] == "user"]
    snapshot = state_root / period
    snapshot.mkdir(parents=True, exist_ok=True)
    write_json(config_file, {
        "item_id": item_id, "source_dir": str(source_dir.resolve()),
        "benchmark_root": str(Path(benchmark_root).resolve()),
        "ingest_roles": ["user"], "deepseek_temperature": 0.0,
        "deepseek_max_tokens": 2048,
    })
    write_json(output_dir / "interactions" / f"{period}.json", interactions)
    manager = manager or ModelManager(deepseek_config=DeepSeekConfig(temperature=0.0, max_tokens=2048))
    embedding = manager.get_embedding()
    nodes, groups = load_memories(memory_file, embedding)
    matcher = MemoryMatcher(manager.get_deepseek(), establish_topics=period_index == 0)
    if period not in completed:
        for index, msg in enumerate(records):
            add_memory(msg["content"], nodes, groups, manager.get_qwen(),
                       embedding, matcher, period_index=period_index)
            print(f"{period} replay [{index + 1}/{len(records)}]", flush=True)
        memory_file = snapshot / "memory.json"
        save_memories(memory_file, nodes, groups, completed_periods=prior + [period])
    run_evaluation(memory_file, period, output_file=score_file, manager=manager,
                   nodes=nodes, groups=groups, benchmark_root=benchmark_root, item_id=item_id,
                   benchmark_output=True)
    print(f"{period} complete: {len(nodes)} memories, {len(groups)} groups; {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", type=int, required=True, help="Zero-based period, e.g. 0 or 1")
    parser.add_argument("--role", type=int, choices=(1, 2, 3), default=2,
                        help="Character position in the benchmark dataset (default: 2)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT)
    args = parser.parse_args()
    data = json.loads((args.benchmark_root / "data/v1.base/data.json").read_text(encoding="utf-8"))
    item_id = data[args.role - 1]["id"]
    args.source_dir = args.source_dir or (args.benchmark_root / "eval-output/test-native/deepseek-v4-flash-mini" / item_id)
    role_name = {1: "first", 2: "second", 3: "third"}[args.role]
    args.output_dir = args.output_dir or ROOT / f"eval-runs/amemgym-{role_name}-role"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)
    with (args.output_dir / "logs/evaluate.log").open("a", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                run_period(args.period, args.output_dir, args.source_dir, args.benchmark_root,
                           item_id=item_id)
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
