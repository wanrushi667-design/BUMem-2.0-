"""Run the first character's 90% experiment: python -m budget.run.

Existing full-budget states and scores are reused until the first over-budget
period. From there, evaluate filtered states and ingest subsequent periods anew.
"""

import argparse
import json
from pathlib import Path

from assess.benchmark import run_evaluation, save_metrics_summary, write_json
from config.model_config import DeepSeekConfig, QwenConfig
from memory.match import MemoryMatcher
from memory.pipeline import add_memory
from memory.storage import load_memories, save_memories
from models import ModelManager

from .manager import BudgetManager
from .text_filter import TEXT_FILTER_SYSTEM_PROMPT, SECOND_FILTER_SYSTEM_PROMPT
from .token_counter import TokenCounter


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "eval-runs/amemgym-first-role"
OUTPUT = ROOT / "eval-runs/amemgym-first-role-budget90-two-pass"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_checkpoint(path, nodes, groups, completed, state):
    save_memories(path, nodes, groups, completed_periods=completed)
    data = read_json(path)
    data["budget"] = state
    write_json(path, data)


def save_scores(output, scores):
    write_json(output / "overall_results.json", scores)
    write_json(output / "overall_metrics.json", {
        metric: [[row["scores"][metric] if row is not None else None for row in period]
                 for period in scores]
        for metric in ("accuracy", "jaccard", "hamming", "accurate_retrieval", "selective_forgetting")
    })
    save_metrics_summary(scores, output / "overall_metrics_summary.json", include_protocol=False)


def run_experiment(baseline_dir=BASELINE, output_dir=OUTPUT, *, manager=None,
                   inspect_only=False, stop_after_period=None):
    baseline, output = Path(baseline_dir).resolve(), Path(output_dir).resolve()
    if baseline == output or baseline in output.parents:
        raise ValueError("Use an output directory outside the full-budget run")
    source_config = read_json(baseline / "agent_config.json")
    manager = manager or ModelManager(
        deepseek_config=DeepSeekConfig(temperature=0.0, max_tokens=2048),
        qwen_config=QwenConfig(max_tokens=1024),
    )
    qwen = manager.get_qwen()
    counter = TokenCounter.from_qwen(qwen)
    budget = BudgetManager.from_run(baseline, counter)
    config = {
        **source_config, "baseline_dir": str(baseline), "ratio": budget.config.ratio,
        "baseline_tokens": budget.config.baseline_tokens,
        "budget_tokens": budget.config.budget_tokens,
        "batch_size": budget.config.batch_size,
        "text_filter_prompt": TEXT_FILTER_SYSTEM_PROMPT,
        "second_filter_prompt": SECOND_FILTER_SYSTEM_PROMPT,
    }
    config_file = output / "agent_config.json"
    if config_file.exists() and read_json(config_file) != config:
        raise ValueError("Experiment configuration changed; use a new output directory")
    write_json(config_file, config)
    inspection = budget.inspect_run(baseline, output / "baseline_tokens.json")
    first_over = inspection["first_over_budget_period"]
    print(f"Full budget: {budget.config.baseline_tokens}; 90% limit: "
          f"{budget.config.budget_tokens}; first trigger: {first_over}", flush=True)
    if inspect_only:
        return inspection

    baseline_scores = read_json(baseline / "overall_results.json")
    score_file = output / "overall_results.json"
    scores = read_json(score_file) if score_file.exists() else [
        [None] * len(period) for period in baseline_scores]
    snapshots = sorted((baseline / "agent_states").glob("period_*/memory.json"))
    for source in snapshots:
        period = source.parent.name
        index = int(period.removeprefix("period_"))
        if stop_after_period is not None and index > stop_after_period:
            break
        target = output / "agent_states" / period / "memory.json"
        if target.exists() and all(row is not None for row in scores[index]):
            continue
        interactions = read_json(baseline / "interactions" / f"{period}.json")
        write_json(output / "interactions" / f"{period}.json", interactions)
        completed = [f"period_{i:02d}" for i in range(index + 1)]

        if first_over is None or period < first_over:
            data = read_json(source)
            write_json(target, data)
            budget.record_period(period, data["nodes"], output / "tokens.json")
            scores[index] = baseline_scores[index]
            save_scores(output, scores)
            print(f"{period}: reused full-budget state and evaluation", flush=True)
            continue

        embedding = manager.get_embedding()
        if target.exists():
            data = read_json(target)
            nodes, groups = load_memories(target, embedding)
            state = data["budget"]
        else:
            if period == first_over:
                # All memories of the trigger period already exist in this snapshot.
                nodes, groups = load_memories(source, embedding)
                filtered_ids = []
                second_filtered_ids = []
            else:
                previous = output / "agent_states" / f"period_{index - 1:02d}" / "memory.json"
                nodes, groups = load_memories(previous, embedding)
                previous_state = read_json(previous)["budget"]
                filtered_ids = previous_state["filtered_ids"]
                second_filtered_ids = previous_state["second_filtered_ids"]
                matcher = MemoryMatcher(manager.get_deepseek())
                records = [msg for session in interactions for msg in session if msg["role"] == "user"]
                for number, msg in enumerate(records, 1):
                    add_memory(msg["content"], nodes, groups, qwen, embedding,
                               matcher, period_index=index)
                    print(f"{period}: ingest {number}/{len(records)}", flush=True)
            counts = counter.count_memories({key: vars(node) for key, node in nodes.items()})
            state = {
                "before_tokens": counts["total_tokens"], "filtered_ids": filtered_ids,
                "second_filtered_ids": second_filtered_ids,
                "batches": [], "ready_for_evaluation": False,
            }
            save_checkpoint(target, nodes, groups, completed, state)

        filtered_ids = set(state["filtered_ids"])
        second_filtered_ids = set(state["second_filtered_ids"])
        try:
            for batch in budget.filter_batches(
                    nodes, filtered_ids, second_filtered_ids, qwen, embedding):
                state["filtered_ids"] = sorted(filtered_ids)
                state["second_filtered_ids"] = sorted(second_filtered_ids)
                state["batches"].append(batch)
                save_checkpoint(target, nodes, groups, completed, state)
                print(f"{period}: pass {batch['pass_index']}, batch {len(state['batches'])}, "
                      f"tokens {batch['before_tokens']} -> {batch['after_tokens']}", flush=True)
        finally:
            result = budget.record_period(
                period, {key: vars(node) for key, node in nodes.items()}, output / "tokens.json")
        state["ready_for_evaluation"] = True
        save_checkpoint(target, nodes, groups, completed, state)
        print(f"{period}: within budget ({result['total_tokens']}); evaluating", flush=True)
        run_evaluation(target, period, output_file=score_file, manager=manager,
                       nodes=nodes, groups=groups,
                       benchmark_root=source_config["benchmark_root"],
                       item_id=source_config["item_id"], benchmark_output=True)
        scores = read_json(score_file)

    # Report only completed periods; unfinished periods remain None.
    tokens = read_json(output / "tokens.json") if (output / "tokens.json").exists() else {}
    comparison = {}
    for period, usage in tokens.items():
        index = int(period.removeprefix("period_"))
        if any(row is None for row in scores[index]):
            continue
        baseline_usage = inspection["periods"][period]
        comparison[period] = {
            "baseline_tokens": baseline_usage["total_tokens"],
            "budget_run_tokens": usage["total_tokens"],
            "budget_tokens": budget.config.budget_tokens,
            "reused": first_over is None or period < first_over,
        }
        for metric in ("accuracy", "jaccard", "hamming", "accurate_retrieval", "selective_forgetting"):
            values = [row["scores"][metric] for row in scores[index]
                      if row["scores"][metric] is not None]
            original = [row["scores"][metric] for row in baseline_scores[index]
                        if row["scores"][metric] is not None]
            comparison[period][metric] = {
                "baseline": sum(original) / len(original) if original else None,
                "budget_run": sum(values) / len(values) if values else None,
            }
    write_json(output / "comparison.json", comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--stop-after-period", type=int)
    args = parser.parse_args()
    run_experiment(args.baseline_dir, args.output_dir,
                   inspect_only=args.inspect_only, stop_after_period=args.stop_after_period)


if __name__ == "__main__":
    main()
