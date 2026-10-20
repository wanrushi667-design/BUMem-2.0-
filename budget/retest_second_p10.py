"""Continue the existing second-character period_10 with the second pass only.

Run explicitly with: python -m budget.retest_second_p10
This preserves the original first-pass prompt and earlier period results.
"""

import hashlib
from pathlib import Path
import shutil

from assess.benchmark import run_evaluation, write_json
from config.model_config import DeepSeekConfig, QwenConfig
from memory.storage import load_memories
from models import ModelManager

from .manager import BudgetManager
from .run import ROOT, read_json, save_checkpoint
from .text_filter import SECOND_FILTER_SYSTEM_PROMPT
from .token_counter import TokenCounter


OUTPUT = ROOT / "eval-runs/amemgym-second-role-budget90"


def main():
    output = OUTPUT
    target = output / "agent_states/period_10/memory.json"
    config = read_json(output / "agent_config.json")
    baseline = Path(config["baseline_dir"])
    backup = output / "logs/p10-before-second-pass"
    if not backup.exists():
        backup.mkdir()
        for name in ("agent_config.json", "run_summary.json", "tokens.json",
                     "overall_results.json", "overall_metrics.json", "overall_metrics_summary.json"):
            shutil.copy2(output / name, backup / name)
        shutil.copy2(target, backup / "memory.json")
        write_json(backup / "earlier_snapshot_hashes.json", {
            path.parent.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((output / "agent_states").glob("period_*/memory.json"))
            if path.parent.name != "period_10"
        })

    models = ModelManager(deepseek_config=DeepSeekConfig(temperature=0.0, max_tokens=2048),
                          qwen_config=QwenConfig(max_tokens=1024))
    qwen = models.get_qwen()
    counter = TokenCounter.from_qwen(qwen)
    budget = BudgetManager.from_run(baseline, counter)
    state = read_json(target)["budget"]
    # Explicitly initialise the new pass for this old single-pass checkpoint.
    state.setdefault("second_filtered_ids", [])
    for batch in state["batches"]:
        batch.setdefault("pass_index", 1)
    config["second_filter_prompt"] = SECOND_FILTER_SYSTEM_PROMPT
    config["continuation"] = {
        "period": "period_10", "mode": "second_pass_on_existing_first_pass",
        "first_pass_prompt": "text_filter_prompt retained from original run",
        "before_tokens": read_json(backup / "memory.json")["budget"]["batches"][-1]["after_tokens"],
    }
    write_json(output / "agent_config.json", config)
    embedding = models.get_embedding()
    nodes, groups = load_memories(target, embedding)
    first_ids, second_ids = set(state["filtered_ids"]), set(state["second_filtered_ids"])
    if first_ids != set(nodes):
        raise ValueError("This continuation requires the first pass to be exhausted")
    print(f"period_10: continuing at {counter.count_file(target)['total_tokens']} tokens; "
          f"limit {budget.config.budget_tokens}", flush=True)
    try:
        for batch in budget.filter_batches(nodes, first_ids, second_ids, qwen, embedding):
            state["second_filtered_ids"] = sorted(second_ids)
            state["batches"].append(batch)
            save_checkpoint(target, nodes, groups, read_json(target)["completed_periods"], state)
            print(f"period_10: pass {batch['pass_index']}, "
                  f"tokens {batch['before_tokens']} -> {batch['after_tokens']}", flush=True)
    finally:
        usage = budget.record_period(
            "period_10", {key: vars(n) for key, n in nodes.items()}, output / "tokens.json")
    state["ready_for_evaluation"] = True
    save_checkpoint(target, nodes, groups, read_json(target)["completed_periods"], state)
    print(f"period_10: within budget ({usage['total_tokens']}); evaluating", flush=True)
    run_evaluation(target, "period_10", output_file=output / "overall_results.json",
                   manager=models, nodes=nodes, groups=groups,
                   benchmark_root=config["benchmark_root"], item_id=config["item_id"],
                   benchmark_output=True)

    scores = read_json(output / "overall_results.json")
    original = read_json(baseline / "overall_results.json")
    baseline_tokens = read_json(output / "baseline_tokens.json")
    comparison = {}
    for period, counts in read_json(output / "tokens.json").items():
        index = int(period.removeprefix("period_"))
        row = {"baseline_tokens": baseline_tokens[period]["total_tokens"],
               "budget_run_tokens": counts["total_tokens"],
               "budget_tokens": budget.config.budget_tokens, "reused": index < 9}
        for metric in ("accuracy", "jaccard", "hamming", "accurate_retrieval", "selective_forgetting"):
            values = [x["scores"][metric] for x in scores[index] if x["scores"][metric] is not None]
            old = [x["scores"][metric] for x in original[index] if x["scores"][metric] is not None]
            row[metric] = {"baseline": sum(old) / len(old) if old else None,
                           "budget_run": sum(values) / len(values) if values else None}
        comparison[period] = row
    write_json(output / "comparison.json", comparison)
    summary = read_json(backup / "run_summary.json")
    summary["status"] = "completed_with_second_pass_continuation"
    summary["completed_evaluation_periods"] = list(range(len(scores)))
    summary["period_10"].update({
        "after_first_pass_tokens": config["continuation"]["before_tokens"],
        "after_tokens": usage["total_tokens"], "excess_tokens": usage["excess_tokens"],
        "second_pass_attempted_memories": len(second_ids), "evaluated": True,
        "baseline_accuracy": comparison["period_10"]["accuracy"]["baseline"],
        "budget_accuracy": comparison["period_10"]["accuracy"]["budget_run"],
    })
    write_json(output / "run_summary.json", summary)
    print(f"Updated results: {output}", flush=True)


if __name__ == "__main__":
    main()
