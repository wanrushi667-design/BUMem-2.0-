"""Fixed-budget checks and batch filtering for the 90% policy."""

import json
from pathlib import Path

from .config import BudgetConfig
from .candidate_selector import select_candidates, select_second_candidates
from .text_filter import filter_text, TEXT_FILTER_SYSTEM_PROMPT, SECOND_FILTER_SYSTEM_PROMPT
from .validator import validate_filter


class BudgetManager:
    def __init__(self, config: BudgetConfig, counter):
        self.config = config
        self.counter = counter

    @classmethod
    def from_run(cls, run_dir, counter, *, ratio=0.9, batch_size=10):
        """Use the final cumulative snapshot, never the sum of period snapshots."""
        snapshots = sorted((Path(run_dir) / "agent_states").glob("period_*/memory.json"))
        if not snapshots:
            raise ValueError("No period memory snapshots found")
        baseline = counter.count_file(snapshots[-1])["total_tokens"]
        return cls(BudgetConfig(baseline, ratio, batch_size), counter)

    def check(self, counts: dict) -> dict:
        current = counts["total_tokens"]
        limit = self.config.budget_tokens
        return {
            **counts,
            "baseline_tokens": self.config.baseline_tokens,
            "ratio": self.config.ratio,
            "budget_tokens": limit,
            "excess_tokens": max(0, current - limit),
            "is_over_budget": current > limit,
        }

    def record_period(self, period: str, nodes, report_file) -> dict:
        """Replace this period's cumulative count, including after filtering.

        The report belongs to one character/run. Its period totals must not be added.
        """
        result = self.check(self.counter.count_memories(nodes))
        path = Path(report_file)
        report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        report[period] = result
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def inspect_run(self, run_dir, report_file) -> dict:
        """Record existing snapshots without ingestion, evaluation or mutation.

        Return the first period requiring filtering. Later baseline snapshots are
        accounting references only, not valid continuations of a compressed run.
        """
        first_over_budget = None
        periods = {}
        for snapshot in sorted((Path(run_dir) / "agent_states").glob("period_*/memory.json")):
            data = json.loads(snapshot.read_text(encoding="utf-8"))
            period = snapshot.parent.name
            result = self.record_period(period, data["nodes"], report_file)
            periods[period] = result
            if result["is_over_budget"] and first_over_budget is None:
                first_over_budget = period
        return {"first_over_budget_period": first_over_budget, "periods": periods}

    def filter_batches(self, nodes, filtered_ids, second_filtered_ids,
                       qwen_client, embedding_client):
        """Update live nodes and yield each completed batch for checkpointing.

        Each pass marks attempts independently, even when no shorter text was
        accepted. Pass two starts only after all pass-one candidates are exhausted.
        """
        if self.config.ratio != 0.9:
            raise NotImplementedError("Only the 90% filtering policy is implemented")
        counts = self.counter.count_memories({key: vars(node) for key, node in nodes.items()})
        while self.check(counts)["is_over_budget"]:
            candidates = select_candidates(
                nodes, filtered_ids, counts["raw_text_by_memory"], self.config.batch_size)
            pass_index = 1
            attempted_ids = filtered_ids
            prompt = TEXT_FILTER_SYSTEM_PROMPT
            if not candidates:
                candidates = select_second_candidates(
                    nodes, second_filtered_ids, counts["raw_text_by_memory"], self.config.batch_size)
                pass_index = 2
                attempted_ids = second_filtered_ids
                prompt = SECOND_FILTER_SYSTEM_PROMPT
            if not candidates:
                raise RuntimeError("Both compression passes exhausted; budget is still exceeded")
            changes = []
            for node in candidates:
                proposed = filter_text(node.raw_text, qwen_client, system_prompt=prompt)
                validation = validate_filter(node.raw_text, proposed, self.counter)
                if validation["accepted"]:
                    vector = embedding_client.encode(proposed).tolist()
                    node.raw_text = proposed
                    node.embedding = vector
                attempted_ids.add(node.memory_id)
                changes.append({"memory_id": node.memory_id,
                                "period_index": node.period_index, **validation})
            before = counts["total_tokens"]
            counts = self.counter.count_memories({key: vars(node) for key, node in nodes.items()})
            yield {
                "pass_index": pass_index,
                "period_index": candidates[0].period_index if pass_index == 1 else None,
                "before_tokens": before,
                "after_tokens": counts["total_tokens"],
                "changes": changes,
            }
