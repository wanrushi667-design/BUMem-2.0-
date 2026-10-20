"""Budget accounting with deterministic token IDs; no model loading required."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from budget import BudgetConfig, BudgetManager, TokenCounter


def memory(text, topics):
    return {"raw_text": text, "topics": topics}


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.counter = TokenCounter(str.split)
        self.nodes = {
            "m1": memory("one two three", ["team work", "audit"]),
            "m2": memory("four five", ["team work"]),
        }
        self.nodes["m1"]["group_ids"] = ["g1", "g2"]

    def test_all_node_topics_are_counted_independently_of_groups(self):
        counts = self.counter.count_memories(self.nodes)
        self.assertEqual(counts, {
            "raw_text_by_memory": {"m1": 3, "m2": 2},
            "topics_by_memory": {"m1": 3, "m2": 2},
            "raw_text_tokens": 5, "topics_tokens": 5, "total_tokens": 10,
        })
        self.assertEqual(self.counter.count_memories({})["total_tokens"], 0)

    def test_budget_boundary_and_decimal_rounding(self):
        manager = BudgetManager(BudgetConfig(100), self.counter)
        self.assertFalse(manager.check({"total_tokens": 90})["is_over_budget"])
        result = manager.check({"total_tokens": 91})
        self.assertTrue(result["is_over_budget"])
        self.assertEqual(result["excess_tokens"], 1)
        self.assertEqual(BudgetConfig(90, ratio=0.7).budget_tokens, 63)

    def test_qwen_uses_plain_text_tokenization(self):
        client = Mock()
        client.model.tokenize.return_value = [1, 2]
        self.assertEqual(TokenCounter.from_qwen(client).count_text("中文"), 2)
        client.model.tokenize.assert_called_once_with(
            "中文".encode("utf-8"), add_bos=False, special=False)

    def test_baseline_is_final_snapshot_and_reports_replace_periods(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {}
            for index, text in enumerate(("one two", "one two three four five")):
                path = root / "agent_states" / f"period_{index:02d}" / "memory.json"
                path.parent.mkdir(parents=True)
                contents = json.dumps({"nodes": {"m1": memory(text, [])}, "groups": {}})
                path.write_text(contents, encoding="utf-8")
                sources[path] = contents
            manager = BudgetManager.from_run(root, self.counter)
            self.assertEqual(manager.config.baseline_tokens, 5)
            self.assertEqual(manager.config.budget_tokens, 4)
            report = root / "budget" / "tokens.json"
            result = manager.inspect_run(root, report)
            self.assertEqual(result["first_over_budget_period"], "period_01")
            self.assertFalse(result["periods"]["period_00"]["is_over_budget"])
            manager.inspect_run(root, report)
            self.assertEqual(len(json.loads(report.read_text())), 2)
            # A future filtering pass updates the same period rather than appending.
            manager.record_period("period_01", {"m1": memory("one", [])}, report)
            saved = json.loads(report.read_text())
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved["period_01"]["total_tokens"], 1)
            self.assertFalse(saved["period_01"]["is_over_budget"])
            for path, contents in sources.items():
                self.assertEqual(path.read_text(encoding="utf-8"), contents)

    def test_character_baselines_are_independent(self):
        with TemporaryDirectory() as directory:
            for role, count in (("first", 10), ("second", 20)):
                root = Path(directory) / role
                path = root / "agent_states/period_00/memory.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "nodes": {"m1": memory(" ".join(["word"] * count), [])}, "groups": {},
                }), encoding="utf-8")
                manager = BudgetManager.from_run(root, self.counter)
                self.assertEqual(manager.config.budget_tokens, int(count * 0.9))


if __name__ == "__main__":
    unittest.main()
