"""Verify replay, scoring and benchmark output layout without model calls."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

spec = importlib.util.spec_from_file_location(
    "replay", Path(__file__).with_name("test_amemgym_replay.py"))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ReplayFlowTests(unittest.TestCase):
    def test_two_periods_and_state_options(self):
        self.check_role(replay.ITEM_ID)

    def test_first_character(self):
        self.check_role("e8bbeec8-7398-4751-8ca4-dd65bbd2d31d")

    def test_third_character(self):
        self.check_role("7b567597-7ba1-463f-aea1-8aea872098f0")

    def check_role(self, item_id):
        manager = Mock()
        manager.get_embedding.return_value.encode.return_value = np.array([1.0, 0.0])
        reply = {"answer_id": None, "answer_status": "insufficient",
                 "current_state": "No evidence", "evidence_ids": []}
        with TemporaryDirectory() as directory, \
             patch.object(replay, "add_memory", return_value=None) as ingest, \
             patch("assess.benchmark.process_query", side_effect=lambda text, *_: {"query": text}), \
             patch("assess.benchmark.retrieve_memories", return_value=[]), \
             patch("assess.benchmark.select_answer", return_value=reply) as answer:
            output = Path(directory)
            def run(period):
                replay.run_period(period, output, replay.SOURCE.parent / item_id,
                                  manager=manager, item_id=item_id)
            with self.assertRaises(ValueError):
                run(1)
            for period in (0, 1):
                run(period)
            count = ingest.call_count
            run(0)
            run(1)
            self.assertEqual(ingest.call_count, count)
            config = json.loads((output / "agent_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["item_id"], item_id)
            with self.assertRaises(ValueError):
                replay.run_period(0, output, manager=manager, item_id="other-character")
            for call in ingest.call_args_list:
                self.assertIn(call.kwargs["period_index"], (0, 1))
            self.assertTrue(all("state" in choice for call in answer.call_args_list
                                for choice in call.args[1]))
            results = json.loads((output / "overall_results.json").read_text(encoding="utf-8"))
            self.assertTrue(all(row["scores"]["accuracy"] == 0 for period in results[:2] for row in period))
            self.assertTrue(all(row is None for row in results[2]))
            metrics = json.loads((output / "overall_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["accuracy"][0], [row["scores"]["accuracy"] for row in results[0]])
            summary = json.loads((output / "overall_metrics_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["accuracy"]["applicable_count"], sum(map(len, results[:2])))
            self.assertNotIn("protocol", summary)
            self.assertEqual({p.name for p in output.iterdir()}, {
                "agent_config.json", "overall_results.json", "overall_metrics.json",
                "overall_metrics_summary.json", "agent_states", "interactions",
            })
            for period in (0, 1):
                snapshot = output / "agent_states" / f"period_{period:02d}"
                self.assertEqual({p.name for p in snapshot.iterdir()}, {"memory.json"})
                self.assertTrue((output / "interactions" / f"period_{period:02d}.json").is_file())
            self.assertEqual(json.loads((output / "agent_states/period_00/memory.json").read_text(
                encoding="utf-8"))["completed_periods"], ["period_00"])


if __name__ == "__main__":
    unittest.main()
