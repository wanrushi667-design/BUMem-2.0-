"""Exercise filtering, continuation and resume without model or API calls."""

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from budget import BudgetConfig, BudgetManager, TokenCounter
from budget.candidate_selector import select_candidates, select_second_candidates
from budget.run import run_experiment
from budget.text_filter import filter_text, TEXT_FILTER_SYSTEM_PROMPT, SECOND_FILTER_SYSTEM_PROMPT
from budget.validator import validate_filter
from memory.node_group import MemoryNode


def node(key, period, length):
    return MemoryNode(key, [], ["topic"], " ".join(["word"] * length),
                      "2026-01-01", [1.0, 0.0], period)


def scored(value):
    return {"scores": {metric: value for metric in (
        "accuracy", "jaccard", "hamming", "accurate_retrieval", "selective_forgetting")},
        "json_error": False}


class BudgetFlowTests(unittest.TestCase):
    def setUp(self):
        self.counter = TokenCounter(str.split)

    def test_candidate_order_stays_in_oldest_period_and_excludes_attempted(self):
        nodes = {f"m{i}": node(f"m{i}", 0, i + 1) for i in range(12)}
        nodes["m99"] = node("m99", 1, 100)
        lengths = {key: self.counter.count_text(n.raw_text) for key, n in nodes.items()}
        selected = select_candidates(nodes, {"m11"}, lengths)
        self.assertEqual([n.memory_id for n in selected], [f"m{i}" for i in range(10, 0, -1)])
        remaining = select_candidates(nodes, {f"m{i}" for i in range(1, 12)}, lengths)
        self.assertEqual([n.memory_id for n in remaining], ["m0"])

    def test_multiple_batches_recount_until_within_limit(self):
        nodes = {f"m{i}": node(f"m{i}", 0, 4) for i in range(12)}
        qwen = Mock()
        qwen.chat.return_value = '{"raw_text": "word"}'
        embedding = Mock()
        embedding.encode.return_value = np.array([0.0, 1.0])
        filtered = set()
        manager = BudgetManager(BudgetConfig(30), self.counter)
        batches = list(manager.filter_batches(nodes, filtered, set(), qwen, embedding))
        self.assertEqual([len(batch["changes"]) for batch in batches], [10, 2])
        self.assertEqual(batches[-1]["after_tokens"], 24)  # 12 text + 12 topic tokens
        self.assertEqual(len(filtered), 12)
        self.assertEqual(embedding.encode.call_count, 12)
        self.assertEqual(nodes["m0"].embedding, [0.0, 1.0])
        self.assertEqual(nodes["m0"].topics, ["topic"])
        self.assertEqual(list(manager.filter_batches(nodes, filtered, set(), qwen, embedding)), [])

    def test_validator_keeps_original_for_empty_equal_or_longer_text(self):
        for proposed in ("", "three four", "one two three"):
            result = validate_filter("one two", proposed, self.counter)
            self.assertFalse(result["accepted"])
            self.assertEqual(result["saved_tokens"], 0)
        self.assertTrue(validate_filter("one two", "one", self.counter)["accepted"])

    def test_unsuccessful_filter_is_marked_and_never_reselected(self):
        nodes = {"m1": node("m1", 0, 5), "m2": node("m2", 1, 10)}
        qwen = Mock()
        qwen.chat.side_effect = [json.dumps({"raw_text": nodes["m1"].raw_text}),
                                 '{"raw_text": "word"}']
        embedding = Mock()
        embedding.encode.return_value = np.array([1.0, 1.0])
        filtered = set()
        batches = list(BudgetManager(BudgetConfig(12), self.counter).filter_batches(
            nodes, filtered, set(), qwen, embedding))
        self.assertEqual(len(batches), 2)
        self.assertFalse(batches[0]["changes"][0]["accepted"])
        self.assertEqual(filtered, {"m1", "m2"})
        self.assertEqual(embedding.encode.call_count, 1)

    def test_qwen_filter_prompt_and_response_contract(self):
        qwen = Mock()
        qwen.chat.return_value = '{"raw_text":" shortened "}'
        self.assertEqual(filter_text("original", qwen), "shortened")
        messages = qwen.chat.call_args.args[0]
        self.assertEqual(messages[0]["content"], TEXT_FILTER_SYSTEM_PROMPT)
        self.assertTrue(qwen.chat.call_args.kwargs["json_mode"])
        for response in ('{"topics":[]}', '{"raw_text":null}'):
            qwen.chat.return_value = response
            with self.assertRaises(ValueError):
                filter_text("original", qwen)

    def test_time_header_is_preserved_without_relying_on_model(self):
        qwen = Mock()
        qwen.chat.return_value = '{"raw_text":"concise body"}'
        header = "[Current Time: 2020-09-22 20:15:06]\n"
        self.assertEqual(filter_text(header + "long original body", qwen), header + "concise body")
        self.assertEqual(qwen.chat.call_args.args[0][1]["content"], "long original body")
        qwen.chat.return_value = '{"raw_text":""}'
        self.assertEqual(filter_text(header + "body", qwen), "")

    def test_second_pass_is_global_uses_current_text_and_recounts_each_batch(self):
        nodes = {"m1": node("m1", 0, 10), "m2": node("m2", 1, 12)}
        qwen = Mock()
        qwen.chat.side_effect = [
            json.dumps({"raw_text": " ".join(["first"] * 5)}),
            json.dumps({"raw_text": nodes["m2"].raw_text}),
            json.dumps({"raw_text": " ".join(["second"] * 3)}),
            json.dumps({"raw_text": "second"}),
        ]
        embedding = Mock()
        embedding.encode.return_value = np.array([0.0, 1.0])
        first_ids, second_ids = set(), set()
        manager = BudgetManager(BudgetConfig(8, batch_size=1), self.counter)
        batches = list(manager.filter_batches(nodes, first_ids, second_ids, qwen, embedding))
        self.assertEqual([b["pass_index"] for b in batches], [1, 1, 2, 2])
        self.assertEqual([b["changes"][0]["memory_id"] for b in batches],
                         ["m1", "m2", "m2", "m1"])
        calls = qwen.chat.call_args_list
        self.assertEqual(calls[2].args[0][0]["content"], SECOND_FILTER_SYSTEM_PROMPT)
        self.assertEqual(calls[3].args[0][1]["content"], " ".join(["first"] * 5))
        self.assertEqual(first_ids, {"m1", "m2"})
        self.assertEqual(second_ids, {"m1", "m2"})
        self.assertEqual(batches[-1]["after_tokens"], 6)
        self.assertEqual(nodes["m1"].topics, ["topic"])

    def test_second_pass_selection_excludes_attempted_and_crosses_periods(self):
        nodes = {"m1": node("m1", 0, 3), "m2": node("m2", 2, 8),
                 "m3": node("m3", 1, 6)}
        lengths = {key: len(n.raw_text.split()) for key, n in nodes.items()}
        selected = select_second_candidates(nodes, {"m2"}, lengths)
        self.assertEqual([n.memory_id for n in selected], ["m3", "m1"])

    def test_two_pass_exhaustion_stops_without_retrying_or_changing_text(self):
        nodes = {"m1": node("m1", 0, 5)}
        original = nodes["m1"].raw_text
        qwen = Mock()
        qwen.chat.return_value = json.dumps({"raw_text": original})
        embedding = Mock()
        first_ids, second_ids = set(), set()
        batches = BudgetManager(BudgetConfig(2), self.counter).filter_batches(
            nodes, first_ids, second_ids, qwen, embedding)
        self.assertEqual(next(batches)["pass_index"], 1)
        self.assertEqual(next(batches)["pass_index"], 2)
        with self.assertRaisesRegex(RuntimeError, "Both compression passes exhausted"):
            next(batches)
        self.assertEqual(qwen.chat.call_count, 2)
        self.assertEqual(nodes["m1"].raw_text, original)
        embedding.encode.assert_not_called()

    def test_new_period_gets_first_pass_before_remaining_second_pass_candidates(self):
        nodes = {"m1": node("m1", 0, 9), "m2": node("m2", 1, 5)}
        qwen = Mock()
        qwen.chat.side_effect = ['{"raw_text":"short new"}', '{"raw_text":"old"}']
        embedding = Mock()
        embedding.encode.return_value = np.array([1.0, 0.0])
        first_ids, second_ids = {"m1"}, set()
        batches = list(BudgetManager(BudgetConfig(6, batch_size=1), self.counter).filter_batches(
            nodes, first_ids, second_ids, qwen, embedding))
        self.assertEqual([b["pass_index"] for b in batches], [1, 2])
        self.assertEqual([b["changes"][0]["memory_id"] for b in batches], ["m2", "m1"])

    def test_experiment_reuses_then_filters_and_ingests_from_compressed_state(self):
        with TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline"
            output = Path(directory) / "budget90"
            baseline.mkdir()
            config = {"item_id": "first-character", "benchmark_root": directory}
            (baseline / "agent_config.json").write_text(json.dumps(config))
            (baseline / "overall_results.json").write_text(json.dumps([[scored(1)]] * 3))
            original = {}
            for i in range(3):
                nodes = {"m1": node("m1", 0, 20)}
                if i >= 1:
                    nodes["m2"] = node("m2", 1, 18)
                if i >= 2:
                    nodes["m3"] = node("m3", 2, 2)
                path = baseline / "agent_states" / f"period_{i:02d}" / "memory.json"
                path.parent.mkdir(parents=True)
                data = {
                    "nodes": {key: {k: v for k, v in asdict(n).items() if k != "embedding"}
                              for key, n in nodes.items()},
                    "groups": {}, "associations": [],
                    "completed_periods": [f"period_{j:02d}" for j in range(i + 1)],
                }
                path.write_text(json.dumps(data))
                interaction = baseline / "interactions" / f"period_{i:02d}.json"
                interaction.parent.mkdir(exist_ok=True)
                interaction.write_text(json.dumps([[{"role": "user", "content": "new dialogue"}]]))
            for path in baseline.rglob("*.json"):
                original[path] = path.read_bytes()

            manager = Mock()
            manager.get_qwen().chat.side_effect = [
                json.dumps({"raw_text": " ".join(["word"] * 20)}),
                json.dumps({"raw_text": " ".join(["word"] * 18)}),
                json.dumps({"raw_text": " ".join(["filtered"] * 6)}),
                json.dumps({"raw_text": " ".join(["word"] * 18)}),
                json.dumps({"raw_text": " ".join(["word"] * 20)}),
                json.dumps({"raw_text": " ".join(["filtered"] * 4)}),
            ]
            manager.get_embedding().encode.side_effect = lambda texts: (
                np.array([[1.0, 0.0]] * len(texts)) if isinstance(texts, list)
                else np.array([1.0, 0.0]))

            def ingest(text, nodes, groups, qwen, embedding, matcher, *, period_index):
                self.assertEqual(len(nodes["m1"].raw_text.split()), 6)
                self.assertEqual(period_index, 2)
                nodes["m3"] = node("m3", 2, 20)

            evaluation_attempts = []

            def evaluate(memory_file, period, *, output_file, **kwargs):
                data = json.loads(Path(memory_file).read_text())
                self.assertTrue(data["budget"]["ready_for_evaluation"])
                self.assertLessEqual(self.counter.count_file(memory_file)["total_tokens"], 38)
                evaluation_attempts.append(period)
                if len(evaluation_attempts) == 1:
                    raise RuntimeError("Simulated evaluation interruption")
                scores = json.loads(Path(output_file).read_text())
                scores[int(period[-2:])] = [scored(0.5)]
                Path(output_file).write_text(json.dumps(scores))

            with patch("budget.run.TokenCounter.from_qwen", return_value=self.counter), \
                 patch("budget.run.add_memory", side_effect=ingest) as add, \
                 patch("budget.run.run_evaluation", side_effect=evaluate) as evaluation:
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    run_experiment(baseline, output, manager=manager)
                report = run_experiment(baseline, output, manager=manager)
                self.assertTrue(report["period_00"]["reused"])
                self.assertEqual(report["period_01"]["budget_run_tokens"], 26)
                self.assertEqual(report["period_02"]["budget_run_tokens"], 31)
                self.assertEqual(add.call_count, 1)
                self.assertEqual(evaluation.call_count, 3)
                run_experiment(baseline, output, manager=manager)
                self.assertEqual(add.call_count, 1)
                self.assertEqual(evaluation.call_count, 3)
                self.assertEqual(manager.get_qwen().chat.call_count, 6)
            final = json.loads((output / "agent_states/period_02/memory.json").read_text())
            self.assertEqual(final["budget"]["filtered_ids"], ["m1", "m2", "m3"])
            self.assertEqual(final["budget"]["second_filtered_ids"], ["m1", "m2", "m3"])
            for path, contents in original.items():
                self.assertEqual(path.read_bytes(), contents)


if __name__ == "__main__":
    unittest.main()
