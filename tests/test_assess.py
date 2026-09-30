"""离线验证选项编号、评分接口及 period 结果保存，不调用真实模型。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from assess import benchmark
from memory.memory_node import MemoryNode


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.manager = Mock()
        self.manager.get_embedding.return_value.encode.side_effect = (
            lambda texts: np.array([1., 0.]) if isinstance(texts, str)
            else np.array([[1., 0.] for _ in texts])
        )
        self.manager.get_qwen.return_value.chat.return_value = '{"topics": []}'
        self.memories = [MemoryNode(
            memory_id=str(i), summary=f"Memory {i}", updated_at="now", embedding=[1., 0.]
        ) for i in range(6)]
        self.qa = {
            "query": "Which state?", "required_info": ["value"],
            "answer_choices": [
                {"answer": f"Option {i}", "state": [f"state_{i}"]} for i in range(1, 6)
            ],
        }

    def test_first_middle_last_number_maps_to_correct_choice_and_score(self):
        for number in (1, 3, 5):
            with self.subTest(number=number):
                self.manager.get_deepseek.return_value.chat.return_value = json.dumps(
                    {"answer": number, "reason": "Evidence from memory 0"}
                )
                score = Mock(return_value={"accuracy": 1.0})
                result = benchmark.evaluate_question(
                    self.qa, {"state": {"value": f"state_{number}"}},
                    {"state": {"value": "old_state"}}, self.memories, self.manager, score,
                )
                self.assertEqual(result["answer"], number - 1)
                self.assertEqual(result["response"], number)
                self.assertEqual(result["response_choice"], f"Option {number}")
                score.assert_called_once_with([f"state_{number}"], [f"state_{number}"], ["old_state"])
                payload = json.loads(self.manager.get_deepseek.return_value.chat.call_args.args[0][1]["content"])
                self.assertEqual([c["id"] for c in payload["answer_choices"]], [1, 2, 3, 4, 5])
                self.assertEqual(
                    payload["answer_choices"],
                    [
                        {"id": i, "state": [f"state_{i}"]}
                        for i in range(1, 6)
                    ],
                )
                self.assertEqual(len(payload["memories"]), 5)
                self.assertNotIn("embedding", json.dumps(payload))
                self.assertNotIn("embedding", json.dumps(result))

    def test_wrong_choice_uses_predicted_state_not_golden_state(self):
        self.manager.get_deepseek.return_value.chat.return_value = '{"answer": 2, "reason": "Evidence"}'
        score = Mock(return_value={"accuracy": 0.0})
        result = benchmark.evaluate_question(
            self.qa, {"state": {"value": "state_1"}}, None,
            self.memories, self.manager, score,
        )
        score.assert_called_once_with(["state_1"], ["state_2"], None)
        self.assertEqual(result["scores"]["accuracy"], 0.0)

    def test_invalid_numbers_never_index_choices(self):
        for number in (0, -1, 6, True, "1"):
            with self.subTest(number=number):
                self.manager.get_deepseek.return_value.chat.return_value = json.dumps(
                    {"answer": number, "reason": "Evidence"}
                )
                score = Mock()
                with self.assertRaises(ValueError):
                    benchmark.evaluate_question(self.qa, {}, None, self.memories, self.manager, score)
                score.assert_not_called()

    def test_real_dataset_and_metrics_preserve_other_periods_and_source(self):
        memory_file = benchmark.ROOT / "data/period_01.json"
        before = memory_file.read_bytes()
        self.manager.get_deepseek.return_value.chat.return_value = '{"answer": 1, "reason": "Mock evidence"}'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "overall_results.json"
            with patch.object(benchmark, "ModelManager", return_value=self.manager), patch("builtins.print"):
                first = benchmark.run_evaluation(memory_file, "period_01", output_file=output)
                second = benchmark.run_evaluation(memory_file, "period_02", output_file=output)
            saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(saved, second)
        self.assertEqual(saved[1], first[1])
        self.assertTrue(all(row is None for row in saved[0]))
        self.assertTrue(all(row is not None for row in saved[2]))
        self.assertEqual(set(saved[2][0]["scores"]), {
            "accuracy", "hamming", "jaccard", "accurate_retrieval", "selective_forgetting",
        })
        self.assertNotIn("embedding", json.dumps(saved))
        self.assertEqual(memory_file.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
