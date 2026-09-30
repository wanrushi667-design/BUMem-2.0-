import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main
from memory.extractor import MemoryExtractionError, extract_memory


class MemoryExtractionTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "summary": "User is planning a workshop for beginners.",
            "topics": ["workshop", "beginner_instruction"],
            "attributes": {
                "student_experience": {"value": "complete_beginners"},
                "fee": {"value": 0},
                "cancelled": {"value": False},
            },
        }
        self.client = Mock()
        self.client.chat.return_value = json.dumps(self.payload)

    def test_fields_and_program_owned_timestamps(self):
        now = "2026-09-26T14:00:00+08:00"
        memory = extract_memory("初学者课程免费，还没有取消", self.client, now)
        self.assertEqual(set(memory.to_dict()), {
            "memory_id", "summary", "updated_at", "topics", "attributes",
            "embedding", "update_count",
        })
        self.assertEqual(memory.updated_at, now)
        self.assertIsNone(memory.embedding)
        self.assertEqual(memory.attributes["fee"]["value"], 0)
        self.assertIs(memory.attributes["cancelled"]["value"], False)
        for item in memory.attributes.values():
            self.assertEqual(set(item), {"value", "time_at"})
            self.assertEqual(item["time_at"], now)
        self.assertTrue(self.client.chat.call_args.kwargs["json_mode"])

    def test_empty_result_and_invalid_output_are_distinct(self):
        self.client.chat.return_value = '{"summary":"","topics":[],"attributes":{}}'
        self.assertIsNone(extract_memory("你好", self.client))
        for content in ("not json", "[]", '{"summary":"oops"}'):
            self.client.chat.return_value = content
            with self.assertRaises(MemoryExtractionError):
                extract_memory("输入", self.client)

    def test_main_appends_vectors_and_reuses_clients(self):
        vector = Mock()
        vector.tolist.return_value = [0.25, 0.75]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data" / "memory.json"
            with patch.object(main, "MEMORY_FILE", path), \
                 patch.object(main, "ModelManager") as manager_class, \
                 patch("sys.argv", ["main.py"]), \
                 patch("builtins.input", side_effect=["课程免费", "学生是初学者", "/exit"]), \
                 patch("builtins.print") as output:
                manager = manager_class.return_value
                manager.get_deepseek.return_value = self.client
                manager.get_embedding.return_value.encode.return_value = vector
                main.main()
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(len(saved), 2)
                self.assertNotEqual(saved[0]["memory_id"], saved[1]["memory_id"])
                self.assertEqual(saved[0]["embedding"], [0.25, 0.75])
                manager.get_deepseek.assert_called_once()
                manager.get_embedding.assert_called_once()
                manager.get_qwen.assert_not_called()
                manager.get_embedding.return_value.encode.assert_called_with(self.payload["summary"])
                self.assertNotIn("0.75", str(output.call_args_list))


if __name__ == "__main__":
    unittest.main()
