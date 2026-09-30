"""离线验证匹配流程，不调用真实模型或写入记忆库。

在项目根目录运行：python -m unittest discover -s tests -p test_memory_matcher.py -v
"""

import copy
import json
import unittest
from unittest.mock import Mock, patch

from memory.match import MemoryMatcher
from memory.memory_node import MemoryNode


def memory(memory_id, topics, embedding=(1.0, 0.0)):
    return MemoryNode(
        memory_id=memory_id,
        summary=f"Summary of {memory_id}",
        updated_at="2026-09-27T12:00:00+08:00",
        topics=topics,
        attributes={"group_size": {"value": 8, "time_at": "period0"}},
        embedding=list(embedding),
    )


class MemoryMatcherTests(unittest.TestCase):
    def setUp(self):
        self.llm = Mock()
        self.reply("CREATE")
        self.matcher = MemoryMatcher(self.llm)
        self.new = memory("new", ["bridge_gathering"])
        self.old = memory("old", ["bridge_gathering"])

    def reply(self, action, target=None):
        self.llm.chat.return_value = json.dumps({
            "action": action, "target_memory_id": target, "reason": "test decision",
        })

    def payload(self):
        args, kwargs = self.llm.chat.call_args
        self.assertTrue(kwargs["json_mode"])
        return json.loads(args[0][1]["content"])

    def test_empty_library_skips_models(self):
        result = self.matcher.find_merge_target(self.new, [])
        self.assertEqual(result.action, "CREATE")
        self.assertIsNone(result.target_memory_id)
        self.llm.chat.assert_not_called()

    def test_empty_new_topics_still_reach_llm(self):
        for topics in ([], ["  ", "___"]):
            with self.subTest(topics=topics):
                self.reply("MERGE", "old")
                result = self.matcher.find_merge_target(memory("new", topics), [self.old])
                self.assertEqual(result.target_memory_id, "old")
                self.assertEqual(self.payload()["new_memory"]["topics"], topics)
        self.assertEqual(self.llm.chat.call_count, 2)

    def test_top10_uses_summary_cosine_and_sends_all_to_llm(self):
        # Use non-unit vectors and unrelated topics to verify summary-only Top10.
        old = [memory(str(i), [f"unrelated_{i}"], (100.0, i * 100.0))
               for i in reversed(range(11))]
        self.matcher.find_merge_target(self.new, old)
        self.assertEqual([c["memory_id"] for c in self.payload()["candidates"]],
                         [str(i) for i in range(10)])
        self.assertAlmostEqual(self.matcher.last_candidates[1]["summary_score"], 2 ** -0.5)
        self.llm.chat.assert_called_once()

    def test_single_exact_match_can_be_rejected(self):
        new = memory("mentoring", ["historical_society"])
        old = memory("volunteering", ["Historical Society"])
        result = self.matcher.find_merge_target(new, [old])
        self.llm.chat.assert_called_once()
        self.assertEqual(result.action, "CREATE")
        self.assertIsNone(result.target_memory_id)

    def test_exact_and_semantic_candidates_are_sent_together(self):
        semantic = memory("semantic", ["card_night"])
        unrelated = memory("unrelated", ["rose_garden"])
        self.reply("MERGE", "semantic")
        result = self.matcher.find_merge_target(self.new, [self.old, semantic, unrelated])
        self.assertEqual([c["memory_id"] for c in self.payload()["candidates"]],
                         ["old", "semantic", "unrelated"])
        self.assertEqual((result.action, result.target_memory_id), ("MERGE", "semantic"))
        self.llm.chat.assert_called_once()

    def test_single_semantic_candidate_requires_llm(self):
        self.reply("MERGE", "semantic")
        result = self.matcher.find_merge_target(self.new, [memory("semantic", ["card_night"])])
        self.llm.chat.assert_called_once()
        self.assertEqual(result.target_memory_id, "semantic")

    def test_unrelated_topics_can_be_selected_by_llm(self):
        self.reply("MERGE", "rose")
        result = self.matcher.find_merge_target(self.new, [
            memory("rose", ["rose_garden"]), memory("other", ["opposite"]),
        ])
        self.assertEqual(result.target_memory_id, "rose")
        self.assertEqual([c["memory_id"] for c in self.payload()["candidates"]],
                         ["rose", "other"])
        self.llm.chat.assert_called_once()

    def test_empty_old_topics_still_reach_llm(self):
        self.reply("MERGE", "empty")
        result = self.matcher.find_merge_target(self.new, [memory("empty", [])])
        self.assertEqual(result.target_memory_id, "empty")
        self.assertEqual(self.payload()["candidates"][0]["topics"], [])
        self.llm.chat.assert_called_once()

    def test_full_semantic_payload_and_no_input_mutation(self):
        before = copy.deepcopy([self.new.to_dict(), self.old.to_dict()])
        self.matcher.find_merge_target(self.new, [self.old])
        payload = self.payload()
        for sent, original in zip([payload["new_memory"], payload["candidates"][0]], before):
            self.assertEqual(sent, {k: v for k, v in original.items() if k != "embedding"})
        self.assertEqual([self.new.to_dict(), self.old.to_dict()], before)

    def test_invalid_llm_responses_raise_instead_of_create(self):
        responses = [
            "not json", "[]", '{}',
            json.dumps({"action": "MERGE", "target_memory_id": "unknown", "reason": "x"}),
            json.dumps({"action": "CREATE", "target_memory_id": "old", "reason": "x"}),
            json.dumps({"action": "OTHER", "target_memory_id": None, "reason": "x"}),
            json.dumps({"action": "CREATE", "target_memory_id": None, "reason": 123}),
        ]
        for response in responses:
            with self.subTest(response=response):
                self.llm.chat.return_value = response
                with self.assertRaises(ValueError):
                    self.matcher.find_merge_target(self.new, [self.old])

    def test_llm_failure_propagates(self):
        self.llm.chat.side_effect = RuntimeError("API unavailable")
        with self.assertRaisesRegex(RuntimeError, "API unavailable"):
            self.matcher.find_merge_target(self.new, [self.old])

    def test_missing_summary_embedding_raises(self):
        for missing in (self.new, self.old):
            with self.subTest(memory_id=missing.memory_id):
                original = missing.embedding
                missing.embedding = None
                with self.assertRaisesRegex(ValueError, "summary embedding"):
                    self.matcher.find_merge_target(self.new, [self.old])
                missing.embedding = original
        self.llm.chat.assert_not_called()

    def test_default_deepseek_is_created_lazily_and_reused(self):
        with patch("memory.match.Settings.from_env") as settings, \
             patch("memory.match.DeepSeekClient", return_value=self.llm) as client:
            matcher = MemoryMatcher()
            matcher.find_merge_target(self.new, [])
            client.assert_not_called()
            matcher.find_merge_target(self.new, [self.old])
            matcher.find_merge_target(self.new, [self.old])
            client.assert_called_once()
            settings.assert_called_once()
            self.assertEqual(client.call_args.args[1].temperature, 0.0)

    def test_diagnostics_reset_between_calls(self):
        self.matcher.find_merge_target(self.new, [self.old])
        self.assertTrue(self.matcher.last_candidates)
        self.matcher.find_merge_target(self.new, [])
        self.assertEqual(self.matcher.last_candidates, [])


if __name__ == "__main__":
    unittest.main()
