"""离线验证字段对齐与完整记忆合并，不加载真实模型。"""

import json
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np

from memory.memory_node import MemoryNode
from memory.merge import align_attributes, align_topics, merge_memory


class TopicAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.embedding = Mock()
        vectors = {
            "old": [2, 0], "alias": [3, 4],
            "unrelated": [0, 2], "other": [4, 0],
        }
        self.embedding.encode.side_effect = lambda texts: [vectors[t] for t in texts]
        self.llm = Mock()
        self.llm.chat.return_value = '{"relation": "equivalent"}'

    def align(self, old, new, **kwargs):
        return align_topics(old, new, self.embedding, self.llm, **kwargs)

    def test_exact_and_empty_skip_models(self):
        for old, new, expected in [
            (["old", "old"], ["old"], ["old"]),
            ([], ["alias", "alias"], ["alias"]),
            (["old"], [], ["old"]),
        ]:
            with self.subTest(old=old, new=new):
                self.assertEqual(self.align(old, new), expected)
        self.embedding.encode.assert_not_called()
        self.llm.chat.assert_not_called()

    def test_threshold_inclusive_and_equivalent_keeps_old(self):
        self.assertEqual(self.align(["old"], ["alias"]), ["old"])
        self.llm.chat.assert_called_once()
        self.assertTrue(self.llm.chat.call_args.kwargs["json_mode"])

    def test_below_threshold_keeps_both_without_llm(self):
        self.assertEqual(self.align(["old"], ["alias"], threshold=0.61),
                         ["old", "alias"])
        self.llm.chat.assert_not_called()

    def test_different_keeps_both(self):
        self.llm.chat.return_value = '{"relation": "different"}'
        self.assertEqual(self.align(["old"], ["alias"]), ["old", "alias"])

    def test_checks_more_than_one_candidate(self):
        self.llm.chat.side_effect = [
            '{"relation": "different"}', '{"relation": "equivalent"}',
        ]
        self.assertEqual(self.align(["old", "other"], ["alias"]), ["old", "other"])
        self.assertEqual(self.llm.chat.call_count, 2)

    def test_equivalent_stops_checks(self):
        self.assertEqual(self.align(["old", "other"], ["alias"]), ["old", "other"])
        self.llm.chat.assert_called_once()

    def test_order_dedup_no_new_new_alignment_or_mutation(self):
        old = ["unrelated", "unrelated"]
        new = ["old", "alias", "old"]
        self.assertEqual(self.align(old, new, threshold=0.9),
                         ["unrelated", "old", "alias"])
        self.assertEqual(old, ["unrelated", "unrelated"])
        self.assertEqual(new, ["old", "alias", "old"])
        self.llm.chat.assert_not_called()

    def test_invalid_model_result_raises(self):
        for response in ('not json', '[]', '{}', '{"relation": "unknown"}'):
            with self.subTest(response=response):
                self.llm.chat.return_value = response
                with self.assertRaises(ValueError):
                    self.align(["old"], ["alias"])

    def test_model_failure_propagates(self):
        self.llm.chat.side_effect = RuntimeError("unavailable")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.align(["old"], ["alias"])


class AttributeAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.embedding = Mock()
        vectors = {
            "group size": [2, 0], "gathering size": [3, 4],
            "location": [0, 2], "guest age range": [0, -2],
        }
        self.embedding.encode.side_effect = lambda texts: [vectors[t] for t in texts]
        self.llm = Mock()
        self.llm.chat.return_value = '{"relation": "equivalent"}'

    def align(self, old, new, **kwargs):
        return align_attributes(old, new, self.embedding, self.llm, **kwargs)

    def test_exact_update_retains_omitted_attribute(self):
        old = {"group_size": "medium_6_12", "guest_age_range": "mostly_50_plus"}
        new = {"guest_age_range": "mixed_ages_with_young_adults"}
        self.assertEqual(self.align(old, new), {
            "group_size": "medium_6_12", "guest_age_range": new["guest_age_range"],
        })
        self.assertEqual(old["guest_age_range"], "mostly_50_plus")
        self.embedding.encode.assert_not_called()
        self.llm.chat.assert_not_called()

    def test_alias_at_threshold_updates_value_and_time_under_old_key(self):
        old = {"group_size": {"value": "medium_6_12", "time_at": "old"}}
        new = {"gathering_size": {"value": "large_13_plus", "time_at": "new"}}
        self.assertEqual(self.align(old, new), {"group_size": new["gathering_size"]})
        self.assertEqual(old["group_size"]["time_at"], "old")
        self.assertEqual(list(new), ["gathering_size"])
        self.llm.chat.assert_called_once()
        self.assertTrue(self.llm.chat.call_args.kwargs["json_mode"])

    def test_below_threshold_or_rejected_key_keeps_both(self):
        old = {"group_size": "medium_6_12"}
        new = {"gathering_size": "large_13_plus"}
        self.assertEqual(self.align(old, new, threshold=0.61), {**old, **new})
        self.llm.chat.assert_not_called()
        self.llm.chat.return_value = '{"relation": "different"}'
        self.assertEqual(self.align(old, new), {**old, **new})
        self.llm.chat.assert_called_once()

    def test_unrelated_and_empty_inputs(self):
        for old, new in [({}, {"location": "backyard"}),
                         ({"group_size": "medium_6_12"}, {}),
                         ({"group_size": "medium_6_12"}, {"location": "backyard"})]:
            with self.subTest(old=old, new=new):
                self.assertEqual(self.align(old, new), {**old, **new})
        self.llm.chat.assert_not_called()

    def test_default_uses_deepseek(self):
        with patch("models.ModelManager") as manager:
            manager.return_value.get_deepseek.return_value = self.llm
            self.assertEqual(align_attributes(
                {"group_size": "medium_6_12"}, {"gathering_size": "large_13_plus"},
                self.embedding,
            ), {"group_size": "large_13_plus"})
            manager.return_value.get_deepseek.assert_called_once()
            manager.return_value.get_qwen.assert_not_called()

    def test_invalid_response_and_failure_propagate(self):
        for response in ('not json', '{}', '{"relation": "unknown"}'):
            with self.subTest(response=response):
                self.llm.chat.return_value = response
                with self.assertRaises(ValueError):
                    self.align({"group_size": "old"}, {"gathering_size": "new"})
        self.llm.chat.side_effect = RuntimeError("unavailable")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.align({"group_size": "old"}, {"gathering_size": "new"})


class MemoryMergeTests(unittest.TestCase):
    def setUp(self):
        self.old = MemoryNode(
            memory_id="stable-id", summary="A medium bridge gathering outdoors.",
            updated_at="2026-01-01T00:00:00+00:00", topics=["bridge"],
            attributes={"group_size": {"value": "medium", "time_at": "old"},
                        "location": {"value": "backyard", "time_at": "old"}},
            embedding=[1.0, 0.0], update_count=4,
        )
        self.new = MemoryNode(
            memory_id="incoming-id", summary="Now a large bridge gathering.",
            updated_at="2026-02-01T00:00:00+00:00", topics=["bridge"],
            attributes={"group_size": {"value": "large", "time_at": "new"}},
            embedding=[0.0, 1.0], update_count=9,
        )
        self.llm = Mock()
        self.llm.chat.return_value = '{"summary": "A large bridge gathering in the backyard."}'
        self.embedding = Mock()
        self.embedding.encode.return_value = np.array([0.6, 0.8])

    def test_current_state_summary_payload_and_metadata(self):
        before = datetime.now().astimezone()
        old_snapshot, new_snapshot = self.old.to_dict(), self.new.to_dict()
        merged = merge_memory(self.old, self.new, self.embedding, self.llm)
        after = datetime.now().astimezone()
        self.assertEqual(merged.memory_id, "stable-id")
        self.assertEqual(merged.update_count, 5)
        updated = datetime.fromisoformat(merged.updated_at)
        self.assertGreaterEqual(updated, before.replace(microsecond=0))
        self.assertLessEqual(updated, after)
        self.assertEqual(merged.topics, ["bridge"])
        self.assertEqual(merged.attributes, {
            "group_size": self.new.attributes["group_size"],
            "location": self.old.attributes["location"],
        })
        self.assertEqual(merged.summary, "A large bridge gathering in the backyard.")
        self.embedding.encode.assert_called_once_with(merged.summary)
        self.assertEqual(merged.embedding, [0.6, 0.8])
        self.llm.chat.assert_called_once()
        payload = json.loads(self.llm.chat.call_args.args[0][1]["content"])
        self.assertEqual(payload, {
            "merged_topics": merged.topics, "merged_attributes": merged.attributes,
            "old_summary": self.old.summary, "new_summary": self.new.summary,
        })
        self.assertTrue(self.llm.chat.call_args.kwargs["json_mode"])
        self.assertEqual(self.old.to_dict(), old_snapshot)
        self.assertEqual(self.new.to_dict(), new_snapshot)

    def test_default_deepseek_is_reused_for_alignment_and_summary(self):
        self.new.topics = ["bridge_games"]
        self.new.attributes = {"gathering_size": {"value": "large", "time_at": "new"}}
        self.llm.chat.side_effect = [
            '{"relation": "equivalent"}', '{"relation": "equivalent"}',
            '{"summary": "A large bridge gathering in the backyard."}',
        ]
        self.embedding.encode.side_effect = [
            np.array([[1, 0], [1, 0]]),
            np.array([[1, 0], [0, 1], [1, 0]]), np.array([0.6, 0.8]),
        ]
        with patch("models.ModelManager") as manager:
            manager.return_value.get_deepseek.return_value = self.llm
            merged = merge_memory(self.old, self.new, self.embedding)
            manager.return_value.get_deepseek.assert_called_once()
            manager.return_value.get_qwen.assert_not_called()
        self.assertEqual(merged.topics, ["bridge"])
        self.assertEqual(merged.attributes["group_size"]["value"], "large")
        self.assertNotIn("gathering_size", merged.attributes)
        self.assertEqual(self.llm.chat.call_count, 3)

    def test_invalid_summary_does_not_update_inputs_or_encode(self):
        snapshot = self.old.to_dict()
        for response in ('not json', '{}', '{"summary": " "}', '{"summary": 7}'):
            with self.subTest(response=response):
                self.llm.chat.return_value = response
                with self.assertRaises(ValueError):
                    merge_memory(self.old, self.new, self.embedding, self.llm)
        self.embedding.encode.assert_not_called()
        self.assertEqual(self.old.to_dict(), snapshot)

    def test_embedding_failure_leaves_inputs_unchanged(self):
        old_snapshot, new_snapshot = self.old.to_dict(), self.new.to_dict()
        self.embedding.encode.side_effect = RuntimeError("embedding unavailable")
        with self.assertRaisesRegex(RuntimeError, "embedding unavailable"):
            merge_memory(self.old, self.new, self.embedding, self.llm)
        self.assertEqual(self.old.to_dict(), old_snapshot)
        self.assertEqual(self.new.to_dict(), new_snapshot)


if __name__ == "__main__":
    unittest.main()
