"""离线验证检索排序和模型输入，不加载模型或访问 API。"""

import copy
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

from memory.memory_node import MemoryNode
from retrieval import process_query, retrieve_memories, select_answer


def memory(index, topics=None):
    return MemoryNode(
        memory_id=str(index), summary=f"Memory {index}", updated_at="now",
        topics=topics if topics is not None else [f"topic_{index}"],
        attributes={"detail": {"value": index, "time_at": "now"}},
        embedding=[10.0, float(index)], update_count=2,
    )


class RetrievalTests(unittest.TestCase):
    def test_query_encodes_before_qwen_and_preserves_question(self):
        calls = []
        embedding = Mock()
        embedding.encode.side_effect = lambda text: (
            calls.append("embedding") or np.array([2.0, 1.0])
        )
        qwen = Mock()
        qwen.chat.side_effect = lambda *args, **kwargs: (
            calls.append("qwen") or '{"topics": ["card_games"]}'
        )
        result = process_query("Which game?", embedding, qwen)
        self.assertEqual(calls, ["embedding", "qwen"])
        self.assertEqual(result, {
            "text": "Which game?", "topics": ["card_games"], "embedding": [2.0, 1.0],
        })
        self.assertTrue(qwen.chat.call_args.kwargs["json_mode"])

    def test_top10_then_max_over_all_topic_pairs_then_top5(self):
        memories = [memory(i) for i in reversed(range(11))]
        memories[1].topics.append("opposite")  # Memory 9: max must win over mean.
        before = copy.deepcopy([m.to_dict() for m in memories])
        query = {"text": "Question", "topics": ["q_a", "q_b"], "embedding": [3, 0]}
        vectors = {"q a": [1, 0, 0], "q b": [0, 1, 0], "opposite": [0, -1, 0]}
        vectors.update({f"topic {i}": [0, i + 1, 10] for i in range(11)})
        embedding = Mock()
        embedding.encode.side_effect = lambda texts: np.array([vectors[t] for t in texts])
        hits = retrieve_memories(query, memories, embedding)
        self.assertEqual([h.memory.memory_id for h in hits], ["9", "8", "7", "6", "5"])
        self.assertAlmostEqual(hits[0].topic_score, 2 ** -0.5)
        self.assertAlmostEqual(hits[0].summary_score, 10 / np.sqrt(181))
        self.assertNotIn("topic 10", embedding.encode.call_args.args[0])
        self.assertEqual([m.to_dict() for m in memories], before)

    def test_empty_topics_and_ties_keep_summary_order(self):
        query = {"topics": [], "embedding": [1, 0]}
        embedding = Mock()
        hits = retrieve_memories(query, [memory(2), memory(0)], embedding)
        self.assertEqual([h.memory.memory_id for h in hits], ["0", "2"])
        embedding.encode.assert_not_called()
        query["topics"] = ["query"]
        embedding.encode.side_effect = lambda texts: np.array([[1, 0] for _ in texts])
        hits = retrieve_memories(query, [memory(2), memory(0)], embedding)
        self.assertEqual([h.memory.memory_id for h in hits], ["0", "2"])
        self.assertEqual(retrieve_memories(query, [], embedding), [])

    def test_missing_topics_rank_below_negative_similarity(self):
        embedding = Mock()
        embedding.encode.return_value = np.array([[1, 0], [-1, 0]])
        hits = retrieve_memories(
            {"topics": ["query"], "embedding": [1, 0]},
            [memory(0, []), memory(1, ["negative"])], embedding,
        )
        self.assertEqual([h.memory.memory_id for h in hits], ["1", "0"])
        self.assertEqual(hits[0].topic_score, -1)
        self.assertIsNone(hits[1].topic_score)

    def test_reply_preserves_fields_without_vectors_and_returns_choice_id(self):
        query = {"text": "Which?", "topics": ["topic"], "embedding": [1, 0]}
        memories = [memory(i) for i in range(5)]
        original = copy.deepcopy([query, [m.to_dict() for m in memories]])
        choices = [{"id": 7, "text": "First"}, {"id": 9, "text": "Second"}]
        deepseek = Mock()
        deepseek.chat.return_value = "9\n"
        with patch("retrieval.reply.REPLY_SYSTEM_PROMPT", "Test prompt"):
            self.assertEqual(select_answer(query, choices, memories, deepseek), 9)
            payload = json.loads(deepseek.chat.call_args.args[0][1]["content"])
            self.assertEqual(payload["query"], {"text": "Which?", "topics": ["topic"]})
            self.assertEqual(payload["memories"], [
                {k: v for k, v in m.to_dict().items() if k != "embedding"} for m in memories
            ])
            self.assertEqual(payload["answer_choices"], choices)
            deepseek.chat.return_value = "1"
            with self.assertRaises(ValueError):
                select_answer(query, choices, memories, deepseek)
        self.assertEqual([query, [m.to_dict() for m in memories]], original)

    def test_empty_reply_prompt_stops_before_api_call(self):
        deepseek = Mock()
        with self.assertRaisesRegex(ValueError, "REPLY_SYSTEM_PROMPT"):
            select_answer({}, [], [], deepseek)
        deepseek.chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
