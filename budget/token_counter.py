"""Count raw_text and topics once per memory, independently of group membership."""

import json
from pathlib import Path


class TokenCounter:
    def __init__(self, tokenize):
        """Accept a callable taking str and returning token IDs without special tokens."""
        self.tokenize = tokenize

    @classmethod
    def from_qwen(cls, qwen_client):
        """Reuse the loaded GGUF tokenizer without adding BOS or parsing special tokens."""
        return cls(lambda text: qwen_client.model.tokenize(
            text.encode("utf-8"), add_bos=False, special=False))

    def count_text(self, text: str) -> int:
        return len(self.tokenize(text)) if text else 0

    def count_memories(self, nodes) -> dict:
        """Accept the nodes dictionary stored in memory.json.

        Count each topic string separately, excluding JSON punctuation. Topics
        stored in different memories count separately even if their text matches.
        """
        raw_text = {key: self.count_text(node["raw_text"]) for key, node in nodes.items()}
        topics = {
            key: sum(self.count_text(topic) for topic in node["topics"])
            for key, node in nodes.items()
        }
        return {
            "raw_text_by_memory": raw_text,
            "topics_by_memory": topics,
            "raw_text_tokens": sum(raw_text.values()),
            "topics_tokens": sum(topics.values()),
            "total_tokens": sum(raw_text.values()) + sum(topics.values()),
        }

    def count_file(self, memory_file) -> dict:
        data = json.loads(Path(memory_file).read_text(encoding="utf-8"))
        return self.count_memories(data["nodes"])
