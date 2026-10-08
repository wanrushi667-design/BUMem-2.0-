"""记忆节点与主题记忆小组的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryNode:
    memory_id: str
    group_id: str
    topics: list[str]
    raw_text: str
    created_at: str
    embedding: list[float]
    prev_id: str | None = None
    next_id: str | None = None


@dataclass
class MemoryGroup:
    group_id: str
    head_id: str
    tail_id: str
    updated_at: str
    memory_count: int = 1
