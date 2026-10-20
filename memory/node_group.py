"""记忆节点与主题记忆小组的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryNode:
    memory_id: str
    group_ids: list[str]
    topics: list[str]
    raw_text: str
    created_at: str
    embedding: list[float]
    period_index: int | None = None


@dataclass
class MemoryGroup:
    group_id: str
    head_id: str
    tail_id: str
    updated_at: str
    memory_count: int = 1
    links: dict[str, dict] = field(default_factory=dict)
