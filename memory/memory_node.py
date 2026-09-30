"""主题级记忆；时间由程序维护。"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class MemoryNode:
    memory_id: str
    summary: str
    updated_at: str
    topics: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    update_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
