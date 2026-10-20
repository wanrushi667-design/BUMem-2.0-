"""串联抽取、匹配和记忆入库；文件保存由调用方负责。"""

from __future__ import annotations

from datetime import datetime

from .connect import connect_memory, create_group
from .extractor import extract_memory
from .match import MatchDecision, MemoryMatcher
from .node_group import MemoryGroup, MemoryNode


def add_memory(raw_text: str, nodes: dict[str, MemoryNode],
               groups: dict[str, MemoryGroup], qwen_client, embedding_client,
               matcher: MemoryMatcher, *, period_index: int | None = None) -> tuple[MemoryNode, MatchDecision] | None:
    extracted = extract_memory(raw_text, qwen_client, embedding_client)
    if extracted is None:
        return None

    memory_number = max((int(key[1:]) for key in nodes), default=0) + 1
    node = MemoryNode(
        memory_id=f"m{memory_number}",
        group_ids=[],
        period_index=period_index,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        **extracted,
    )
    decision = matcher.find_connect_target(node, list(nodes.values()), groups)
    if decision.action == "CREATE":
        group_number = max((int(key[1:]) for key in groups), default=0) + 1
        create_group(node, f"g{group_number}", nodes, groups)
    else:
        connect_memory(node, [item["group_id"] for item in decision.related_groups], nodes, groups)
    return node, decision
