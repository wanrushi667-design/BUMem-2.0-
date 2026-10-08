"""创建主题小组，或将记忆节点追加到已有小组的尾部。"""

from .node_group import MemoryGroup, MemoryNode


def create_group(node: MemoryNode, group_id: str,
                 nodes: dict[str, MemoryNode], groups: dict[str, MemoryGroup]) -> None:
    node.group_id = group_id
    node.prev_id = None
    node.next_id = None
    groups[group_id] = MemoryGroup(
        group_id=group_id,
        head_id=node.memory_id,
        tail_id=node.memory_id,
        updated_at=node.created_at,
    )
    nodes[node.memory_id] = node


def connect_memory(node: MemoryNode, target_memory_id: str,
                   nodes: dict[str, MemoryNode], groups: dict[str, MemoryGroup]) -> None:
    group = groups[nodes[target_memory_id].group_id]
    tail = nodes[group.tail_id]
    node.group_id = group.group_id
    node.prev_id = tail.memory_id
    node.next_id = None
    tail.next_id = node.memory_id
    group.tail_id = node.memory_id
    group.memory_count += 1
    group.updated_at = node.created_at
    nodes[node.memory_id] = node
