"""?????????????????????"""
from .node_group import MemoryGroup, MemoryNode


def create_group(node, group_id, nodes, groups):
    node.group_ids = [group_id]
    groups[group_id] = MemoryGroup(
        group_id=group_id, head_id=node.memory_id, tail_id=node.memory_id,
        updated_at=node.created_at,
        links={node.memory_id: {"prev_id": None, "next_id": None}},
    )
    nodes[node.memory_id] = node


def connect_memory(node, group_ids, nodes, groups):
    node.group_ids = list(group_ids)
    for group_id in group_ids:
        group = groups[group_id]
        group.links[group.tail_id]["next_id"] = node.memory_id
        group.links[node.memory_id] = {"prev_id": group.tail_id, "next_id": None}
        group.tail_id = node.memory_id
        group.memory_count += 1
        group.updated_at = node.created_at
    nodes[node.memory_id] = node
