"""Choose the longest unfiltered memories within the oldest remaining period."""


def select_candidates(nodes, filtered_ids, raw_text_tokens, batch_size=10):
    available = [node for key, node in nodes.items() if key not in filtered_ids]
    if not available:
        return []
    oldest = min(node.period_index for node in available)
    candidates = [node for node in available if node.period_index == oldest]
    return sorted(candidates, key=lambda node: (
        -raw_text_tokens[node.memory_id], node.memory_id,
    ))[:batch_size]


def select_second_candidates(nodes, second_filtered_ids, raw_text_tokens, batch_size=10):
    """Rank current raw_text lengths globally, without a period preference."""
    candidates = [node for key, node in nodes.items() if key not in second_filtered_ids]
    return sorted(candidates, key=lambda node: (
        -raw_text_tokens[node.memory_id], node.memory_id,
    ))[:batch_size]
