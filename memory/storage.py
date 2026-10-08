"""仅保存可读记忆；加载时重新生成向量，向量不落盘。"""

from dataclasses import asdict
import json
from pathlib import Path

from .node_group import MemoryGroup, MemoryNode


def load_memories(path, embedding_client) -> tuple[dict[str, MemoryNode], dict[str, MemoryGroup]]:
    path = Path(path)
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = list(data["nodes"])
    vectors = embedding_client.encode([data["nodes"][key]["raw_text"] for key in keys]) if keys else []
    embeddings = {key: vector.tolist() for key, vector in zip(keys, vectors)}
    nodes = {
        key: MemoryNode(**value, embedding=embeddings[key])
        for key, value in data["nodes"].items()
    }
    groups = {key: MemoryGroup(**value) for key, value in data["groups"].items()}
    return nodes, groups


def save_memories(path, nodes: dict[str, MemoryNode],
                  groups: dict[str, MemoryGroup], *, completed_periods=None) -> None:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data["nodes"] = {
        key: {field: value for field, value in asdict(node).items() if field != "embedding"}
        for key, node in nodes.items()
    }
    data["groups"] = {key: asdict(group) for key, group in groups.items()}
    if completed_periods is not None:
        data["completed_periods"] = completed_periods
    path.parent.mkdir(parents=True, exist_ok=True)
    # 通过临时文件替换，避免写入中断破坏已有记忆。
    for destination, content in (
        (path, data),
    ):
        pending = destination.with_suffix(".tmp")
        pending.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(destination)
    path.with_suffix(".embeddings.json").unlink(missing_ok=True)
