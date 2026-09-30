"""按 query 文本向量召回 Top 10，再按最大 topic cosine 重排取 Top 5。"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from memory.memory_node import MemoryNode


@dataclass
class RetrievalHit:
    memory: MemoryNode
    summary_score: float
    topic_score: Optional[float] = None


def retrieve_memories(
    query: dict,
    memories: list[MemoryNode],
    embedding_client,
    *,
    candidate_k: int = 10,
    top_k: int = 5,
) -> list[RetrievalHit]:
    """接收 process_query 的结果；只检索，不修改输入或调用 DeepSeek。"""
    if candidate_k < 1 or top_k < 1:
        raise ValueError("candidate_k 和 top_k 必须大于 0")
    if not memories:
        return []
    candidates = _recall_candidates(query["embedding"], memories, candidate_k)
    return _rerank_by_topics(query["topics"], candidates, embedding_client)[:top_k]


def _recall_candidates(query_embedding, memories, candidate_k):
    if any(memory.embedding is None for memory in memories):
        raise ValueError("请先为记忆生成 summary embedding")
    query_vector = np.asarray(query_embedding, dtype=float)
    vectors = np.asarray([memory.embedding for memory in memories], dtype=float)
    query_norm = np.linalg.norm(query_vector)
    norms = np.linalg.norm(vectors, axis=1)
    if query_norm == 0 or np.any(norms == 0):
        raise ValueError("query 和 summary embedding 不能是零向量")
    scores = vectors @ query_vector / (norms * query_norm)
    indices = np.argsort(-scores, kind="stable")[:candidate_k]
    return [RetrievalHit(memories[i], float(scores[i])) for i in indices]


def _rerank_by_topics(query_topics, candidates, embedding_client):
    # 没有 query topics 时沿用召回顺序；无 topic 的记忆排在有分数的记忆之后。
    if not query_topics or not any(hit.memory.topics for hit in candidates):
        return candidates
    topics = list(dict.fromkeys(
        query_topics + [topic for hit in candidates for topic in hit.memory.topics]
    ))
    vectors = np.asarray(embedding_client.encode(
        [topic.replace("_", " ") for topic in topics]
    ), dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("topic embedding 不能是零向量")
    vectors = vectors / norms
    topic_vectors = dict(zip(topics, vectors))
    query_vectors = np.asarray([topic_vectors[topic] for topic in query_topics])
    for hit in candidates:
        if hit.memory.topics:
            memory_vectors = np.asarray([
                topic_vectors[topic] for topic in hit.memory.topics
            ])
            scores = np.clip(query_vectors @ memory_vectors.T, -1.0, 1.0)
            hit.topic_score = float(scores.max())
    # Python 排序稳定，同分时保留 summary 召回顺序。
    return sorted(candidates, key=lambda hit: (
        hit.topic_score if hit.topic_score is not None else float("-inf")
    ), reverse=True)
