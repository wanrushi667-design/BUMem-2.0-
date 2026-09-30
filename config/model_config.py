"""各模型的推理参数。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekConfig:
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: float = 60.0


@dataclass(frozen=True)
class QwenConfig:
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    n_batch: int = 128
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9


@dataclass(frozen=True)
class EmbeddingConfig:
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True
