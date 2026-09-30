"""同一管理器内按需创建并复用三个客户端。"""

from config.model_config import DeepSeekConfig, EmbeddingConfig, QwenConfig
from config.settings import Settings
from .deepseek_client import DeepSeekClient
from .embedding_client import EmbeddingClient
from .qwen_client import QwenClient


class ModelManager:
    def __init__(self, settings=None, deepseek_config=None, qwen_config=None,
                 embedding_config=None):
        self.settings = settings if settings is not None else Settings.from_env()
        self.deepseek_config = deepseek_config or DeepSeekConfig()
        self.qwen_config = qwen_config or QwenConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self._deepseek = None
        self._qwen = None
        self._embedding = None

    def get_deepseek(self) -> DeepSeekClient:
        if self._deepseek is None:
            self._deepseek = DeepSeekClient(self.settings, self.deepseek_config)
        return self._deepseek

    def get_qwen(self) -> QwenClient:
        if self._qwen is None:
            self._qwen = QwenClient(self.settings, self.qwen_config)
        return self._qwen

    def get_embedding(self) -> EmbeddingClient:
        if self._embedding is None:
            self._embedding = EmbeddingClient(self.settings, self.embedding_config)
        return self._embedding
