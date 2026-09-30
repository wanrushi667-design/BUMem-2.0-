"""从本地目录加载 sentence-transformers 文本向量模型。"""

from typing import Union

from config.model_config import EmbeddingConfig
from config.settings import Settings, local_model_path


class EmbeddingClient:
    def __init__(self, settings: Settings, config: EmbeddingConfig):
        path = local_model_path(settings.embedding_model_path, "EMBEDDING_MODEL_PATH")
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.model = SentenceTransformer(
            path, device=config.device, local_files_only=True,
            trust_remote_code=False,
        )

    def encode(self, texts: Union[str, list[str]]):
        """返回 numpy 数组：单文本为一维，文本列表为二维。"""
        return self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
