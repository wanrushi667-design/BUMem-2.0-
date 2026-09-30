"""运行：python -m unittest discover -s tests -v

真实模型测试需填写 .env 并在 PowerShell 中设置 $env:RUN_MODEL_TESTS="1"。
这些测试会加载本地模型，DeepSeek 测试会发起一次真实 API 请求。
未配置的模型跳过；已配置但加载/调用失败的模型报错，不自动下载。
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from config.settings import Settings, local_model_path
from models import ModelManager


class ModelManagerTests(unittest.TestCase):
    def test_lazy_loading_and_reuse(self):
        with patch("models.model_manager.DeepSeekClient") as deepseek, \
             patch("models.model_manager.QwenClient") as qwen, \
             patch("models.model_manager.EmbeddingClient") as embedding:
            manager = ModelManager(Settings())
            for factory in (deepseek, qwen, embedding):
                factory.assert_not_called()
            self.assertIs(manager.get_deepseek(), manager.get_deepseek())
            qwen.assert_not_called()
            embedding.assert_not_called()
            self.assertIs(manager.get_qwen(), manager.get_qwen())
            self.assertIs(manager.get_embedding(), manager.get_embedding())
            for factory in (deepseek, qwen, embedding):
                factory.assert_called_once()

    def test_local_path_required(self):
        for value in ("", "not-a-local-directory/remote-model-id"):
            with self.assertRaises(ValueError):
                local_model_path(value, "QWEN_MODEL_PATH")
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(os.path.isabs(local_model_path(directory, "QWEN_MODEL_PATH")))


@unittest.skipUnless(os.getenv("RUN_MODEL_TESTS") == "1", "未启用真实模型测试")
class ModelIntegrationTests(unittest.TestCase):
    def test_deepseek(self):
        settings = Settings.from_env()
        if not all((settings.deepseek_api_key, settings.deepseek_base_url,
                    settings.deepseek_model)):
            self.skipTest("尚未配置 DeepSeek")
        client = ModelManager(settings).get_deepseek()
        result = client.chat([{"role": "user", "content": "请回复：你好"}])
        self.assertIsInstance(result, str)
        self.assertTrue(result.strip())

    def test_qwen(self):
        settings = Settings.from_env()
        if not settings.qwen_model_path:
            self.skipTest("尚未配置 QWEN_MODEL_PATH")
        client = ModelManager(settings).get_qwen()
        result = client.chat([{"role": "user", "content": "请回复：你好"}])
        self.assertIsInstance(result, str)
        self.assertTrue(result.strip())

    def test_embedding(self):
        settings = Settings.from_env()
        if not settings.embedding_model_path:
            self.skipTest("尚未配置 EMBEDDING_MODEL_PATH")
        import numpy as np

        client = ModelManager(settings).get_embedding()
        vectors = client.encode(["你好", "BUMem 模型测试"])
        self.assertEqual(vectors.ndim, 2)
        self.assertEqual(vectors.shape[0], 2)
        self.assertGreater(vectors.shape[1], 0)
        self.assertTrue(np.isfinite(vectors).all())
        vector = client.encode("你好")
        self.assertEqual(vector.shape, (vectors.shape[1],))


if __name__ == "__main__":
    unittest.main()
