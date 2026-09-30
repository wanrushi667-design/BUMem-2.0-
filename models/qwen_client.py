"""使用 llama.cpp 加载并复用本地 Qwen GGUF 模型。"""

from pathlib import Path

from config.model_config import QwenConfig
from config.settings import PROJECT_ROOT, Settings


class QwenClient:
    def __init__(self, settings: Settings, config: QwenConfig):
        from llama_cpp import Llama

        path = Path(settings.qwen_model_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise ValueError("QWEN_MODEL_PATH 必须指向已存在的本地 GGUF 文件")

        self.config = config
        self.model = Llama(
            model_path=str(path.resolve()),
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            # 限制提示词批次，降低长上下文下 Vulkan 计算缓冲区的显存占用。
            n_batch=config.n_batch,
            n_ubatch=config.n_batch,
        )
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        # Qwen3 的 JSON 生成必须从正文开始，不能先生成思考标签。
        self.json_chat_handler = Jinja2ChatFormatter(
            template="{% set enable_thinking = false %}" + self.model.metadata["tokenizer.chat_template"],
            eos_token="<|im_end|>",
            bos_token="<|endoftext|>",
        ).to_chat_handler()

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        options = {"response_format": {"type": "json_object"}} if json_mode else {}
        chat = self.json_chat_handler if json_mode else self.model.create_chat_completion
        if json_mode:
            options["llama"] = self.model
        response = chat(
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            **options,
        )
        return response["choices"][0]["message"]["content"]
