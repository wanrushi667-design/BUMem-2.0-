"""DeepSeek 的 OpenAI 兼容接口。"""

from config.model_config import DeepSeekConfig
from config.settings import Settings


class DeepSeekClient:
    def __init__(self, settings: Settings, config: DeepSeekConfig):
        if not all((settings.deepseek_api_key, settings.deepseek_base_url,
                    settings.deepseek_model)):
            raise ValueError("请填写 DEEPSEEK_API_KEY、DEEPSEEK_BASE_URL 和 DEEPSEEK_MODEL")
        from openai import OpenAI
        import httpx

        self.config = config
        self.model = settings.deepseek_model
        self.client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=config.timeout,
            # 当前机器的系统 HTTPS 代理握手失败；DeepSeek 使用直连。
            http_client=httpx.Client(trust_env=False, timeout=config.timeout),
        )

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        options = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **options,
        )
        return response.choices[0].message.content or ""
