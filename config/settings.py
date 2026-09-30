"""读取项目根目录的 .env；已有环境变量优先。"""

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str = field(default="", repr=False)
    deepseek_base_url: str = ""
    deepseek_model: str = ""
    qwen_model_path: str = ""
    embedding_model_path: str = ""

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
        return cls(**{
            name: os.getenv(name.upper(), "").strip()
            for name in cls.__dataclass_fields__
        })


def local_model_path(value: str, variable: str) -> str:
    """相对模型路径以项目根目录为基准，不接受远程模型 ID。"""
    if not value:
        raise ValueError(f"请在 .env 中填写 {variable}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_dir():
        raise ValueError(f"{variable} 必须指向已存在的本地模型目录")
    return str(path.resolve())
