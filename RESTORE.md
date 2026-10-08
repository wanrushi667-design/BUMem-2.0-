# 项目备份与恢复

本仓库保存 BUMem 项目的源代码、测试、依赖声明和当前 data 目录中的数据，并保留已有 Git 提交历史。

## 在 Windows PowerShell 中恢复

```powershell
git clone https://github.com/wanrushi667-design/GROUP.git BUMem
cd BUMem
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，填写 DeepSeek API 配置，以及 `QWEN_MODEL_PATH` 和 `EMBEDDING_MODEL_PATH` 对应的本地模型目录。配置完成后运行：

```powershell
.\.venv\Scripts\python.exe main.py
```

Python 版本要求和依赖见 `requirements.txt`；本地模型架构须由所安装的模型依赖支持。

## 需要单独保管的内容

- `.env` 包含本机配置与密钥，未上传；请自行安全保存，或恢复时重新填写。
- `.venv`、Python 缓存不上传，通过安装依赖重新生成。
- 本地模型权重不随仓库上传，需单独保存或重新下载，再更新 `.env` 中的路径。
- 空目录不由 Git 保存，使用时可重新创建。

这是当前项目文件和已有版本历史的备份，不是包含密钥、模型权重及完整运行环境的系统镜像。依赖声明包含版本范围，日后安装可能得到不同的小版本。
