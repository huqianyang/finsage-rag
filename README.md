# FinSage 智能金融知识库（finsage-rag）

面向普通用户的金融资料问答系统：导入基金/理财/公告/政策/FAQ 等资料后，用自然语言提问，系统检索资料并生成**有依据、可追溯、合规**的回答（流式输出、多轮追问、引用来源、风险提示）。

> 定位为金融知识查询工具，不提供投资建议、不承诺收益。

## 技术栈

FastAPI · LangGraph · Milvus 2.5（稠密向量 + 内置中文 BM25）· MongoDB · MinIO · MinerU 云解析 ·
硅基流动 DeepSeek-V4.1-Flash（问答/改写/元数据/图片摘要）· 本地 bge-m3 向量 + bge-reranker-large 重排（GPU）·
阿里云百炼 WebSearch MCP（联网搜索）· 原生 HTML/JS 前端

## 运行

### 1. 前置条件
- 中间件：虚拟机 `192.168.10.100` 上的 Milvus 2.5.5、MinIO、MongoDB 7.0（Docker）
- 本地模型：`D:/ai_models/modelscope_cache/models/BAAI/bge-m3`、`D:/ai_models/rerank/rerank/BAAI/bge-reranker-large`
- GPU 版 PyTorch wheel：`D:/ai_models/wheels/torch-2.14.1+cu126-cp313-cp313-win_amd64.whl`
- Python 3.13、uv

### 2. 安装与配置
```bash
uv sync
```
```bash
cp .env.example .env
```
在 `.env` 中填写：`SILICONFLOW_API_KEY`、`MINERU_API_TOKEN`、`DASHSCOPE_API_KEY`（可空，空则跳过联网搜索）、`MONGO_PASSWORD`、`MINIO_SECRET_KEY`。

### 3. 环境检查（9 项）
```bash
uv run python -m scripts.check_env
```

### 4. 启动服务（两个终端）
```bash
uv run uvicorn app.import_app:app --host 127.0.0.1 --port 8000
```
```bash
uv run uvicorn app.query_app:app --host 127.0.0.1 --port 8001
```
- 导入页：http://127.0.0.1:8000/
- 问答页：http://127.0.0.1:8001/
- 接口文档：http://127.0.0.1:8000/docs 、http://127.0.0.1:8001/docs

### 5. 批量导入首批资料（按目录对应资料类型）
```bash
uv run python -m scripts.batch_import "<金融/数据 目录>"
```

### 6. 测试
```bash
uv run pytest -q
```
```bash
uv run python -m scripts.eval_questions eval_report.md
```

## 目录

```text
app/
  import_app.py / query_app.py   两个服务入口
  config.py                      配置（.env）
  api/                           路由
  services/                      导入 / 问答 / 会话服务
  pipeline/importing/            导入图：parse → image → split → metadata → embed → index
  pipeline/querying/             查询图：rewrite → entity → retrieve(稠密/BM25/联网) → rrf → rerank → answer
  infra/                         Milvus / Mongo / MinIO / LLM / 本地模型 / MinerU / 联网搜索
  prompts/  compliance/  common/
web/          import.html  chat.html
scripts/      check_env.py  batch_import.py  eval_questions.py
tests/        单元测试
```
