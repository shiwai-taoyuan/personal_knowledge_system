# Local RAG QA System

使用大模型 + RAG + 本地向量知识库数据实现的智能问答系统。
默认配置为本地模型：`BGE` 负责向量化，`Qwen` 负责问答生成。

## 功能

- 本地文档入库：读取 `data/docs` 下 `md/txt/pdf` 文件。
- 文档切分优化：段落优先、句子切分回退、小分片合并、去重入库。
- 向量索引持久化：向量与元数据保存在 `data/index`。
- 检索增强问答：候选召回 + 阈值过滤 + 词重排 + 来源去重，再交给大模型回答。
- HTTP API：提供 `/ingest`、`/ask`、`/health` 接口。
- Web 演示页：访问根路径 `/` 即可交互问答。

## 目录结构

- `app/main.py`：FastAPI 服务入口
- `app/rag_service.py`：RAG 核心流程（切分、检索、生成）
- `app/vector_store.py`：本地向量索引存储与检索
- `scripts/ingest.py`：命令行入库脚本
- `tests/`：单元测试

## 环境准备

1. 复制环境变量模板：

   ```bash
   cp .env.example .env
   ```

2. 配置 `.env` 里的模型参数（默认使用本地 BGE + Qwen）。
3. 安装依赖：

   ```bash
   python3 -m pip install -r requirements.txt
   ```

### 本地模型默认路径

- `BGE_MODEL_PATH=/Users/wei/Documents/code/checkpoints/bge-small-zh-v1d5`
- `QWEN_MODEL_PATH=/Users/wei/Documents/code/checkpoints/Qwen3-0d6B`
- `EMBEDDING_PROVIDER=local_bge`
- `CHAT_PROVIDER=local_qwen`

## 运行

### 启动服务

```bash
uvicorn app.main:app --reload
```

### 打开网页演示

浏览器访问：`http://127.0.0.1:8000/`

页面内可直接：
- 点击“构建/刷新索引”执行入库
- 输入问题并点击“提问”查看答案与引用来源

### 文档入库

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"rebuild": true}'
```

`rebuild=true` 表示清空旧索引后重建，避免重复入库。

### 问答

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"这个系统怎么工作？","top_k":3}'
```

## 测试

项目测试采用 `unittest`，执行：

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
