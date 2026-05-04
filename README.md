# 基于知识库的智能问答系统

本项目是一个本地优先的 RAG 问答服务，提供文档入库（`/ingest`）与问答（`/ask`）接口。

## 核心能力

- 基于 LlamaIndex 的索引与检索链路。
- 支持 `md/txt/pdf` 文档入库。
- 可在本地模型（BGE + Qwen）和 OpenAI 兼容接口之间切换。
- `/ask` 返回固定结构：`{"answer": "...", "references": ["..."]}`。

## 快速开始

1. 安装依赖：
   - `pip install -r requirements.txt`
2. 配置环境变量：
   - 复制 `.env.example` 为 `.env` 并按需修改。
3. 启动服务：
   - `uvicorn app.main:app --reload`
4. 构建索引：
   - 调用 `POST /ingest`（可选 `rebuild`）。
5. 发起问答：
   - 调用 `POST /ask`，请求体示例：`{"question":"请假流程是什么？","top_k":4}`。

## 关键配置

- `VECTOR_INDEX_DIR`：LlamaIndex 持久化索引目录。
- `DOCS_DIR`：默认文档目录。
- `RETRIEVAL_CANDIDATE_K`：召回候选数。
- `RETRIEVAL_SCORE_THRESHOLD`：相似度阈值。
- `RERANKER_TYPE`：`similarity` 或 `sentence_transformer`。
- `RERANKER_MODEL`：外部 reranker 模型名（当 `RERANKER_TYPE=sentence_transformer` 时生效）。
