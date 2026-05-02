# 实现说明

## 变更内容

- 新增本地 RAG 智能问答服务，支持文档入库与问答 API。
- 新增本地向量索引持久化（`vectors.npy + metadata.json`）。
- 新增可测试的服务层设计（支持注入 embedding/chat 函数）。
- 默认启用本地模型：BGE 向量化 + Qwen 问答生成。
- 切分策略升级为段落优先 + 句子回退，支持小分片合并和去重。
- 检索链路升级为候选召回、阈值过滤、词重排和来源去重。

## 接口说明

- `GET /health`：健康检查。
- `POST /ingest`：对本地文档目录建索引。
  - 请求体：`{"directory": "可选目录", "rebuild": true}`
  - 返回：`{"files_processed": int, "chunks_added": int}`
- `POST /ask`：提问并返回答案。
  - 请求体：`{"question": "问题", "top_k": 4}`
  - 返回：`{"answer": "...", "references": ["..."]}`

## 已知限制

- 文档解析支持 `md/txt/pdf`，暂不支持 Word/Excel。
- 对表格、代码块等强结构文档，仍建议后续做专用切分器。
- 词法重排目前为轻量规则，尚未引入 cross-encoder reranker。

## 后续建议

- 增加 Word/HTML 解析与结构化切分。
- 增加 reranker 与答案置信度评分。
- 增加会话记忆与多轮问答。
