from __future__ import annotations

import re
import inspect
from copy import deepcopy
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Optional

from llama_index.core import (
    Document,
    QueryBundle,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.postprocessor import SimilarityPostprocessor
from pydantic import PrivateAttr

from app.config import Settings
from app.vector_store import EmbeddingFn, RetrievalResult

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


class CallableEmbedding(BaseEmbedding):
    model_name: str = "callable-embedding"
    _embed_fn: EmbeddingFn = PrivateAttr()

    def __init__(self, embed_fn: EmbeddingFn, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._embed_fn = embed_fn

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed_fn([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed_fn([text])[0]


class RAGService:
    def __init__(
        self,
        settings: Settings,
        vector_store: Optional[Any] = None,
        embedding_fn: Optional[EmbeddingFn] = None,
        chat_fn: Optional[Callable[[str, str], str]] = None,
    ) -> None:
        self.settings = settings
        # 兼容旧测试参数，LlamaIndex 迁移后不再使用自定义 vector_store。
        self.vector_store = vector_store
        self._embedding_fn = embedding_fn
        self._chat_fn = chat_fn
        self._client = None
        self._bge_model: Any = None
        self._qwen_generator: Any = None
        self._storage_dir = Path(self.settings.vector_index_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        needs_openai = (
            (self._embedding_fn is None and settings.embedding_provider == "openai")
            or (self._chat_fn is None and settings.chat_provider == "openai")
        )
        if needs_openai:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )

    def _resolve_model_path(self, explicit_path: str, keyword: str) -> Path:
        path = Path(explicit_path)
        if path.exists():
            return path

        root = Path(self.settings.checkpoints_dir)
        if root.exists():
            candidates = sorted(
                p for p in root.iterdir() if p.is_dir() and keyword.lower() in p.name.lower()
            )
            if candidates:
                return candidates[0]
        raise FileNotFoundError(f"未找到本地模型目录: {explicit_path}（keyword={keyword}）")

    def _get_bge_model(self) -> Any:
        if self._bge_model is None:
            from sentence_transformers import SentenceTransformer

            model_path = self._resolve_model_path(self.settings.bge_model_path, "bge")
            self._bge_model = SentenceTransformer(str(model_path))
        return self._bge_model

    def _get_qwen_generator(self) -> Any:
        if self._qwen_generator is None:
            from transformers import pipeline

            model_path = self._resolve_model_path(self.settings.qwen_model_path, "qwen")
            self._qwen_generator = pipeline(
                task="text-generation",
                model=str(model_path),
                tokenizer=str(model_path),
                trust_remote_code=True,
            )
        return self._qwen_generator

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedding_fn is not None:
            return self._embedding_fn(texts)
        if self.settings.embedding_provider == "local_bge":
            model = self._get_bge_model()
            embeddings = model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        if self._client is None:
            raise RuntimeError("OpenAI client not initialized")
        response = self._client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def _get_llama_embedding(self) -> BaseEmbedding:
        return CallableEmbedding(self._embed)

    def _load_index(self) -> Optional[VectorStoreIndex]:
        if not any(self._storage_dir.iterdir()):
            return None
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self._storage_dir)
            )
            return load_index_from_storage(
                storage_context=storage_context,
                embed_model=self._get_llama_embedding(),
            )
        except Exception:
            return None

    def _get_postprocessors(self, top_k: int) -> list[Any]:
        processors: list[Any] = [
            SimilarityPostprocessor(similarity_cutoff=self.settings.retrieval_score_threshold)
        ]
        reranker_type = self.settings.reranker_type.lower().strip()
        if reranker_type == "sentence_transformer":
            try:
                from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

                processors.append(
                    SentenceTransformerRerank(
                        top_n=top_k,
                        model=self.settings.reranker_model,
                    )
                )
            except Exception:
                # 降级到仅使用相似度后处理，保证链路可用。
                pass
        return processors

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        if self._chat_fn is not None:
            return self._chat_fn(system_prompt, user_prompt)
        if self.settings.chat_provider == "local_qwen":
            generator = self._get_qwen_generator()
            user_content = user_prompt
            if self.settings.qwen_disable_thinking:
                # Qwen 系列支持 /no_think 指令，减少推理链显式输出。
                user_content = "/no_think\n" + user_prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            tokenizer = getattr(generator, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                try:
                    signature = inspect.signature(tokenizer.apply_chat_template)
                    if (
                        self.settings.qwen_disable_thinking
                        and "enable_thinking" in signature.parameters
                    ):
                        kwargs["enable_thinking"] = False
                except (TypeError, ValueError):
                    pass
                prompt = tokenizer.apply_chat_template(messages, **kwargs)
            else:
                prompt = (
                    f"<|system|>\n{system_prompt}\n"
                    f"<|user|>\n{user_prompt}\n"
                    "<|assistant|>\n"
                )
            model_generation_config = getattr(generator.model, "generation_config", None)
            if model_generation_config is not None:
                generation_config = deepcopy(model_generation_config)
                generation_config.max_new_tokens = 512
                generation_config.do_sample = True
                generation_config.temperature = self.settings.temperature
                if hasattr(generation_config, "max_length"):
                    generation_config.max_length = None
                outputs = generator(
                    prompt,
                    generation_config=generation_config,
                    return_full_text=False,
                    clean_up_tokenization_spaces=False,
                )
            else:
                outputs = generator(
                    prompt,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=self.settings.temperature,
                    return_full_text=False,
                    clean_up_tokenization_spaces=False,
                )
            text = outputs[0].get("generated_text", "").strip()
            return self._clean_qwen_output(text)
        if self._client is None:
            raise RuntimeError("OpenAI client not initialized")
        response = self._client.chat.completions.create(
            model=self.settings.chat_model,
            temperature=self.settings.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _clean_qwen_output(text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^\s*思考过程[:：].*$", "", cleaned, flags=re.MULTILINE).strip()
        return cleaned or text.strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _mask_sensitive_text(text: str) -> str:
        # 脱敏身份证号（15/18位，末位可X），避免私有信息入库。
        text = re.sub(r"\b\d{15}(?:\d{2}[0-9Xx])?\b", "[ID_REDACTED]", text)
        # 脱敏中国大陆手机号。
        text = re.sub(r"\b1[3-9]\d{9}\b", "[PHONE_REDACTED]", text)
        return text

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        pieces = re.split(r"(?<=[。！？.!?；;])", text)
        return [p.strip() for p in pieces if p.strip()]

    def _split_long_text(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return self._fixed_window_split(text)

        chunks: list[str] = []
        buf = ""
        for sentence in sentences:
            candidate = f"{buf}{sentence}"
            if len(candidate) <= self.settings.chunk_size:
                buf = candidate
                continue
            if buf:
                chunks.append(buf.strip())
            if len(sentence) <= self.settings.chunk_size:
                buf = sentence
            else:
                chunks.extend(self._fixed_window_split(sentence))
                buf = ""
        if buf:
            chunks.append(buf.strip())
        return chunks

    def chunk_text(self, text: str) -> list[str]:
        text = self._normalize_text(text)
        text = self._mask_sensitive_text(text)
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        raw_chunks: list[str] = []
        for para in paragraphs:
            if len(para) <= self.settings.chunk_size:
                raw_chunks.append(para)
            else:
                raw_chunks.extend(self._split_long_text(para))

        merged: list[str] = []
        for chunk in raw_chunks:
            if not merged:
                merged.append(chunk)
                continue
            if len(chunk) < self.settings.min_chunk_chars:
                merged[-1] = f"{merged[-1]}\n{chunk}".strip()
            else:
                merged.append(chunk)

        # 去重，避免重复片段污染索引。
        deduped: list[str] = []
        seen: set[str] = set()
        recent_norms: list[str] = []
        for chunk in merged:
            normalized = re.sub(r"\s+", " ", chunk).strip()
            if normalized and normalized not in seen:
                if any(self._is_near_duplicate(normalized, prev) for prev in recent_norms):
                    continue
                seen.add(normalized)
                deduped.append(chunk)
                recent_norms.append(normalized)
                if len(recent_norms) > 6:
                    recent_norms.pop(0)
        return deduped

    @staticmethod
    def _is_near_duplicate(a: str, b: str) -> bool:
        return SequenceMatcher(None, a, b).ratio() >= 0.92

    def _fixed_window_split(self, text: str) -> list[str]:
        size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap
        if overlap >= size:
            overlap = max(0, size // 5)

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start = end - overlap
        return [c for c in chunks if c]

    def _read_pdf_text(self, file: Path) -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(file))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        merged = "\n".join(pages)
        return self._compress_repeated_lines(merged)

    @staticmethod
    def _compress_repeated_lines(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""

        # 压缩连续重复行。
        compressed: list[str] = []
        last = None
        for line in lines:
            if line == last:
                continue
            compressed.append(line)
            last = line

        # 过滤高频短噪声行（页眉页脚/签名扫描重复）。
        counts = Counter(compressed)
        filtered: list[str] = []
        seen_high_freq: set[str] = set()
        for line in compressed:
            digit_ratio = len(re.findall(r"\d", line)) / max(1, len(line))
            noisy = counts[line] >= 6 and len(line) <= 64 and digit_ratio >= 0.2
            if noisy:
                if line in seen_high_freq:
                    continue
                seen_high_freq.add(line)
            filtered.append(line)
        return "\n".join(filtered)

    def _read_file_content(self, file: Path) -> str:
        suffix = file.suffix.lower()
        if suffix in {".md", ".txt"}:
            return file.read_text(encoding="utf-8")
        if suffix == ".pdf":
            return self._read_pdf_text(file)
        return ""

    def _clear_storage_dir(self) -> None:
        if not self._storage_dir.exists():
            return
        for path in sorted(self._storage_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()

    def ingest_directory(
        self,
        directory: Optional[str] = None,
        rebuild: Optional[bool] = None,
    ) -> tuple[int, int]:
        target = Path(directory or self.settings.docs_dir)
        if not target.exists():
            raise FileNotFoundError(f"文档目录不存在: {target}")
        do_rebuild = self.settings.ingest_rebuild_default if rebuild is None else rebuild
        if do_rebuild:
            self._clear_storage_dir()
            self._storage_dir.mkdir(parents=True, exist_ok=True)

        documents: list[Document] = []
        files_processed = 0

        for file in sorted(target.rglob("*")):
            if not file.is_file() or file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            content = self._read_file_content(file)
            chunks = self.chunk_text(content)
            if not chunks:
                continue
            for chunk in chunks:
                documents.append(
                    Document(
                        text=chunk,
                        metadata={"source": str(file)},
                    )
                )
            files_processed += 1

        if not documents:
            return files_processed, 0

        index = self._load_index() if not do_rebuild else None
        if index is None:
            index = VectorStoreIndex.from_documents(
                documents=documents,
                embed_model=self._get_llama_embedding(),
            )
        else:
            for doc in documents:
                index.insert(doc)
        index.storage_context.persist(persist_dir=str(self._storage_dir))
        return files_processed, len(documents)

    def ask(self, question: str, top_k: Optional[int] = None) -> tuple[str, list[str]]:
        k = top_k or self.settings.default_top_k
        index = self._load_index()
        # if index is None:
        #     return "知识库中未找到足够依据，暂时无法确定答案。", []

        retriever = index.as_retriever(
            similarity_top_k=max(k, self.settings.retrieval_candidate_k)
        )
        node_results = retriever.retrieve(question)
        query_bundle = QueryBundle(question)
        for processor in self._get_postprocessors(k):
            node_results = self._run_postprocessor(
                processor=processor,
                node_results=node_results,
                query_bundle=query_bundle,
            )
        retrieved = self._dedupe_and_limit(node_results, k)
        # if not retrieved:
        #     return "知识库中未找到足够依据，暂时无法确定答案。", []

        context = self._build_context(retrieved, self.settings.max_context_chars)
        system_prompt = (
            "你是个人知识库问答助手小帅"
        )
        user_prompt = (
            f"问题：{question}\n\n"
            f"上下文：\n{context}\n\n"
            "请基于上下文给出最终答案。"
        )
        answer = self._chat(system_prompt, user_prompt).strip()
        references = self._extract_references(retrieved)
        return answer, references

    @staticmethod
    def _run_postprocessor(
        processor: Any,
        node_results: list[Any],
        query_bundle: QueryBundle,
    ) -> list[Any]:
        # LlamaIndex 在不同版本中参数名存在差异，优先使用位置参数保证兼容。
        try:
            return processor.postprocess_nodes(node_results, query_bundle=query_bundle)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument" not in message:
                raise
        try:
            return processor.postprocess_nodes(nodes=node_results, query_bundle=query_bundle)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument" not in message:
                raise
        return processor.postprocess_nodes(node_results)

    @staticmethod
    def _build_context(retrieved: list[RetrievalResult], max_chars: int) -> str:
        if not retrieved:
            return "（无可用上下文）"
        rows: list[str] = []
        used_chars = 0
        for idx, item in enumerate(retrieved, start=1):
            block = (
                f"[{idx}] source={item.source}; score={item.score:.4f}\n{item.text}"
            )
            if used_chars + len(block) > max_chars:
                break
            rows.append(block)
            used_chars += len(block)
        return "\n\n".join(rows)

    def _dedupe_and_limit(self, node_results: list[Any], top_k: int) -> list[RetrievalResult]:
        selected: list[RetrievalResult] = []
        source_count: dict[str, int] = {}
        seen_signature: set[str] = set()

        for node in node_results:
            text = node.node.get_content().strip()
            if not text:
                continue
            score = float(node.score or 0.0)
            if score < self.settings.retrieval_score_threshold:
                continue
            source = str(node.node.metadata.get("source", "unknown"))
            signature = re.sub(r"\s+", " ", text)[:180]
            if signature in seen_signature:
                continue
            count = source_count.get(source, 0)
            if count >= self.settings.max_per_source:
                continue
            selected.append(RetrievalResult(text=text, source=source, score=score))
            seen_signature.add(signature)
            source_count[source] = count + 1
            if len(selected) >= top_k:
                break

        return selected

    @staticmethod
    def _extract_references(retrieved: list[RetrievalResult]) -> list[str]:
        references: list[str] = []
        for item in retrieved:
            if item.source not in references:
                references.append(item.source)
        return references
