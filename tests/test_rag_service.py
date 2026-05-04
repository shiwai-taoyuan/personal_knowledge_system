import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import Settings
from app.rag_service import RAGService
from app.vector_store import LocalVectorStore


def fake_embed(texts: list[str]) -> list[list[float]]:
    vectors = {
        "公司请假制度：员工请假需要在OA系统提交审批。": [1.0, 0.0],
        "研发提测流程：代码合并后需要通过自动化测试。": [0.0, 1.0],
        "请假怎么做": [0.9, 0.1],
    }
    return [vectors[text] for text in texts]


def fake_chat(_: str, user_prompt: str) -> str:
    if "OA系统提交审批" in user_prompt:
        return "员工请假需要在OA系统提交审批。来源：docs/hr.md"
    return "未找到答案。"


class TestRAGService(unittest.TestCase):
    def test_rag_ask_with_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir(parents=True, exist_ok=True)
            (docs_dir / "hr.md").write_text(
                "公司请假制度：员工请假需要在OA系统提交审批。",
                encoding="utf-8",
            )

            settings = Settings(
                docs_dir=str(docs_dir),
                vector_index_dir=str(tmp_path / "index"),
                chunk_size=100,
                chunk_overlap=10,
            )
            service = RAGService(
                settings=settings,
                vector_store=LocalVectorStore(settings.vector_index_dir),
                embedding_fn=fake_embed,
                chat_fn=fake_chat,
            )
            service.ingest_directory()
            answer, refs = service.ask("请假怎么做", top_k=1)

            self.assertIn("OA系统提交审批", answer)
            self.assertEqual(len(refs), 1)
            self.assertTrue(refs[0].endswith("hr.md"))

    def test_no_context_returns_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir(parents=True, exist_ok=True)

            settings = Settings(
                docs_dir=str(docs_dir),
                vector_index_dir=str(tmp_path / "index"),
                retrieval_score_threshold=1.1,
            )

            def should_not_run_chat(_: str, __: str) -> str:
                raise AssertionError("when no context, chat should not be called")

            service = RAGService(
                settings=settings,
                vector_store=LocalVectorStore(settings.vector_index_dir),
                embedding_fn=lambda texts: [[0.1, 0.2] for _ in texts],
                chat_fn=should_not_run_chat,
            )
            answer, refs = service.ask("不存在的问题", top_k=2)
            self.assertIn("未找到足够依据", answer)
            self.assertEqual(refs, [])

    def test_chunk_text_paragraph_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                chunk_size=40,
                chunk_overlap=8,
                min_chunk_chars=10,
                vector_index_dir=str(Path(tmp) / "index"),
            )
            service = RAGService(
                settings=settings,
                vector_store=LocalVectorStore(settings.vector_index_dir),
                embedding_fn=lambda texts: [[0.1, 0.2] for _ in texts],
                chat_fn=lambda _, __: "ok",
            )
            text = (
                "第一段很短。\n\n"
                "第二段很长，需要被拆分成多个句子。它包含很多内容。"
                "并且仍然要保持语义完整。"
            )
            chunks = service.chunk_text(text)
            self.assertGreaterEqual(len(chunks), 2)
            self.assertTrue(any("第一段很短" in c for c in chunks))

    def test_ingest_pdf_file(self) -> None:
        def pdf_embed(texts: list[str]) -> list[list[float]]:
            vectors = {
                "报销制度：员工报销需提交发票原件。": [1.0, 0.0],
                "报销需要什么": [0.9, 0.1],
            }
            return [vectors[text] for text in texts]

        def pdf_chat(_: str, user_prompt: str) -> str:
            if "提交发票原件" in user_prompt:
                return "报销需要提交发票原件。"
            return "未找到答案。"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir(parents=True, exist_ok=True)
            (docs_dir / "expense.pdf").write_bytes(b"%PDF-1.4 fake")

            settings = Settings(
                docs_dir=str(docs_dir),
                vector_index_dir=str(tmp_path / "index"),
                chunk_size=100,
                chunk_overlap=10,
            )
            service = RAGService(
                settings=settings,
                vector_store=LocalVectorStore(settings.vector_index_dir),
                embedding_fn=pdf_embed,
                chat_fn=pdf_chat,
            )

            with patch.object(
                RAGService,
                "_read_pdf_text",
                return_value="报销制度：员工报销需提交发票原件。",
            ):
                files_processed, chunks_added = service.ingest_directory(rebuild=True)

            answer, refs = service.ask("报销需要什么", top_k=1)
            self.assertEqual(files_processed, 1)
            self.assertGreaterEqual(chunks_added, 1)
            self.assertIn("发票原件", answer)
            self.assertEqual(len(refs), 1)
            self.assertTrue(refs[0].endswith("expense.pdf"))

    def test_ingest_rebuild_avoids_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir(parents=True, exist_ok=True)
            (docs_dir / "a.txt").write_text("同一段内容", encoding="utf-8")

            settings = Settings(
                docs_dir=str(docs_dir),
                vector_index_dir=str(tmp_path / "index"),
                chunk_size=100,
                chunk_overlap=10,
                min_chunk_chars=1,
            )
            service = RAGService(
                settings=settings,
                vector_store=LocalVectorStore(settings.vector_index_dir),
                embedding_fn=lambda texts: [[1.0, 0.0] for _ in texts],
                chat_fn=lambda _, __: "ok",
            )

            _, first_chunks = service.ingest_directory(rebuild=True)
            _, second_chunks = service.ingest_directory(rebuild=True)
            self.assertEqual(first_chunks, second_chunks)

    def test_repeated_lines_are_compressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(vector_index_dir=str(Path(tmp) / "index"))
            service = RAGService(
                settings=settings,
                vector_store=LocalVectorStore(settings.vector_index_dir),
                embedding_fn=lambda texts: [[1.0, 0.0] for _ in texts],
                chat_fn=lambda _, __: "ok",
            )
            text = "\n".join(
                ["测试用户 [ID_PLACEHOLDER]"] * 20 + ["有效条款：员工离职需办理交接。"]
            )
            compressed = service._compress_repeated_lines(text)
            self.assertEqual(compressed.count("测试用户 [ID_PLACEHOLDER]"), 1)

    def test_near_duplicate_chunks_are_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                vector_index_dir=str(Path(tmp) / "index"),
                chunk_size=80,
                chunk_overlap=40,
                min_chunk_chars=1,
            )
            service = RAGService(
                settings=settings,
                vector_store=LocalVectorStore(settings.vector_index_dir),
                embedding_fn=lambda texts: [[1.0, 0.0] for _ in texts],
                chat_fn=lambda _, __: "ok",
            )
            text = ("测试用户 [ID_PLACEHOLDER]\n" * 40).strip()
            raw = service._fixed_window_split(text)
            chunks = service.chunk_text(text)
            self.assertLess(len(chunks), len(raw))

    def test_sensitive_info_is_masked_before_chunking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(vector_index_dir=str(Path(tmp) / "index"))
            service = RAGService(
                settings=settings,
                vector_store=LocalVectorStore(settings.vector_index_dir),
                embedding_fn=lambda texts: [[1.0, 0.0] for _ in texts],
                chat_fn=lambda _, __: "ok",
            )
            text = "员工证件 370784199801271311，联系电话 13812345678。"
            chunks = service.chunk_text(text)
            self.assertTrue(chunks)
            merged = "\n".join(chunks)
            self.assertIn("[ID_REDACTED]", merged)
            self.assertIn("[PHONE_REDACTED]", merged)

    def test_sentence_transformer_reranker_degrades_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir(parents=True, exist_ok=True)
            (docs_dir / "hr.md").write_text(
                "公司请假制度：员工请假需要在OA系统提交审批。",
                encoding="utf-8",
            )

            settings = Settings(
                docs_dir=str(docs_dir),
                vector_index_dir=str(tmp_path / "index"),
                reranker_type="sentence_transformer",
                reranker_model="BAAI/bge-reranker-base",
            )
            service = RAGService(
                settings=settings,
                embedding_fn=fake_embed,
                chat_fn=fake_chat,
            )
            service.ingest_directory(rebuild=True)
            answer, refs = service.ask("请假怎么做", top_k=1)

            self.assertIn("OA系统提交审批", answer)
            self.assertEqual(len(refs), 1)


if __name__ == "__main__":
    unittest.main()
