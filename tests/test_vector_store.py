import unittest
from pathlib import Path

from app.vector_store import LocalVectorStore


def fake_embed(texts: list[str]) -> list[list[float]]:
    mapping = {
        "python": [1.0, 0.0],
        "java": [0.0, 1.0],
        "query_python": [0.9, 0.1],
    }
    return [mapping[t] for t in texts]


class TestVectorStore(unittest.TestCase):
    def test_vector_store_search(self) -> None:
        tmp_path = Path("tests/.tmp_vector")
        if tmp_path.exists():
            for file in tmp_path.rglob("*"):
                if file.is_file():
                    file.unlink()
        tmp_path.mkdir(parents=True, exist_ok=True)

        store = LocalVectorStore(str(tmp_path / "index"))
        store.add_texts(
            texts=["python", "java"],
            sources=["a.md", "b.md"],
            embedding_fn=fake_embed,
        )

        result = store.search("query_python", top_k=1, embedding_fn=fake_embed)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source, "a.md")

    def test_vector_store_clear(self) -> None:
        tmp_path = Path("tests/.tmp_vector_clear")
        if tmp_path.exists():
            for file in tmp_path.rglob("*"):
                if file.is_file():
                    file.unlink()
        tmp_path.mkdir(parents=True, exist_ok=True)
        store = LocalVectorStore(str(tmp_path / "index"))
        store.add_texts(
            texts=["python"],
            sources=["a.md"],
            embedding_fn=fake_embed,
        )
        store.clear()
        result = store.search("query_python", top_k=1, embedding_fn=fake_embed)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
