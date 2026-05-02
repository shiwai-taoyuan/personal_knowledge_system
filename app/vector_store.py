from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

EmbeddingFn = Callable[[list[str]], list[list[float]]]


@dataclass
class RetrievalResult:
    text: str
    source: str
    score: float


class LocalVectorStore:
    def __init__(self, index_dir: str) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_path = self.index_dir / "vectors.npy"
        self.meta_path = self.index_dir / "metadata.json"
        self.vectors: Optional[np.ndarray] = None
        self.metadata: list[dict[str, str]] = []
        self._load()

    def clear(self) -> None:
        self.vectors = None
        self.metadata = []
        if self.vectors_path.exists():
            self.vectors_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()

    def _load(self) -> None:
        if self.vectors_path.exists():
            self.vectors = np.load(self.vectors_path)
        if self.meta_path.exists():
            self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        if self.vectors is not None:
            np.save(self.vectors_path, self.vectors)
        self.meta_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_texts(
        self,
        texts: list[str],
        sources: list[str],
        embedding_fn: EmbeddingFn,
    ) -> int:
        if not texts:
            return 0
        embeddings = np.array(embedding_fn(texts), dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        normalized = embeddings / norms

        if self.vectors is None:
            self.vectors = normalized
        else:
            self.vectors = np.concatenate([self.vectors, normalized], axis=0)

        for text, source in zip(texts, sources):
            self.metadata.append({"text": text, "source": source})

        self._save()
        return len(texts)

    def search(
        self,
        query: str,
        top_k: int,
        embedding_fn: EmbeddingFn,
        candidate_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[RetrievalResult]:
        if self.vectors is None or len(self.metadata) == 0:
            return []

        query_vec = np.array(embedding_fn([query])[0], dtype=np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        scores = self.vectors @ query_vec
        size = candidate_k if candidate_k is not None else top_k
        top_indices = np.argsort(scores)[::-1][:size]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            score = float(scores[idx])
            if min_score is not None and score < min_score:
                continue
            meta = self.metadata[int(idx)]
            results.append(
                RetrievalResult(
                    text=meta["text"],
                    source=meta["source"],
                    score=score,
                )
            )
        return results
