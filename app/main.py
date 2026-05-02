from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.rag_service import RAGService
from app.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse

app = FastAPI(title="Local RAG QA", version="0.1.0")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    settings: Settings = get_settings()
    return RAGService(settings=settings)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def homepage() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    service = get_rag_service()
    try:
        files_processed, chunks_added = service.ingest_directory(
            payload.directory,
            payload.rebuild,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unknown runtime errors
        raise HTTPException(status_code=500, detail=f"索引失败: {exc}") from exc
    return IngestResponse(files_processed=files_processed, chunks_added=chunks_added)


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    service = get_rag_service()
    try:
        answer, references = service.ask(payload.question, payload.top_k)
    except Exception as exc:  # pragma: no cover - unknown runtime errors
        raise HTTPException(status_code=500, detail=f"问答失败: {exc}") from exc
    return AskResponse(answer=answer, references=references)
