from app.config import get_settings
from app.rag_service import RAGService


def main() -> None:
    settings = get_settings()
    service = RAGService(settings=settings)
    files_processed, chunks_added = service.ingest_directory()
    print(f"processed={files_processed}, chunks={chunks_added}")


if __name__ == "__main__":
    main()
