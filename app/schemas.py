from pydantic import BaseModel, Field
from typing import Optional


class IngestRequest(BaseModel):
    directory: Optional[str] = Field(default=None, description="文档目录，默认读取配置中的 docs_dir")
    rebuild: Optional[bool] = Field(
        default=None,
        description="是否先清空旧索引再重建，默认使用配置项",
    )


class IngestResponse(BaseModel):
    files_processed: int
    chunks_added: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    references: list[str]
