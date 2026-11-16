from typing import Any, Literal

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    mode: Literal["incremental", "rebuild"] = Field(
        default="incremental",
        description="索引模式：incremental 为基于文件 MD5 的增量索引，rebuild 为清空并全量重建",
    )
    chunk_size: int = Field(
        default=400, ge=100, le=2000, description="文本切分块字符数"
    )
    overlap: int = Field(default=80, ge=0, le=500, description="相邻切分块重叠字符数")
    collection_name: str = Field(default="enterprise_kb", description="Chroma 集合名称")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="用户提问")
    mode: Literal["baseline", "threshold_tuned", "hybrid", "rerank", "optimized"] = (
        Field(
            default="optimized",
            description="检索问答模式：baseline / threshold_tuned / hybrid / rerank / optimized",
        )
    )
    top_k: int = Field(default=5, ge=1, le=20, description="送入生成模型的上下文片段数")
    recall_k: int = Field(default=8, ge=1, le=30, description="初始召回候选片段数")
    refusal_threshold: float | None = Field(
        default=None, description="自定义拒答距离阈值"
    )
    fallback_threshold: float | None = Field(
        default=None, description="自定义降级距离阈值"
    )
    collection_name: str = Field(default="enterprise_kb", description="Chroma 集合名称")
    use_cache: bool = Field(default=True, description="是否启用问答 LRU 缓存")


class SourceItem(BaseModel):
    source: str = Field(..., description="来源文档名称")
    chunk_id: str = Field(..., description="片段唯一ID")
    chunk_index: int = Field(default=0, description="片段在文档中的序号")
    section: str = Field(default="", description="所属章节标题")
    page: int | None = Field(default=None, description="页码")
    distance: float = Field(..., description="余弦距离")
    rrf_score: float | None = Field(default=None, description="混合检索 RRF 分数")
    rerank_score: float | None = Field(default=None, description="重排相关性得分")
    text: str = Field(default="", description="片段正文内容")


class TokenUsage(BaseModel):
    prompt: int = 0
    completion: int = 0
    total: int = 0
    rerank_total: int = 0


class AskResponse(BaseModel):
    answer: str = Field(..., description="系统生成的回答或拒答说明")
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="来源摘要列表：包含 source, chunk_id, distance",
    )
    docs: list[SourceItem] = Field(
        default_factory=list,
        description="检索器初始召回的候选片段列表",
    )
    contexts: list[SourceItem] = Field(
        default_factory=list,
        description="经过过滤或重排后送入生成模型的上下文片段列表",
    )
    path: Literal["normal", "fallback", "refusal", "error"] = Field(
        ..., description="问答执行路径标签"
    )
    mode: str = Field(default="optimized", description="本次问答使用的检索模式")
    best_distance: float | None = Field(
        default=None, description="最佳召回片段的向量距离"
    )
    latency_ms: int = Field(..., description="端到端问答耗时（毫秒）")
    cache_hit: bool = Field(default=False, description="是否命中问答缓存")
    token_usage: dict[str, int] = Field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0},
        description="生成阶段 Token 消耗统计",
    )
    extra_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="附加诊断指标",
    )
