import os
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

# 运行时自动加载项目根目录下的 .env 环境变量
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Settings:
    """全局配置中心：统一从环境变量与默认值加载系统及模型配置"""

    PROJECT_NAME: str = "企业知识库问答系统 (RAG)"
    VERSION: str = "1.0.0"

    # 目录配置
    ROOT_DIR: Path = ROOT_DIR
    MODELS_DIR: Path = ROOT_DIR / "models"
    DATA_DIR: Path = ROOT_DIR / "data"
    DOCS_DIR: Path = DATA_DIR / "docs"
    CHROMA_DIR: Path = DATA_DIR / "chroma_db"
    INDEX_STATE_FILE: Path = DATA_DIR / "index_state.json"
    EVAL_DATASET_FILE: Path = DATA_DIR / "eval_dataset.json"
    OUTPUTS_DIR: Path = ROOT_DIR / "outputs"

    # 上传文件限制
    ALLOWED_EXTENSIONS: ClassVar[set[str]] = {".pdf", ".docx", ".txt", ".md"}
    MAX_FILE_SIZE_MB: int = 20
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

    # LLM API 配置（动态读取 OPENAI_API_KEY 与通用 OPENAI_BASE_URL）
    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "").strip()

    @property
    def openai_base_url(self) -> str | None:
        url = (
            os.getenv("OPENAI_BASE_URL", "").strip()
            or os.getenv("BASE_URL", "").strip()
        )
        return url or None

    MODEL_NAME: str = os.getenv("MODEL_NAME", "").strip()
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # Embedding 模型配置：统一通过 EMBEDDING_MODEL 配置模型名称，自动解析 models/ 目录下的同名本地目录
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "").strip()

    @property
    def embedding_model_path(self) -> str:
        local_dir = self.MODELS_DIR / self.EMBEDDING_MODEL
        return str(local_dir) if local_dir.exists() else self.EMBEDDING_MODEL

    # 切分默认参数
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))

    # 检索与余弦距离阈值配置
    DEFAULT_TOP_K: int = int(os.getenv("TOP_K", "5"))
    DEFAULT_REFUSAL_THRESHOLD: float = float(os.getenv("REFUSAL_THRESHOLD", "0.46"))
    DEFAULT_FALLBACK_THRESHOLD: float = float(os.getenv("FALLBACK_THRESHOLD", "0.36"))

    # 预置典型问答与演示示例（单数据源管理，供 API 与前端共享）
    SAMPLE_QUESTIONS: ClassVar[list[dict[str, str]]] = [
        {
            "id": "direct_fact",
            "category": "直接事实题",
            "subtitle": "迟到豁免次数与事假扣薪公式",
            "query": "员工每个自然月有几次15分钟以内的轻微迟到豁免机会？事假扣薪公式是什么？",
            "theme": "blue",
        },
        {
            "id": "multi_hop",
            "category": "跨段多跳计算",
            "subtitle": "深圳同性拼房上限与超标审批",
            "query": "两名P5级普通工程师同去深圳出差3天（住2晚）并选择同性拼房住一间标间，两晚住宿费合计最高可报销多少元？若超标12%找谁审批？",
            "theme": "indigo",
        },
        {
            "id": "error_code",
            "category": "专有报错码",
            "subtitle": "ERR-4029 与 ERR-5003 排查",
            "query": "私有化部署调用 /v1/knowledge/ask 接口返回 HTTP 429 和 ERR-4029 是什么原因？如果是 ERR-5003 又该如何处理？",
            "theme": "purple",
        },
        {
            "id": "refusal",
            "category": "无答案拒答题",
            "subtitle": "春节年会奖品与班车发车时间",
            "query": "公司2026年春节年会的抽奖一等奖奖品是什么？班车几点发车？",
            "theme": "amber",
        },
    ]


settings = Settings()

# 确保必要目录存在
for _dir in [
    settings.MODELS_DIR,
    settings.DATA_DIR,
    settings.DOCS_DIR,
    settings.CHROMA_DIR,
    settings.OUTPUTS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)
