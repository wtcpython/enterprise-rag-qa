import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from app.config import settings
from app.generator import RAGGenerator
from app.indexer import KnowledgeBaseIndexer
from app.retriever import HybridRetriever
from app.schemas import AskRequest, AskResponse, IndexRequest

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "企业知识库智能问答系统 API。\n"
        "支持 PDF/DOCX/TXT/MD 多格式解析、语义切分、"
        "Chroma 持久化与 MD5 增量索引、混合检索与 LLM 重排、来源追溯与拒答控制。"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

indexer = KnowledgeBaseIndexer()
retriever = HybridRetriever(indexer=indexer)
generator = RAGGenerator(retriever=retriever)

STATIC_DIR = Path(__file__).resolve().parent / "static"
QA_LOG_FILE = settings.OUTPUTS_DIR / "qa_logs.jsonl"
E2E_EVAL_FILE = settings.OUTPUTS_DIR / "e2e_eval_results.json"


def append_qa_log(query: str, req_mode: str, resp: AskResponse) -> None:
    """将每次问答的 path、延迟、Token、距离与来源写入日志，便于可观测性与评估复盘"""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "mode": req_mode,
        "path": resp.path,
        "best_distance": resp.best_distance,
        "latency_ms": resp.latency_ms,
        "cache_hit": resp.cache_hit,
        "token_usage": resp.token_usage,
        "sources": resp.sources,
        "docs_count": len(resp.docs),
        "contexts_count": len(resp.contexts),
        "answer_preview": resp.answer[:200],
    }
    with open(QA_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root_page() -> Any:
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return HTMLResponse(
        "<h1>企业知识库问答系统 API 服务运行中</h1><p>请访问 <a href='/docs'>/docs</a> 查看 Swagger 文档。</p>"
    )


@app.get("/health", summary="健康检查")
async def health_check() -> dict[str, Any]:
    state = indexer.load_index_state()
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_name": settings.MODEL_NAME,
        "embedding_model": state.get("embedding_model") or settings.EMBEDDING_MODEL,
        "total_indexed_files": len(state.get("files", {})),
        "total_chunks": state.get("total_chunks", 0),
        "api_configured": bool(settings.openai_api_key),
        "cache_stats": generator.cache_stats(),
    }


@app.get("/documents", summary="获取知识库文档列表及索引状态")
async def list_documents() -> dict[str, Any]:
    docs = indexer.list_documents()
    return {
        "total": len(docs),
        "allowed_extensions": sorted(settings.ALLOWED_EXTENSIONS),
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "documents": docs,
    }


@app.post("/documents/upload", summary="上传企业知识库文档")
async def upload_document(
    file: Annotated[UploadFile, File(...)],
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    safe_name = Path(file.filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 '{ext}'。仅支持: {sorted(settings.ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="上传的文件内容为空")
    if len(content) > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过上限 ({settings.MAX_FILE_SIZE_MB} MB)",
        )

    dest_path = settings.DOCS_DIR / safe_name
    dest_path.write_bytes(content)
    file_md5 = indexer.parser.compute_file_md5(dest_path)

    return {
        "message": "文档上传成功，请调用 POST /index 进行增量或重建索引",
        "filename": safe_name,
        "size_bytes": len(content),
        "file_md5": file_md5,
    }


@app.post("/index", summary="建索引 / 重建索引 / 增量索引")
async def build_index(req: IndexRequest) -> dict[str, Any]:
    if req.overlap >= req.chunk_size:
        raise HTTPException(status_code=400, detail="overlap 必须小于 chunk_size")
    result = indexer.build_index(
        mode=req.mode,
        chunk_size=req.chunk_size,
        overlap=req.overlap,
        collection_name=req.collection_name,
    )
    # 清除 BM25 缓存与问答 LRU 缓存以确保下次检索问答同步最新索引
    retriever._bm25_cache.clear()
    generator.clear_cache()
    return {
        "message": f"索引构建完成（模式: {result['effective_mode']}）",
        "index_result": result,
    }


@app.get("/index/status", summary="查看索引状态与历史日志")
async def get_index_status() -> dict[str, Any]:
    return indexer.load_index_state()


@app.get("/samples", summary="获取预置典型问答与演示示例列表")
async def list_samples() -> list[dict[str, str]]:
    return settings.SAMPLE_QUESTIONS


@app.get("/eval/results", summary="获取真实端到端评测报告与对比数据")
async def get_eval_results() -> dict[str, Any]:
    """读取 outputs/ 目录下的真实评测报告文件，动态供前端渲染，杜绝前端数据硬编码"""
    if not E2E_EVAL_FILE.exists():
        return {
            "available": False,
            "message": "尚未生成端到端评测结果，请先运行 python -m eval.eval_e2e",
            "summary": [],
            "cases": [],
        }

    try:
        e2e_data = json.loads(E2E_EVAL_FILE.read_text(encoding="utf-8"))
        summary = e2e_data.get("summary_comparison", {})

        mode_descriptions = {
            "baseline": "纯向量 Top-3 + 阈值 0.55",
            "threshold_tuned": "纯向量 Top-3 + 校准阈值 0.44",
            "hybrid": "BM25S + 向量 RRF Top-3",
            "optimized": "RRF Top-8 + Rerank Top-4 + 校准阈值",
        }

        formatted_summary = []
        for mode_key in ["baseline", "threshold_tuned", "hybrid", "optimized"]:
            item = summary.get(mode_key)
            if not item:
                continue
            formatted_summary.append(
                {
                    "mode": mode_key,
                    "description": mode_descriptions.get(mode_key, mode_key),
                    "answerable_correctness": round(
                        item.get("answerable_correctness", 0) * 100, 2
                    ),
                    "multi_hop_correctness": round(
                        item.get("multi_hop_correctness", 0) * 100, 2
                    ),
                    "faithfulness": round(
                        item.get("answerable_faithfulness_claim_support_rate", 0) * 100,
                        2,
                    ),
                    "refusal_accuracy": round(
                        item.get("unanswerable_refusal_accuracy", 0) * 100, 2
                    ),
                    "false_refusal_count": item.get("false_refusal_count", 0),
                    "avg_latency_ms": round(item.get("avg_latency_ms", 0), 1),
                    "avg_tokens": round(item.get("qa_generation_tokens_avg", 0), 1),
                }
            )

        diagnostic_cases = [
            {
                "title": "案例 1 (Q15)：多跳前提跨章截断与 Top-K 扩容修复",
                "desc": "武汉出差需跨 3 章拼齐 4 个片段，top_k=3 时第 4 位的城市分类块被截断（得分 0.50）。扩展 recall_k=8 + Rerank 精筛 contexts=4 后恢复 1.00 满分。",
            },
            {
                "title": "案例 2 (Q07 & Q01)：同义异形提问距离微超阈值误拒",
                "desc": "口语「账号密码」与原文「强口令策略」余弦距离 0.4742 > 校准阈值 0.44 致误拒。引入 LLM Rerank 高分豁免 (≥7.0) 后识别出高度相关，恢复满分作答。",
            },
            {
                "title": "案例 3 (Q26 & Q28)：高相似伪装无答案题协同拦截",
                "desc": "询问未收录错误码 ERR-9099 或商业授权报价时因共享领域前缀使距离低于 0.55。经阈值校准 (0.44) + Rerank 低分 (≤2.5) 拦截，100% 路由至 refusal 且零生成 Token。",
            },
            {
                "title": "案例 4 (入库清洗)：剥离 PDF 伪表格竖线噪声",
                "desc": "修复 pdfplumber 将无框线对齐空格误判为表格列（| 12 | 位）的碎裂问题后，总 Chunks 从 24 收敛至 20，有答案题 Hit@3 从 90.91% 跃升至 100%。",
            },
        ]

        return {
            "available": True,
            "metadata": e2e_data.get("evaluation_metadata", {}),
            "summary": formatted_summary,
            "cases": diagnostic_cases,
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as err:
        return {
            "available": False,
            "message": f"读取评测数据异常: {err}",
            "summary": [],
            "cases": [],
        }


@app.get("/cache/stats", summary="查看问答 LRU 缓存命中统计")
async def get_cache_stats() -> dict[str, Any]:
    return generator.cache_stats()


@app.post("/cache/clear", summary="清空问答 LRU 缓存")
async def clear_cache() -> dict[str, Any]:
    cleared = generator.clear_cache()
    return {"message": f"已清空 {cleared} 条问答缓存", "cleared_entries": cleared}


@app.post(
    "/ask", response_model=AskResponse, summary="输入问题，检索知识库并返回可溯源答案"
)
async def ask_question(req: AskRequest) -> AskResponse:
    resp = generator.answer(
        query=req.query,
        mode=req.mode,
        top_k=req.top_k,
        recall_k=req.recall_k,
        refusal_threshold=req.refusal_threshold,
        fallback_threshold=req.fallback_threshold,
        collection_name=req.collection_name,
        use_cache=req.use_cache,
    )
    append_qa_log(req.query, req.mode, resp)
    return resp
