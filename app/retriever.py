import re
import time
from typing import Any

import bm25s
import rjieba
from openai import OpenAI, OpenAIError

from app.config import settings
from app.indexer import KnowledgeBaseIndexer
from app.schemas import SourceItem
from eval.utils_json import parse_rerank_scores_fallback

RERANK_SYSTEM_PROMPT = """你是一个专业的企业知识库检索重排器（Reranker）。
请评估每个候选文档片段对回答用户问题的相关程度，打分范围为 0 到 10 分（10分为直接包含精确答案，0分为完全无关）。
必须严格输出 JSON 数组格式，不要输出任何额外解释文字。格式示例：
[
  {"index": 0, "score": 9.5},
  {"index": 1, "score": 2.0}
]"""


class HybridRetriever:
    """
    多策略检索器：
    - 支持纯向量检索
    - 支持基于 bm25s + rjieba 与向量双路召回的 RRF 融合
    - 支持基于 LLM 的语义重排
    - 区分初始召回列表 `docs` 与最终上下文 `contexts`
    """

    def __init__(self, indexer: KnowledgeBaseIndexer | None = None) -> None:
        self.indexer = indexer or KnowledgeBaseIndexer()
        self._bm25_cache: dict[str, tuple[int, bm25s.BM25, list[dict[str, Any]]]] = {}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """基于 rjieba 的中英文与专有编号分词"""
        text_lower = text.lower()
        codes = re.findall(r"[a-z]{2,}-\d+(?:-\d+)?(?:-[a-z0-9]+)?", text_lower)
        words = [
            w.strip()
            for w in rjieba.cut(text_lower)
            if w.strip() and not re.match(r"^[\s\W_]+$", w.strip())
        ]
        return words + codes

    def _get_bm25_index(
        self, collection_name: str
    ) -> tuple[bm25s.BM25 | None, list[dict[str, Any]]]:
        collection = self.indexer.get_collection(collection_name)
        count = collection.count()
        if count == 0:
            return None, []

        cached = self._bm25_cache.get(collection_name)
        if cached and cached[0] == count:
            return cached[1], cached[2]

        all_data = collection.get(include=["documents", "metadatas"])
        docs_list: list[dict[str, Any]] = []
        tokenized_corpus: list[list[str]] = []

        ids = all_data.get("ids", [])
        documents = all_data.get("documents", []) or []
        metadatas = all_data.get("metadatas", []) or []

        for cid, doc_text, meta in zip(ids, documents, metadatas):
            item = {
                "chunk_id": cid,
                "text": doc_text,
                "metadata": meta or {},
            }
            docs_list.append(item)
            sec = (meta or {}).get("section", "")
            src = (meta or {}).get("source", "")
            tokenized_corpus.append(self.tokenize(f"{src} {sec} {doc_text}"))

        bm25 = bm25s.BM25()
        bm25.index(tokenized_corpus, show_progress=False)
        self._bm25_cache[collection_name] = (count, bm25, docs_list)
        return bm25, docs_list

    @staticmethod
    def _build_source_item(
        chunk_id: Any,
        text: Any,
        meta: dict[str, Any] | Any,
        distance: float,
    ) -> SourceItem:
        meta_dict = meta if isinstance(meta, dict) else {}
        raw_idx = meta_dict.get("chunk_index", 0)
        raw_page = meta_dict.get("page")
        return SourceItem(
            source=str(meta_dict.get("source", "unknown")),
            chunk_id=str(chunk_id),
            chunk_index=int(raw_idx) if isinstance(raw_idx, (int, float, str)) else 0,
            section=str(meta_dict.get("section", "")),
            page=int(raw_page) if isinstance(raw_page, (int, float, str)) else None,
            distance=round(distance, 4),
            text=str(text),
        )

    def vector_search(
        self,
        query: str,
        top_k: int = 8,
        collection_name: str = "enterprise_kb",
    ) -> list[SourceItem]:
        """纯向量余弦距离检索"""
        collection = self.indexer.get_collection(collection_name)
        total = collection.count()
        if total == 0:
            return []

        n_results = min(top_k, total)
        query_vec = self.indexer.embed_texts([query], is_query=True)
        res = collection.query(
            query_embeddings=query_vec,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        return [
            self._build_source_item(cid, text, meta, float(dist))
            for cid, text, meta, dist in zip(ids, docs, metas, dists)
        ]

    def hybrid_search(
        self,
        query: str,
        recall_k: int = 8,
        collection_name: str = "enterprise_kb",
        rrf_k: int = 60,
    ) -> list[SourceItem]:
        """
        向量检索 + bm25s 稀疏检索 + RRF（倒数秩融合）双路召回。
        针对所有候选统一计算真实的向量余弦距离与 RRF 融合分数。
        """
        collection = self.indexer.get_collection(collection_name)
        total = collection.count()
        if total == 0:
            return []

        vec_candidates = self.vector_search(
            query=query,
            top_k=min(total, max(recall_k * 3, 25)),
            collection_name=collection_name,
        )
        vec_map: dict[str, tuple[int, SourceItem]] = {
            item.chunk_id: (rank, item)
            for rank, item in enumerate(vec_candidates, start=1)
        }

        bm25, corpus_docs = self._get_bm25_index(collection_name)
        if bm25 is None or not corpus_docs:
            return vec_candidates[:recall_k]

        query_tokens = self.tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)
        ranked_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: float(bm25_scores[i]),
            reverse=True,
        )[: max(recall_k * 2, 15)]

        bm25_rank_map: dict[str, tuple[int, float, dict[str, Any]]] = {}
        for rank, idx in enumerate(ranked_bm25_indices, start=1):
            score = float(bm25_scores[idx])
            if score > 0:
                doc_entry = corpus_docs[idx]
                bm25_rank_map[doc_entry["chunk_id"]] = (rank, score, doc_entry)

        all_candidate_ids = set(vec_map.keys()) | set(bm25_rank_map.keys())
        fused_list: list[SourceItem] = []

        for cid in all_candidate_ids:
            vec_info = vec_map.get(cid)
            bm25_info = bm25_rank_map.get(cid)

            vec_rank = vec_info[0] if vec_info else len(corpus_docs) + 50
            bm25_rank = bm25_info[0] if bm25_info else len(corpus_docs) + 50

            # RRF 融合打分：向量权重 1.0，BM25 权重 0.85
            rrf_score = (1.0 / (rrf_k + vec_rank)) + (0.85 / (rrf_k + bm25_rank))

            if vec_info:
                base_item = vec_info[1].model_copy()
            elif bm25_info:
                doc_entry = bm25_info[2]
                base_item = self._build_source_item(
                    cid, doc_entry["text"], doc_entry["metadata"], 0.45
                )
            else:
                continue

            base_item.rrf_score = round(rrf_score, 6)
            fused_list.append(base_item)

        fused_list.sort(key=lambda x: (-(x.rrf_score or 0.0), x.distance))
        return fused_list[:recall_k]

    def rerank_with_llm(
        self,
        query: str,
        docs: list[SourceItem],
        top_k: int = 4,
    ) -> tuple[list[SourceItem], int, bool]:
        """
        使用 LLM 对召回的候选 `docs` 进行相关性打分与重排，筛选出 Top-K `contexts`。
        返回: (reranked_contexts, rerank_total_tokens, rerank_degraded)
        """
        if not docs:
            return [], 0, False

        candidates_prompt_lines = []
        for idx, d in enumerate(docs):
            snippet = d.text[:350].replace("\n", " ")
            candidates_prompt_lines.append(
                f"[{idx}] 来源:{d.source} ({d.section}) 内容:{snippet}"
            )

        user_prompt = (
            f"用户问题：{query}\n\n"
            f"候选文档片段列表：\n" + "\n".join(candidates_prompt_lines) + "\n\n"
            "请对以上每个候选片段与用户问题的相关性打分（0~10分），仅输出 JSON 数组："
        )

        try:
            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
            resp = client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[
                    {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=300,
                extra_body={"enable_thinking": False},
            )
            raw_content = resp.choices[0].message.content or ""
            usage_total = resp.usage.total_tokens if resp.usage else 0

            score_items = parse_rerank_scores_fallback(raw_content, len(docs))
            if not score_items:
                return [d.model_copy() for d in docs[:top_k]], usage_total, True

            score_map = {item["index"]: item["score"] for item in score_items}
            scored_docs: list[SourceItem] = []
            for idx, d in enumerate(docs):
                copy_d = d.model_copy()
                copy_d.rerank_score = round(float(score_map.get(idx, 0.0)), 2)
                scored_docs.append(copy_d)

            scored_docs.sort(key=lambda x: (-(x.rerank_score or 0.0), x.distance))
            return scored_docs[:top_k], usage_total, False
        except OpenAIError, OSError, ValueError, TypeError, KeyError:
            return [d.model_copy() for d in docs[:top_k]], 0, True

    def retrieve(
        self,
        query: str,
        mode: str = "optimized",
        top_k: int = 5,
        recall_k: int = 8,
        collection_name: str = "enterprise_kb",
    ) -> dict[str, Any]:
        """
        统一检索入口，支持五种检索模式：
        - baseline: 纯向量检索，初始阈值 0.55
        - threshold_tuned: 纯向量检索，使用校准后的距离阈值
        - hybrid: 向量 + bm25s RRF 混合检索
        - rerank: 纯向量召回 recall_k + LLM Rerank 精选 top_k contexts
        - optimized: 混合检索召回 recall_k + LLM Rerank 精选 top_k contexts + 校准阈值
        """
        t0 = time.perf_counter()
        rerank_tokens = 0
        rerank_degraded = False

        if mode in {"baseline", "threshold_tuned"}:
            docs = self.vector_search(
                query=query, top_k=max(top_k, recall_k), collection_name=collection_name
            )
            contexts = [d.model_copy() for d in docs[:top_k]]
        elif mode == "hybrid":
            docs = self.hybrid_search(
                query=query,
                recall_k=max(top_k, recall_k),
                collection_name=collection_name,
            )
            contexts = [d.model_copy() for d in docs[:top_k]]
        elif mode == "rerank":
            docs = self.vector_search(
                query=query, top_k=max(top_k, recall_k), collection_name=collection_name
            )
            contexts, rerank_tokens, rerank_degraded = self.rerank_with_llm(
                query, docs, top_k=top_k
            )
        else:  # optimized
            docs = self.hybrid_search(
                query=query,
                recall_k=max(top_k, recall_k),
                collection_name=collection_name,
            )
            min_vec_dist = min((d.distance for d in docs), default=1.0)
            if min_vec_dist > 0.52:
                contexts = [d.model_copy() for d in docs[:top_k]]
            else:
                contexts, rerank_tokens, rerank_degraded = self.rerank_with_llm(
                    query, docs, top_k=top_k
                )

        retrieval_ms = int((time.perf_counter() - t0) * 1000)
        best_distance = min((d.distance for d in docs), default=None)

        return {
            "docs": docs,
            "contexts": contexts,
            "best_distance": best_distance,
            "retrieval_ms": retrieval_ms,
            "rerank_tokens": rerank_tokens,
            "rerank_degraded": rerank_degraded,
        }
