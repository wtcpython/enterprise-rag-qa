import time
from collections import OrderedDict
from typing import Any, Literal

from openai import OpenAI, OpenAIError

from app.config import settings
from app.retriever import HybridRetriever
from app.schemas import AskResponse, SourceItem

QA_SYSTEM_PROMPT = """你是企业内部知识库智能问答助手。请严格遵循以下铁律回答员工问题：
1. 【只依据材料回答】：你只能依据下方提供的【检索材料（Contexts）】回答问题，严禁动用任何外部常识、严禁编造或脑补材料中未出现的数字、条款或流程。
2. 【无相关内容明确拒答】：如果检索材料中没有包含回答该问题所需的信息，或者材料与问题无关，你必须直接回答：“抱歉，当前企业知识库材料中没有提到该问题相关的信息，无法为您解答。”，绝不能猜测或拼凑答案。
3. 【强制标注来源】：回答中的关键事实、数据、条款，必须在句末标注来源文件名与章节/片段编号，格式如：`[来源: 员工手册与考勤休假制度.md - 第三章]`。
4. 【跨段综合与精确计算】：当问题涉及多个段落或计算时，请分步引用对应材料条款并给出清晰的推导结果。"""


FALLBACK_SYSTEM_PROMPT = """你是企业内部知识库智能问答助手。当前检索到的材料相关性处于临界区间（降级模式）。
请极其谨慎地核对下方【检索材料（Contexts）】：
1. 若材料中确实包含问题所需的部分或全部依据，请仅就材料明确提到的部分作答，必须在每个关键结论句末标注 `[来源: 文件名 - 章节]`，并在开头提示“【提示：检索相关度中等，以下仅基于现有相近条款供参考】”。
2. 若材料实际上无法回答该问题，请直接明确说明：“抱歉，当前企业知识库材料中没有提到该问题的确切信息。”，严禁猜测或编造。"""


class RAGGenerator:
    """
    RAG 问答生成与路径控制器：
    负责将检索结果按照距离阈值与重排得分路由至 `normal | fallback | refusal | error` 四条路径，
    组装上下文调用大模型，支持 LRU 问答缓存，并记录耗时与 Token 消耗。
    """

    def __init__(
        self, retriever: HybridRetriever | None = None, max_cache_size: int = 128
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.max_cache_size = max_cache_size
        self._cache: OrderedDict[tuple[Any, ...], AskResponse] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    def clear_cache(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count

    def cache_stats(self) -> dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        return {
            "entries": len(self._cache),
            "max_size": self.max_cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": round(self._cache_hits / total, 4) if total > 0 else 0.0,
        }

    def _put_cache(
        self, key: tuple[Any, ...], resp: AskResponse, use_cache: bool
    ) -> AskResponse:
        if not use_cache or resp.path == "error":
            return resp
        if len(self._cache) >= self.max_cache_size and key not in self._cache:
            self._cache.popitem(last=False)
        self._cache[key] = resp
        self._cache.move_to_end(key)
        return resp

    @staticmethod
    def resolve_thresholds(
        mode: str,
        explicit_refusal: float | None = None,
        explicit_fallback: float | None = None,
    ) -> tuple[float, float]:
        """
        根据检索模式返回 (refusal_threshold, fallback_threshold)：
        - baseline 模式使用初始默认阈值 0.55 / 0.48
        - 其他模式使用基于评估集距离分布校准后的阈值
        """
        if explicit_refusal is not None and explicit_fallback is not None:
            return explicit_refusal, explicit_fallback

        if mode == "baseline":
            return (explicit_refusal or 0.55), (explicit_fallback or 0.48)

        return (
            explicit_refusal
            if explicit_refusal is not None
            else settings.DEFAULT_REFUSAL_THRESHOLD,
            explicit_fallback
            if explicit_fallback is not None
            else settings.DEFAULT_FALLBACK_THRESHOLD,
        )

    @staticmethod
    def build_context_text(contexts: list[SourceItem]) -> str:
        blocks = []
        for idx, item in enumerate(contexts, start=1):
            header = (
                f"【材料片段 {idx}】来源文件: {item.source} | 章节: {item.section} "
                f"| chunk_id: {item.chunk_id} | 向量距离: {item.distance}"
            )
            blocks.append(f"{header}\n{item.text}")
        return "\n\n---\n\n".join(blocks)

    def answer(
        self,
        query: str,
        mode: str = "optimized",
        top_k: int = 5,
        recall_k: int = 8,
        refusal_threshold: float | None = None,
        fallback_threshold: float | None = None,
        collection_name: str = "enterprise_kb",
        use_cache: bool = True,
    ) -> AskResponse:
        t_start = time.perf_counter()
        cache_key = (
            query.strip(),
            mode,
            top_k,
            recall_k,
            refusal_threshold,
            fallback_threshold,
            collection_name,
        )

        if use_cache and cache_key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(cache_key)
            cached = self._cache[cache_key]
            latency_ms = max(0, int((time.perf_counter() - t_start) * 1000))
            return cached.model_copy(
                update={
                    "cache_hit": True,
                    "latency_ms": latency_ms,
                    "token_usage": {"prompt": 0, "completion": 0, "total": 0},
                    "extra_metrics": {
                        **cached.extra_metrics,
                        "cache_hit": True,
                        "original_latency_ms": cached.latency_ms,
                    },
                }
            )

        self._cache_misses += 1

        try:
            ret = self.retriever.retrieve(
                query=query,
                mode=mode,
                top_k=top_k,
                recall_k=recall_k,
                collection_name=collection_name,
            )
            docs: list[SourceItem] = ret["docs"]
            contexts: list[SourceItem] = ret["contexts"]
            best_dist: float | None = ret["best_distance"]
            retrieval_ms: int = ret["retrieval_ms"]
            rerank_tokens: int = ret["rerank_tokens"]
            rerank_degraded: bool = ret["rerank_degraded"]

            ref_th, fb_th = self.resolve_thresholds(
                mode, refusal_threshold, fallback_threshold
            )

            # 1. 空库或无召回直接拒答
            if not docs or best_dist is None:
                latency_ms = int((time.perf_counter() - t_start) * 1000)
                return AskResponse(
                    answer="抱歉，当前企业知识库中尚无可用文档或未检索到任何相关材料，无法回答您的问题。",
                    sources=[],
                    docs=[],
                    contexts=[],
                    path="refusal",
                    mode=mode,
                    best_distance=None,
                    latency_ms=latency_ms,
                    token_usage={"prompt": 0, "completion": 0, "total": 0},
                    extra_metrics={
                        "retrieval_ms": retrieval_ms,
                        "llm_ms": 0,
                        "rerank_tokens": rerank_tokens,
                        "refusal_threshold": ref_th,
                        "fallback_threshold": fb_th,
                        "reason": "empty_recall",
                    },
                )

            # 2. 结合向量距离阈值与 Rerank 分数判定路径
            top_rerank_score = (
                contexts[0].rerank_score
                if (contexts and contexts[0].rerank_score is not None)
                else None
            )

            is_refusal_by_dist = best_dist > ref_th
            # 若启用了混合检索且 BM25 命中了极强的专有编号/词汇，且 Rerank 打分很高 (>=7.0)，允许赦免距离微超阈值的情况
            if (
                is_refusal_by_dist
                and top_rerank_score is not None
                and top_rerank_score >= 7.0
            ):
                is_refusal_by_dist = False

            # 反之，若 Rerank 明确判定所有候选片段均无关（最高分 <= 2.5），即便距离低于阈值也触发拒答
            is_refusal_by_rerank = (
                top_rerank_score is not None and top_rerank_score <= 2.5
            )

            if is_refusal_by_dist or is_refusal_by_rerank:
                latency_ms = int((time.perf_counter() - t_start) * 1000)
                sources_brief = [
                    {"source": c.source, "chunk_id": c.chunk_id, "distance": c.distance}
                    for c in contexts
                ]
                refusal_resp = AskResponse(
                    answer=(
                        "抱歉，当前企业知识库材料中没有提到与您问题相关的信息（检索最佳距离 "
                        f"{best_dist:.4f} 超出相关性安全阈值），系统拒绝编造回答。"
                    ),
                    sources=sources_brief,
                    docs=docs,
                    contexts=[],  # 拒答路径不将无关材料送入生成模型
                    path="refusal",
                    mode=mode,
                    best_distance=best_dist,
                    latency_ms=latency_ms,
                    cache_hit=False,
                    token_usage={"prompt": 0, "completion": 0, "total": 0},
                    extra_metrics={
                        "retrieval_ms": retrieval_ms,
                        "llm_ms": 0,
                        "rerank_tokens": rerank_tokens,
                        "refusal_threshold": ref_th,
                        "fallback_threshold": fb_th,
                        "reason": "distance_or_rerank_refusal",
                    },
                )
                return self._put_cache(cache_key, refusal_resp, use_cache)

            # 判断走 normal 还是 fallback 路径：若 Rerank 明确给出高分 (>=7.0)，说明相关性强，直接走 normal
            is_fallback_by_dist = best_dist > fb_th and not (
                top_rerank_score is not None and top_rerank_score >= 7.0
            )
            path_label: Literal["normal", "fallback", "refusal", "error"]
            if (
                is_fallback_by_dist
                or rerank_degraded
                or (top_rerank_score is not None and top_rerank_score < 5.5)
            ):
                path_label = "fallback"
                sys_prompt = FALLBACK_SYSTEM_PROMPT
            else:
                path_label = "normal"
                sys_prompt = QA_SYSTEM_PROMPT

            # 3. 组装 Prompt 并调用生成模型
            context_block = self.build_context_text(contexts)
            user_message = (
                f"【检索材料（Contexts）】\n{context_block}\n\n"
                f"【员工提问】\n{query}\n\n"
                "请严格依据上述检索材料作答，并标注来源文件名及章节："
            )

            t_llm = time.perf_counter()
            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
            resp = client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=900,
                extra_body={"enable_thinking": False},
            )
            llm_ms = int((time.perf_counter() - t_llm) * 1000)

            answer_text = (resp.choices[0].message.content or "").strip()
            prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
            completion_tokens = resp.usage.completion_tokens if resp.usage else 0
            total_tokens = resp.usage.total_tokens if resp.usage else 0

            # 如果生成模型阅读材料后明确判断“材料中没有提到”，将最终 path 校准为 refusal
            refusal_keywords = [
                "材料中没有提到",
                "知识库材料中没有提到",
                "未提到该问题",
                "无法为您解答",
                "没有找到相关",
            ]
            if (
                any(kw in answer_text for kw in refusal_keywords)
                and len(answer_text) < 160
            ):
                path_label = "refusal"

            latency_ms = int((time.perf_counter() - t_start) * 1000)
            sources_brief = [
                {"source": c.source, "chunk_id": c.chunk_id, "distance": c.distance}
                for c in contexts
            ]

            final_resp = AskResponse(
                answer=answer_text,
                sources=sources_brief,
                docs=docs,
                contexts=contexts,
                path=path_label,
                mode=mode,
                best_distance=best_dist,
                latency_ms=latency_ms,
                cache_hit=False,
                token_usage={
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens,
                },
                extra_metrics={
                    "retrieval_ms": retrieval_ms,
                    "llm_ms": llm_ms,
                    "rerank_tokens": rerank_tokens,
                    "total_tokens_with_rerank": total_tokens + rerank_tokens,
                    "refusal_threshold": ref_th,
                    "fallback_threshold": fb_th,
                    "rerank_degraded": rerank_degraded,
                },
            )
            return self._put_cache(cache_key, final_resp, use_cache)

        except (OpenAIError, OSError, ValueError, TypeError, KeyError) as exc:
            latency_ms = int((time.perf_counter() - t_start) * 1000)
            return AskResponse(
                answer=f"系统处理问答请求时发生异常，已触发安全降级：{exc}",
                sources=[],
                docs=[],
                contexts=[],
                path="error",
                mode=mode,
                best_distance=None,
                latency_ms=latency_ms,
                token_usage={"prompt": 0, "completion": 0, "total": 0},
                extra_metrics={"error": str(exc)},
            )
