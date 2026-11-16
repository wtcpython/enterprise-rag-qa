import json
import re
import statistics
import time
from typing import Any, TypedDict

from app.config import settings
from app.indexer import KnowledgeBaseIndexer
from app.retriever import HybridRetriever
from app.schemas import SourceItem


def normalize_for_match(text: str) -> str:
    """归一化空白符与大小写，用于判断标准答案片段是否被召回 chunk 覆盖"""
    return re.sub(r"\s+", "", text).lower()


def is_snippet_in_chunk(snippet: str, chunk_text: str) -> bool:
    return normalize_for_match(snippet) in normalize_for_match(chunk_text)


def evaluate_single_query_retrieval(
    retrieved: list[SourceItem],
    expected_snippets: list[str],
    k_values: tuple[int, ...] = (3, 5, 8),
) -> dict[str, float]:
    """
    计算单条查询的检索指标：
    - Hit@K: Top-K 中只要命中至少 1 个必含片段即为 1.0，否则为 0.0
    - Recall@K: Top-K 中命中的必含片段数量 / 该题全部必含片段总数
    - Precision@K: Top-K 中包含至少 1 个必含片段的有效 chunk 数量 / K
    - MRR@K: 第一个命中必含片段的 chunk 排名的倒数
    """
    metrics: dict[str, float] = {}
    if not expected_snippets:
        return metrics

    for k in k_values:
        top_k_chunks = retrieved[:k]
        # 1. 计算覆盖了多少个 expected_snippets
        covered_snippets = 0
        for snip in expected_snippets:
            if any(is_snippet_in_chunk(snip, c.text) for c in top_k_chunks):
                covered_snippets += 1

        # 2. 计算有多少个 chunk 是相关的，以及首个相关 chunk 的排名
        relevant_chunks = 0
        first_hit_rank = 0
        for rank, c in enumerate(top_k_chunks, start=1):
            is_rel = any(
                is_snippet_in_chunk(snip, c.text) for snip in expected_snippets
            )
            if is_rel:
                relevant_chunks += 1
                if first_hit_rank == 0:
                    first_hit_rank = rank

        hit_at_k = 1.0 if covered_snippets > 0 else 0.0
        recall_at_k = covered_snippets / len(expected_snippets)
        precision_at_k = relevant_chunks / k
        mrr_at_k = (1.0 / first_hit_rank) if first_hit_rank > 0 else 0.0

        metrics[f"Hit@{k}"] = round(hit_at_k, 4)
        metrics[f"Recall@{k}"] = round(recall_at_k, 4)
        metrics[f"Precision@{k}"] = round(precision_at_k, 4)
        metrics[f"MRR@{k}"] = round(mrr_at_k, 4)

    return metrics


def run_retrieval_setting(
    retriever: HybridRetriever,
    dataset: list[dict[str, Any]],
    search_type: str = "vector",
    collection_name: str = "enterprise_kb",
    refusal_threshold: float = 0.46,
    k_values: tuple[int, ...] = (3, 5, 8),
) -> dict[str, Any]:
    """
    在给定索引集合上运行纯检索评估（不调用 LLM）：
    - 有答案题（direct_fact + multi_hop，共22题）：计算 Hit@K, Recall@K, Precision@K, MRR@K
    - 无答案题（unanswerable，共8题）：绝不混入 Hit/MRR，单独统计基于距离阈值的拒答准确率
    - 同时收集全部题目的 dists[0] 分布与逐题检索日志
    """
    max_k = max(k_values)
    answerable_metrics: list[dict[str, float]] = []
    direct_metrics: list[dict[str, float]] = []
    multihop_metrics: list[dict[str, float]] = []

    answerable_dists: list[float] = []
    unanswerable_dists: list[float] = []
    per_query_logs: list[dict[str, Any]] = []

    for item in dataset:
        qid = item["id"]
        category = item["category"]
        query = item["query"]
        expected_snippets = item.get("expected_doc_snippets", [])

        t0 = time.perf_counter()
        if search_type == "vector":
            docs = retriever.vector_search(
                query=query, top_k=max_k, collection_name=collection_name
            )
        else:
            docs = retriever.hybrid_search(
                query=query, recall_k=max_k, collection_name=collection_name
            )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # dists[0] 为向量路最相似距离（对混合检索也取候选集中最小向量距离以便统一校准）
        vec_docs = (
            docs
            if search_type == "vector"
            else retriever.vector_search(
                query=query, top_k=1, collection_name=collection_name
            )
        )
        dist0 = vec_docs[0].distance if vec_docs else 1.0

        if category == "unanswerable":
            unanswerable_dists.append(dist0)
            q_metrics = {}
            is_refused = dist0 > refusal_threshold
            pred_path = "refusal" if is_refused else "normal"
        else:
            answerable_dists.append(dist0)
            q_metrics = evaluate_single_query_retrieval(
                docs, expected_snippets, k_values=k_values
            )
            answerable_metrics.append(q_metrics)
            if category == "direct_fact":
                direct_metrics.append(q_metrics)
            elif category == "multi_hop":
                multihop_metrics.append(q_metrics)
            is_refused = dist0 > refusal_threshold
            pred_path = "refusal" if is_refused else "normal"

        per_query_logs.append(
            {
                "id": qid,
                "category": category,
                "query": query,
                "dist0": round(dist0, 4),
                "predicted_path_by_dist": pred_path,
                "latency_ms": latency_ms,
                "metrics": q_metrics,
                "top_docs": [
                    {
                        "chunk_id": d.chunk_id,
                        "source": d.source,
                        "section": d.section,
                        "distance": d.distance,
                        "rrf_score": d.rrf_score,
                        "snippet_preview": d.text[:100].replace("\n", " "),
                    }
                    for d in docs[:5]
                ],
            }
        )

    def avg_dict(m_list: list[dict[str, float]]) -> dict[str, float]:
        if not m_list:
            return {}
        keys = m_list[0].keys()
        return {k: round(sum(d[k] for d in m_list) / len(m_list), 4) for k in keys}

    # 计算不同距离阈值下的拒答准确率（无答案题）与误拒率（有答案题）
    def eval_threshold(th: float) -> dict[str, float]:
        correct_refusals = sum(1 for d in unanswerable_dists if d > th)
        false_refusals = sum(1 for d in answerable_dists if d > th)
        refusal_acc = (
            correct_refusals / len(unanswerable_dists) if unanswerable_dists else 0.0
        )
        false_refusal_rate = (
            false_refusals / len(answerable_dists) if answerable_dists else 0.0
        )
        # 综合 F1 / 平衡准确率
        ans_pass_rate = 1.0 - false_refusal_rate
        balanced_acc = (refusal_acc + ans_pass_rate) / 2.0
        return {
            "threshold": th,
            "unanswerable_refusal_accuracy": round(refusal_acc, 4),
            "answerable_pass_rate": round(ans_pass_rate, 4),
            "false_refusal_rate": round(false_refusal_rate, 4),
            "balanced_accuracy": round(balanced_acc, 4),
        }

    # 扫描候选阈值寻找最优校准阈值
    candidate_thresholds = [round(0.28 + i * 0.02, 2) for i in range(18)]
    threshold_scan = [eval_threshold(th) for th in candidate_thresholds]
    best_th_entry = max(
        threshold_scan,
        key=lambda x: (x["balanced_accuracy"], x["unanswerable_refusal_accuracy"]),
    )

    return {
        "search_type": search_type,
        "collection_name": collection_name,
        "answerable_count": len(answerable_metrics),
        "unanswerable_count": len(unanswerable_dists),
        "overall_answerable_metrics": avg_dict(answerable_metrics),
        "direct_fact_metrics": avg_dict(direct_metrics),
        "multi_hop_metrics": avg_dict(multihop_metrics),
        "dist0_distribution": {
            "answerable": {
                "min": round(min(answerable_dists), 4),
                "mean": round(statistics.mean(answerable_dists), 4),
                "median": round(statistics.median(answerable_dists), 4),
                "p90": round(
                    sorted(answerable_dists)[int(len(answerable_dists) * 0.9)], 4
                ),
                "max": round(max(answerable_dists), 4),
                "values": [round(x, 4) for x in answerable_dists],
            },
            "unanswerable": {
                "min": round(min(unanswerable_dists), 4),
                "mean": round(statistics.mean(unanswerable_dists), 4),
                "median": round(statistics.median(unanswerable_dists), 4),
                "p90": round(
                    sorted(unanswerable_dists)[int(len(unanswerable_dists) * 0.9)], 4
                ),
                "max": round(max(unanswerable_dists), 4),
                "values": [round(x, 4) for x in unanswerable_dists],
            },
        },
        "uncalibrated_055_eval": eval_threshold(0.55),
        "calibrated_threshold_eval": best_th_entry,
        "threshold_scan": threshold_scan,
        "per_query_logs": per_query_logs,
    }


def populate_must_have_chunk_ids(
    indexer: KnowledgeBaseIndexer,
    dataset: list[dict[str, Any]],
    collection_name: str = "enterprise_kb",
) -> None:
    """将标准索引下的必含 chunk_id 回写到 eval_dataset.json 中，同时满足 expected_doc_snippets 与 must_have_chunk_ids"""
    collection = indexer.get_collection(collection_name)
    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data.get("ids", [])
    docs = all_data.get("documents", []) or []

    for item in dataset:
        snippets = item.get("expected_doc_snippets", [])
        matched_ids: list[str] = []
        for cid, text in zip(ids, docs):
            if any(is_snippet_in_chunk(snip, text) for snip in snippets):
                matched_ids.append(cid)
        item["must_have_chunk_ids"] = matched_ids

    settings.EVAL_DATASET_FILE.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    dataset = json.loads(settings.EVAL_DATASET_FILE.read_text(encoding="utf-8"))
    indexer = KnowledgeBaseIndexer()
    retriever = HybridRetriever(indexer=indexer)

    # 实验一：单变量切分参数 chunk_size / overlap 对比（包含 overlap=0 vs 真实 overlap 对比）
    class ChunkConfig(TypedDict):
        name: str
        chunk_size: int
        overlap: int

    chunk_configs: list[ChunkConfig] = [
        {"name": "chunk220_no_overlap", "chunk_size": 220, "overlap": 0},
        {"name": "chunk220_overlap50", "chunk_size": 220, "overlap": 50},
        {"name": "chunk400_overlap80", "chunk_size": 400, "overlap": 80},
        {"name": "chunk700_overlap100", "chunk_size": 700, "overlap": 100},
    ]

    chunk_exp_results: dict[str, Any] = {}
    for cfg in chunk_configs:
        col_name = f"eval_{cfg['name']}"
        print(
            f"\n[Retrieval Eval] 构建切分实验索引: {cfg['name']} (chunk_size={cfg['chunk_size']}, overlap={cfg['overlap']})..."
        )
        idx_log = indexer.build_index(
            mode="rebuild",
            chunk_size=cfg["chunk_size"],
            overlap=cfg["overlap"],
            collection_name=col_name,
        )
        retriever._bm25_cache.clear()
        res = run_retrieval_setting(
            retriever=retriever,
            dataset=dataset,
            search_type="vector",
            collection_name=col_name,
        )
        res["index_log"] = idx_log
        chunk_exp_results[cfg["name"]] = res
        m = res["overall_answerable_metrics"]
        print(
            f"  -> {cfg['name']} (chunks={idx_log['total_chunks']}): "
            f"Hit@3={m['Hit@3']}, Recall@3={m['Recall@3']}, Hit@5={m['Hit@5']}, Recall@5={m['Recall@5']}, MRR@5={m['MRR@5']}"
        )

    # 构建默认主索引 enterprise_kb (chunk_size=400, overlap=80)
    print(
        "\n[Retrieval Eval] 构建主知识库索引 enterprise_kb (chunk_size=400, overlap=80)..."
    )
    main_idx_log = indexer.build_index(
        mode="rebuild",
        chunk_size=400,
        overlap=80,
        collection_name="enterprise_kb",
    )
    # 测试增量索引（再次运行 incremental 应全部 skipped）
    inc_idx_log = indexer.build_index(
        mode="incremental",
        chunk_size=400,
        overlap=80,
        collection_name="enterprise_kb",
    )
    print(
        f"  -> 增量索引验证: skipped={len(inc_idx_log['skipped'])}/{len(inc_idx_log['skipped']) + len(inc_idx_log['added_or_updated'])}, 耗时={inc_idx_log['latency_ms']}ms"
    )

    populate_must_have_chunk_ids(indexer, dataset, collection_name="enterprise_kb")

    # 实验二：在主索引上对比 纯向量检索 (Baseline Vector) vs 混合检索 (Hybrid Vector+BM25 RRF)
    retriever._bm25_cache.clear()
    baseline_vec_res = run_retrieval_setting(
        retriever=retriever,
        dataset=dataset,
        search_type="vector",
        collection_name="enterprise_kb",
    )
    hybrid_res = run_retrieval_setting(
        retriever=retriever,
        dataset=dataset,
        search_type="hybrid",
        collection_name="enterprise_kb",
    )

    calibrated_th = baseline_vec_res["calibrated_threshold_eval"]["threshold"]
    print(
        f"\n[Threshold Calibration] 初始阈值 0.55 下无答案拒答准确率: {baseline_vec_res['uncalibrated_055_eval']['unanswerable_refusal_accuracy']}"
    )
    print(
        f"[Threshold Calibration] 校准后最优距离阈值: {calibrated_th} -> 无答案拒答准确率: {baseline_vec_res['calibrated_threshold_eval']['unanswerable_refusal_accuracy']}, 有答案通过率: {baseline_vec_res['calibrated_threshold_eval']['answerable_pass_rate']}"
    )

    summary = {
        "embedding_model": main_idx_log["embedding_model"],
        "main_index_log": main_idx_log,
        "incremental_index_verification_log": inc_idx_log,
        "calibrated_refusal_threshold": calibrated_th,
        "chunk_size_overlap_ablation": {
            k: {
                "chunk_size": v["index_log"]["chunk_size"],
                "overlap": v["index_log"]["overlap"],
                "total_chunks": v["index_log"]["total_chunks"],
                "overall_answerable_metrics": v["overall_answerable_metrics"],
                "direct_fact_metrics": v["direct_fact_metrics"],
                "multi_hop_metrics": v["multi_hop_metrics"],
            }
            for k, v in chunk_exp_results.items()
        },
        "baseline_vector_retrieval": baseline_vec_res,
        "hybrid_rrf_retrieval": hybrid_res,
    }

    out_path = settings.OUTPUTS_DIR / "retrieval_eval_results.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[Retrieval Eval] 纯检索评估完成！结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
