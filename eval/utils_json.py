import json
import re
from typing import Any


def clean_json_text(text: str, prefer_array: bool = False) -> str:
    """
    清理并提取 LLM 输出中的 JSON 字符串：
    1. 剥离 Markdown 代码块包裹；
    2. 当 prefer_array=True 或文本中 `[` 出现在 `{` 之前时，优先提取最外层 `[...]` 数组；
    3. 否则提取最外层 `{...}` 对象。
    """
    if not text:
        return ""

    cleaned = text.strip()
    # 剥离 ```json ... ``` 或 ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")

    has_array = (
        first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket
    )
    has_object = first_brace != -1 and last_brace != -1 and last_brace > first_brace

    if prefer_array and has_array:
        return cleaned[first_bracket : last_bracket + 1]

    if has_array and has_object:
        if first_bracket < first_brace:
            return cleaned[first_bracket : last_bracket + 1]
        return cleaned[first_brace : last_brace + 1]
    elif has_array:
        return cleaned[first_bracket : last_bracket + 1]
    elif has_object:
        return cleaned[first_brace : last_brace + 1]

    return cleaned


def parse_json_safely(text: str, prefer_array: bool = False) -> Any:
    """
    带多层容错与正则兜底的 JSON 解析器：
    1. 尝试直接解析 clean_json_text 的输出；
    2. 尝试修复末尾多余逗号等常见 LLM 坏 JSON 格式；
    3. 若仍失败，返回 None 由调用方触发专门的正则字段提取兜底。
    """
    candidate = clean_json_text(text, prefer_array=prefer_array)
    if not candidate:
        return None

    for attempt in [
        candidate,
        re.sub(r",\s*([\]}])", r"\1", candidate),  # 去除尾随逗号
    ]:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None


def parse_rerank_scores_fallback(
    text: str, num_candidates: int
) -> list[dict[str, Any]]:
    """
    当 LLM Rerank 输出的 JSON 数组解析失败时的正则兜底提取器，
    支持匹配形如 {"index": 0, "score": 8.5} 或 "chunk_index": 1, "relevance": 9 等模式。
    """
    parsed = parse_json_safely(text, prefer_array=True)
    results: list[dict[str, Any]] = []

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and ("index" in item or "id" in item):
                idx = item.get("index", item.get("id"))
                score = item.get("score", item.get("relevance", 0.0))
                try:
                    results.append({"index": int(idx), "score": float(score)})
                except ValueError, TypeError:
                    continue
        if results:
            return results
    elif isinstance(parsed, dict):
        # 兼容形如 {"rankings": [...]} 或 {"results": [...]}
        for key in ("rankings", "results", "scores", "items", "data"):
            if isinstance(parsed.get(key), list):
                return parse_rerank_scores_fallback(
                    json.dumps(parsed[key]), num_candidates
                )

    # 正则兜底：查找 index/id 与 score 对
    pattern = re.compile(
        r'"?(?:index|id)"?\s*:\s*(\d+)[^0-9\-]*?"?(?:score|relevance)"?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        idx = int(match.group(1))
        score = float(match.group(2))
        if 0 <= idx < num_candidates:
            results.append({"index": idx, "score": score})

    return results
