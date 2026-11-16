import re
from pathlib import Path
from typing import Any


class SemanticOverlapChunker:
    """
    支持 chunk_size 与 overlap 可配置的语义感知切分器：
    1. 自动识别章节标题作为 `section` 元数据；
    2. 自动识别 PDF 页码标记 `<!-- PAGE:N -->` 作为 `page` 元数据；
    3. 优先按语义单元切分，再通过滑动窗口在相邻块之间注入长度为 `overlap` 的前序上下文重叠；
    4. 生成完整元数据：source, chunk_id, chunk_index, section, page, file_md5, char_len, overlap_len。
    """

    HEADING_PATTERN = re.compile(
        r"^(?:#{1,4}\s+.+|[一二三四五六七八九十]+、.+|第[一二三四五六七八九十百0-9]+[章节条]\s+.+|【.+】|Q\d+[\.：:].+)$"
    )
    PAGE_TAG_PATTERN = re.compile(r"^<!--\s*PAGE:(\d+)\s*-->$")

    def __init__(self, chunk_size: int = 400, overlap: int = 80) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_long_unit(self, text: str, max_len: int) -> list[str]:
        """当单个自然段超过 max_len 时，按句子边界（。！？；\n）二次切分"""
        if len(text) <= max_len:
            return [text]

        sentences = re.split(r"(?<=[。！？；!?;])\s*|\n", text)
        sub_units: list[str] = []
        current = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) > max_len:
                if current:
                    sub_units.append(current)
                    current = ""
                for i in range(0, len(sent), max_len):
                    sub_units.append(sent[i : i + max_len])
            elif len(current) + len(sent) + 1 <= max_len:
                current = f"{current}{sent}" if not current else f"{current} {sent}"
            else:
                if current:
                    sub_units.append(current)
                current = sent
        if current:
            sub_units.append(current)
        return sub_units

    def chunk_document(self, parsed_doc: dict[str, Any]) -> list[dict[str, Any]]:
        """
        对解析后的文档执行带真实 overlap 的切分，返回 chunk 字典列表。
        """
        source: str = parsed_doc["source"]
        file_md5: str = parsed_doc["file_md5"]
        content: str = parsed_doc["content"]
        doc_stem = Path(source).stem

        raw_paragraphs = re.split(r"\n\s*\n", content)
        units: list[dict[str, Any]] = []
        current_section = "总则/概述"
        current_page = 1

        for para in raw_paragraphs:
            lines = [ln.strip() for ln in para.strip().split("\n") if ln.strip()]
            if not lines:
                continue

            para_lines: list[str] = []
            for line in lines:
                page_match = self.PAGE_TAG_PATTERN.match(line)
                if page_match:
                    current_page = int(page_match.group(1))
                    continue
                if self.HEADING_PATTERN.match(line) and len(line) <= 60:
                    current_section = re.sub(r"^#{1,4}\s*", "", line).strip()
                para_lines.append(line)

            if not para_lines:
                continue

            merged_para = "\n".join(para_lines)
            for sub in self._split_long_unit(merged_para, self.chunk_size):
                units.append(
                    {
                        "text": sub,
                        "section": current_section,
                        "page": current_page,
                    }
                )

        if not units:
            return []

        chunks: list[dict[str, Any]] = []
        idx = 0
        chunk_index = 0
        prev_body_text = ""

        while idx < len(units):
            chunk_section = units[idx]["section"]
            chunk_page = units[idx]["page"]

            # 组装当前块的新增语义内容（不超过 chunk_size）
            body_buffer = ""
            consumed = 0
            while idx < len(units):
                candidate = units[idx]["text"]
                sep = "\n" if body_buffer else ""
                if (
                    consumed == 0
                    or len(body_buffer) + len(sep) + len(candidate) <= self.chunk_size
                ):
                    body_buffer = (
                        f"{body_buffer}{sep}{candidate}" if body_buffer else candidate
                    )
                    idx += 1
                    consumed += 1
                else:
                    break

            # 真正实现 overlap：在当前块前端拼接前一个块尾部的 overlap 个字符
            if self.overlap > 0 and prev_body_text:
                overlap_prefix = prev_body_text[-self.overlap :].strip()
                chunk_text = f"{overlap_prefix}\n{body_buffer}".strip()
                actual_overlap_len = len(overlap_prefix)
            else:
                chunk_text = body_buffer.strip()
                actual_overlap_len = 0

            chunk_id = f"{doc_stem}_{file_md5[:6]}_{chunk_index:03d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "section": chunk_section,
                        "page": chunk_page,
                        "file_md5": file_md5,
                        "char_len": len(chunk_text),
                        "overlap_len": actual_overlap_len,
                    },
                }
            )

            prev_body_text = body_buffer
            chunk_index += 1

        return chunks
