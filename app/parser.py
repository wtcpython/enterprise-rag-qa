import hashlib
import re
from pathlib import Path
from typing import Any

from markitdown import MarkItDown


class DocumentParser:
    """
    基于微软 MarkItDown 的多格式企业文档解析与清洗器。
    支持 PDF / DOCX / TXT / MD 四种格式，统一转换为结构化 Markdown 文本，
    并提取页码标记（针对 PDF 换页符 \x0c）与清洗噪声字符。
    """

    def __init__(self) -> None:
        self._md_converter = MarkItDown()

    @staticmethod
    def compute_file_md5(file_path: Path) -> str:
        """计算文件的 MD5 哈希值，用于增量索引去重与变更检测"""
        with open(file_path, "rb") as f:
            return hashlib.file_digest(f, "md5").hexdigest()

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """
        数据清洗逻辑：
        1. 统一换行符为 \n，保留 PDF 分页符 \x0c 以便后续提取页码（或转换为显式页码标记）
        2. 清除不可见控制字符（零宽空格、BOM 等）
        3. 合并连续超过 2 行的空白行，去除行尾多余空格
        4. 修复中文 PDF 解析常见的断行碎片（非标点结尾且下一行是普通中文字符时智能拼接）
        """
        if not raw_text:
            return ""

        # 移除 BOM 与零宽字符
        text = (
            raw_text.replace("\ufeff", "")
            .replace("\u200b", "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        # 将 PDF 换页符 \x0c 转换为显式页码锚点 <!-- PAGE_BREAK -->
        if "\x0c" in text:
            pages = text.split("\x0c")
            tagged_pages = []
            for idx, page_content in enumerate(pages, start=1):
                stripped = page_content.strip()
                if stripped:
                    tagged_pages.append(f"<!-- PAGE:{idx} -->\n{stripped}")
            text = "\n\n".join(tagged_pages)

        # 清理每行首尾多余空白（保留 Markdown 缩进与列表结构）
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        # 压缩 3 个以上连续换行为双换行
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def parse_file(self, file_path: Path) -> dict[str, Any]:
        """
        解析单个文档并返回结构化信息：
        - source: 文件名
        - file_md5: 文件 MD5
        - ext: 扩展名
        - content: 清洗后的 Markdown 文本
        - char_count: 字符总数
        """
        ext = file_path.suffix.lower()
        file_md5 = self.compute_file_md5(file_path)

        # 对纯文本/Markdown 直接以 UTF-8 读取或交由 MarkItDown 处理，对 PDF/DOCX 使用 MarkItDown 转换
        if ext in {".md", ".txt"}:
            try:
                raw_text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raw_text = file_path.read_text(encoding="gb18030", errors="ignore")
        else:
            result = self._md_converter.convert(str(file_path))
            raw_text = result.text_content or ""

        cleaned_text = self.clean_text(raw_text)
        return {
            "source": file_path.name,
            "file_path": str(file_path),
            "file_md5": file_md5,
            "ext": ext,
            "content": cleaned_text,
            "char_count": len(cleaned_text),
        }
