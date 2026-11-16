import contextlib
import json
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from app.chunker import SemanticOverlapChunker
from app.config import settings
from app.parser import DocumentParser


class KnowledgeBaseIndexer:
    """
    企业知识库向量索引管理器：
    1. 加载 `models/` 目录下的本地向量模型，免去在线调用 HuggingFace 的过程；
    2. 基于 ChromaDB 实现本地持久化存储（余弦距离 hnsw:space = cosine）；
    3. 支持基于文件 MD5 的增量索引（Incremental）与全量重建索引（Rebuild）；
    4. 维护持久化索引状态与操作日志（data/index_state.json）。
    """

    _embedding_model: SentenceTransformer | None = None
    _active_model_name: str = ""

    def __init__(self) -> None:
        self.parser = DocumentParser()
        self.chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))

    @classmethod
    def get_embedding_model(cls) -> tuple[SentenceTransformer, str]:
        """单例加载本地 Embedding 模型，返回 (model, active_model_name)"""
        if cls._embedding_model is not None:
            return cls._embedding_model, cls._active_model_name

        model_name = settings.EMBEDDING_MODEL
        model_path = settings.embedding_model_path
        is_local = Path(model_path).exists()
        print(f"[Indexer] 正在加载 Embedding 模型: {model_name} ({model_path}) ...")
        cls._embedding_model = SentenceTransformer(
            model_path,
            local_files_only=is_local,
        )
        cls._active_model_name = model_name
        print(f"[Indexer] 成功加载 Embedding 模型: {cls._active_model_name}")
        return cls._embedding_model, cls._active_model_name

    def embed_texts(
        self, texts: list[str], is_query: bool = False
    ) -> list[Sequence[float]]:
        """将文本列表编码为归一化向量"""
        model, _ = self.get_embedding_model()
        if is_query:
            with contextlib.suppress(Exception):
                vecs = model.encode(
                    texts,
                    prompt_name="query",
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return vecs.tolist()
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def load_index_state(self) -> dict[str, Any]:
        """读取持久化索引状态与日志"""
        if settings.INDEX_STATE_FILE.exists():
            with contextlib.suppress(Exception):
                return json.loads(settings.INDEX_STATE_FILE.read_text(encoding="utf-8"))
        return {
            "collection_name": "enterprise_kb",
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunk_size": settings.DEFAULT_CHUNK_SIZE,
            "overlap": settings.DEFAULT_CHUNK_OVERLAP,
            "total_chunks": 0,
            "files": {},
            "logs": [],
        }

    def save_index_state(self, state: dict[str, Any]) -> None:
        settings.INDEX_STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_collection(
        self, collection_name: str = "enterprise_kb"
    ) -> chromadb.Collection:
        return self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def list_documents(self) -> list[dict[str, Any]]:
        """列出知识库目录中的所有合规文档及其索引状态"""
        state = self.load_index_state()
        indexed_files = state.get("files", {})
        docs_info: list[dict[str, Any]] = []

        if not settings.DOCS_DIR.exists():
            return docs_info

        for file_path in sorted(settings.DOCS_DIR.iterdir()):
            if (
                not file_path.is_file()
                or file_path.suffix.lower() not in settings.ALLOWED_EXTENSIONS
            ):
                continue
            md5 = self.parser.compute_file_md5(file_path)
            stat = file_path.stat()
            indexed_meta = indexed_files.get(file_path.name)
            is_indexed = bool(indexed_meta and indexed_meta.get("file_md5") == md5)
            docs_info.append(
                {
                    "filename": file_path.name,
                    "extension": file_path.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "file_md5": md5,
                    "is_indexed": is_indexed,
                    "chunk_count": indexed_meta.get("chunk_count", 0)
                    if is_indexed
                    else 0,
                    "indexed_at": indexed_meta.get("indexed_at")
                    if is_indexed
                    else None,
                }
            )
        return docs_info

    def build_index(
        self,
        mode: str = "incremental",
        chunk_size: int = 400,
        overlap: int = 80,
        collection_name: str = "enterprise_kb",
    ) -> dict[str, Any]:
        """
        执行建索引/重建索引/增量索引：
        - mode == 'rebuild': 删除旧集合，对全部文档重新解析、切分、向量化入库
        - mode == 'incremental': 基于文件 MD5 判断，仅处理新增或修改过的文件，跳过未变更文件
        """
        t0 = time.perf_counter()
        _, active_model_name = self.get_embedding_model()
        state = self.load_index_state()
        collection = self.get_collection(collection_name)

        params_changed = (
            state.get("chunk_size") != chunk_size
            or state.get("overlap") != overlap
            or state.get("embedding_model") != active_model_name
            or state.get("collection_name") != collection_name
        )
        collection_empty = collection.count() == 0
        effective_mode = (
            "rebuild"
            if (mode == "rebuild" or params_changed or collection_empty)
            else "incremental"
        )

        if effective_mode == "rebuild":
            with contextlib.suppress(Exception):
                self.chroma_client.delete_collection(name=collection_name)
            collection = self.get_collection(collection_name)
            indexed_files: dict[str, Any] = {}
        else:
            indexed_files = state.get("files", {})

        chunker = SemanticOverlapChunker(chunk_size=chunk_size, overlap=overlap)

        doc_files = [
            p
            for p in sorted(settings.DOCS_DIR.iterdir())
            if p.is_file() and p.suffix.lower() in settings.ALLOWED_EXTENSIONS
        ]

        added_or_updated: list[str] = []
        skipped: list[str] = []
        removed: list[str] = []
        new_chunks_added = 0

        current_filenames = {p.name for p in doc_files}
        for old_name in list(indexed_files.keys()):
            if old_name not in current_filenames:
                collection.delete(where={"source": old_name})
                del indexed_files[old_name]
                removed.append(old_name)

        for file_path in doc_files:
            file_md5 = self.parser.compute_file_md5(file_path)
            existing = indexed_files.get(file_path.name)

            if (
                effective_mode == "incremental"
                and existing
                and existing.get("file_md5") == file_md5
            ):
                skipped.append(file_path.name)
                continue

            if existing:
                collection.delete(where={"source": file_path.name})

            parsed_doc = self.parser.parse_file(file_path)
            chunks = chunker.chunk_document(parsed_doc)
            if not chunks:
                continue

            ids = [c["chunk_id"] for c in chunks]
            texts = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            embeddings = self.embed_texts(texts, is_query=False)

            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            indexed_files[file_path.name] = {
                "file_md5": file_md5,
                "char_count": parsed_doc["char_count"],
                "chunk_count": len(chunks),
                "indexed_at": datetime.now().isoformat(timespec="seconds"),
            }
            added_or_updated.append(file_path.name)
            new_chunks_added += len(chunks)

        total_chunks = collection.count()
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        log_entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "requested_mode": mode,
            "effective_mode": effective_mode,
            "collection_name": collection_name,
            "embedding_model": active_model_name,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "added_or_updated": added_or_updated,
            "skipped": skipped,
            "removed": removed,
            "new_chunks_added": new_chunks_added,
            "total_chunks": total_chunks,
            "latency_ms": elapsed_ms,
        }

        logs = state.get("logs", [])
        logs.append(log_entry)
        state.update(
            {
                "collection_name": collection_name,
                "embedding_model": active_model_name,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "total_chunks": total_chunks,
                "last_indexed_at": log_entry["timestamp"],
                "files": indexed_files,
                "logs": logs[-50:],
            }
        )
        self.save_index_state(state)
        return log_entry
