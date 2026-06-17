"""
su-memory SDK — Document Ingestion Pipeline (v3.5.5 P1-1)

自动检测文档格式、分块、批量写入记忆库。

核心类：
- FormatDetector: 自动检测 PDF/MD/JSON/CSV/TXT 格式
- ChunkStrategy: 可插拔分块策略 (FixedSize/Sentence/Semantic/MarkdownHeader)
- DocumentIngestionPipeline: 文档摄入管道主类

使用示例:
    >>> from su_memory.sdk.document_pipeline import DocumentIngestionPipeline
    >>> pipe = DocumentIngestionPipeline(client)
    >>> ids = pipe.ingest_file("/path/to/doc.md")
    >>> ids = pipe.ingest_text("长文本内容...", chunk_size=512)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 可选依赖 ──────────────────────────────────────────────────────────────────

_PYPDF2_AVAILABLE = False
try:
    import PyPDF2  # noqa: F401
    _PYPDF2_AVAILABLE = True
except ImportError:
    logger.debug("PyPDF2 不可用，PDF 解析将返回错误提示")

# ── 格式检测 ──────────────────────────────────────────────────────────────────


@dataclass
class DetectedFormat:
    """检测到的文档格式信息"""
    format: str          # "txt" | "md" | "json" | "csv" | "pdf"
    encoding: str        # "utf-8" | "gbk" | "latin-1"
    mime_type: str       # MIME 类型
    extension: str       # 文件扩展名


class FormatDetector:
    """文档格式自动检测器

    根据文件扩展名和内容特征自动判断文档格式。
    支持：TXT, MD, JSON, CSV, PDF
    """

    EXTENSION_MAP: dict[str, str] = {
        ".txt": "txt",
        ".md": "md",
        ".markdown": "md",
        ".json": "json",
        ".jsonl": "json",
        ".csv": "csv",
        ".pdf": "pdf",
    }

    MIME_MAP: dict[str, str] = {
        "txt": "text/plain",
        "md": "text/markdown",
        "json": "application/json",
        "csv": "text/csv",
        "pdf": "application/pdf",
    }

    @staticmethod
    def detect(path: str | Path) -> DetectedFormat:
        """根据文件路径检测格式

        Args:
            path: 文件路径（str 或 Path）

        Returns:
            DetectedFormat 包含 format/encoding/mime_type/extension

        Raises:
            ValueError: 不支持的格式
            FileNotFoundError: 文件不存在
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        ext = path.suffix.lower()
        fmt = FormatDetector.EXTENSION_MAP.get(ext)
        if not fmt:
            raise ValueError(
                f"不支持的文件格式: {ext}。"
                f"支持的格式: {list(FormatDetector.EXTENSION_MAP.keys())}"
            )

        # 自动检测编码
        encoding = FormatDetector._detect_encoding(path)

        return DetectedFormat(
            format=fmt,
            encoding=encoding,
            mime_type=FormatDetector.MIME_MAP.get(fmt, "application/octet-stream"),
            extension=ext,
        )

    @staticmethod
    def _detect_encoding(path: Path) -> str:
        """自动检测文件编码（优先 UTF-8 → GBK → Latin-1）"""
        for enc in ["utf-8", "gbk", "latin-1"]:
            try:
                with open(path, encoding=enc) as f:
                    f.read(4096)
                return enc
            except (UnicodeDecodeError, UnicodeError):
                continue
        return "utf-8"  # fallback


# ── 分块策略 ──────────────────────────────────────────────────────────────────


@dataclass
class ChunkResult:
    """分块结果"""
    chunks: list[str]
    total_chunks: int
    strategy: str
    avg_chunk_size: int = 0

    def __post_init__(self):
        if self.chunks:
            self.avg_chunk_size = sum(len(c) for c in self.chunks) // len(self.chunks)


class ChunkStrategy:
    """可插拔分块策略基类"""

    name: str = "base"

    def chunk(self, text: str, **kwargs) -> ChunkResult:
        raise NotImplementedError


class FixedSizeChunker(ChunkStrategy):
    """固定大小分块策略

    Args:
        chunk_size: 每块字符数 (默认 512)
        chunk_overlap: 块间重叠字符数 (默认 64)
    """

    name = "fixed_size"

    def chunk(
        self,
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        **kwargs,
    ) -> ChunkResult:
        if not text:
            raise ValueError("文本不能为空")

        chunks: list[str] = []
        pos = 0
        while pos < len(text):
            chunk = text[pos:pos + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            pos += chunk_size - chunk_overlap
            if pos >= len(text):
                break

        return ChunkResult(
            chunks=chunks,
            total_chunks=len(chunks),
            strategy=self.name,
        )


class SentenceChunker(ChunkStrategy):
    """按句子分块策略

    以句号、感叹号、问号、换行为分界，
    合并句子直到达到目标块大小。

    Args:
        target_size: 目标块大小（字符数，默认 512）
        max_size: 最大块大小（字符数，默认 1024）
    """

    name = "sentence"

    def chunk(
        self,
        text: str,
        target_size: int = 512,
        max_size: int = 1024,
        **kwargs,
    ) -> ChunkResult:
        if not text:
            raise ValueError("文本不能为空")

        # 按句子分割
        sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > target_size and current:
                chunks.append(" ".join(current))
                current = [sent]
                current_len = sent_len
            else:
                current.append(sent)
                current_len += sent_len
                if current_len > max_size:
                    chunks.append(" ".join(current))
                    current = []
                    current_len = 0

        if current:
            chunks.append(" ".join(current))

        return ChunkResult(
            chunks=chunks or [text],
            total_chunks=len(chunks) or 1,
            strategy=self.name,
        )


class MarkdownHeaderChunker(ChunkStrategy):
    """Markdown 标题分块策略

    按 Markdown 标题（# ## ###）分段，保持结构完整性。

    Args:
        min_chunk_size: 最小块大小，小于此大小的块会合并（默认 128）
    """

    name = "markdown_header"

    def chunk(self, text: str, min_chunk_size: int = 128, **kwargs) -> ChunkResult:
        if not text:
            raise ValueError("文本不能为空")

        # 按标题分割
        sections = re.split(r'\n(?=#{1,6}\s)', text)
        chunks: list[str] = []
        buffer: str = ""

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(section) < min_chunk_size:
                buffer += "\n\n" + section if buffer else section
            else:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.append(section)

        if buffer:
            if chunks:
                chunks[-1] += "\n\n" + buffer
            else:
                chunks.append(buffer)

        return ChunkResult(
            chunks=chunks or [text],
            total_chunks=len(chunks) or 1,
            strategy=self.name,
        )


def get_chunker(strategy: str = "fixed_size", **kwargs) -> ChunkStrategy:
    """工厂函数：根据策略名创建分块器

    Args:
        strategy: "fixed_size" | "sentence" | "markdown_header"
        **kwargs: 传递给具体分块器的参数

    Returns:
        ChunkStrategy 实例
    """
    strategies: dict[str, type[ChunkStrategy]] = {
        "fixed_size": FixedSizeChunker,
        "sentence": SentenceChunker,
        "markdown_header": MarkdownHeaderChunker,
    }
    if strategy not in strategies:
        raise ValueError(
            f"未知分块策略: {strategy}。可用: {list(strategies.keys())}"
        )
    return strategies[strategy](**kwargs)


# ── 文档解析器 ────────────────────────────────────────────────────────────────


class BaseParser:
    """文档解析器基类"""

    format: str = "base"

    def parse(self, path: Path) -> str:
        raise NotImplementedError


class TextParser(BaseParser):
    """纯文本解析器 (TXT/MD)"""

    format = "txt"

    def parse(self, path: Path) -> str:
        detected = FormatDetector.detect(path)
        with open(path, encoding=detected.encoding) as f:
            return f.read()


class JsonParser(BaseParser):
    """JSON 解析器

    将 JSON 对象/数组递归展开为可读文本行。
    """

    format = "json"

    def __init__(self, flatten_depth: int = 3):
        self._flatten_depth = flatten_depth

    def parse(self, path: Path) -> str:
        detected = FormatDetector.detect(path)
        with open(path, encoding=detected.encoding) as f:
            data = json.load(f)
        return self._json_to_text(data, depth=0)

    def _json_to_text(self, obj: Any, depth: int = 0) -> str:
        """递归将 JSON 转为可读文本"""
        if depth > self._flatten_depth:
            return str(obj)[:200]

        if isinstance(obj, dict):
            lines: list[str] = []
            for key, val in obj.items():
                prefix = "#" * min(depth + 1, 6) + " " if depth < 6 else "- "
                val_text = self._json_to_text(val, depth + 1)
                lines.append(f"{prefix}{key}: {val_text}")
            return "\n".join(lines)
        elif isinstance(obj, list):
            return "\n".join(
                f"- {self._json_to_text(item, depth + 1)}" for item in obj[:50]
            )
        else:
            return str(obj)


class CsvParser(BaseParser):
    """CSV 解析器

    将 CSV 行转为 JSON 风格文本行。
    """

    format = "csv"

    def parse(self, path: Path) -> str:
        import csv

        detected = FormatDetector.detect(path)
        lines: list[str] = []
        with open(path, encoding=detected.encoding) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 1000:  # 限制最大行数
                    lines.append(f"... 已截断，共 {i}+ 行")
                    break
                parts = [f"{k}: {v}" for k, v in row.items()]
                lines.append(" | ".join(parts))
        return "\n".join(lines)


class PdfParser(BaseParser):
    """PDF 解析器 (需 PyPDF2)"""

    format = "pdf"

    def parse(self, path: Path) -> str:
        if not _PYPDF2_AVAILABLE:
            raise ImportError(
                "PDF 解析需要 PyPDF2。请安装: pip install PyPDF2>=3.0"
                " 或 pip install su-memory[documents]"
            )
        import PyPDF2

        lines: list[str] = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    lines.append(text)
                if i >= 200:  # 限制最大页数
                    lines.append(f"\n... 已截断，共 {len(reader.pages)} 页")
                    break
        return "\n\n".join(lines)


PARSER_REGISTRY: dict[str, BaseParser] = {
    "txt": TextParser(),
    "md": TextParser(),
    "json": JsonParser(),
    "csv": CsvParser(),
    "pdf": PdfParser(),
}


# ── 文档摄入管道 ──────────────────────────────────────────────────────────────


@dataclass
class IngestResult:
    """文档摄入结果"""
    source: str                     # 文件路径或标识
    format: str                     # 检测到的格式
    total_chunks: int               # 分块总数
    memory_ids: list[str]           # 写入的记忆 ID 列表
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentIngestionPipeline:
    """文档摄入管道 (v3.5.5 P1-1)

    自动检测格式 → 解析 → 分块 → 批量写入记忆库。

    Args:
        client: SuMemory / SuMemoryLite / SuMemoryLitePro 实例
        default_strategy: 默认分块策略 ("fixed_size" | "sentence" | "markdown_header")
        default_chunk_size: 默认块大小
        default_chunk_overlap: 默认重叠量
        auto_detect_strategy: 是否根据格式自动选择分块策略

    Example:
        >>> pipe = DocumentIngestionPipeline(client)
        >>> results = pipe.ingest_file("/path/to/doc.md")
        >>> print(f"摄入 {results.total_chunks} 块")
    """

    def __init__(
        self,
        client,  # SuMemory / SuMemoryLite / SuMemoryLitePro
        default_strategy: str = "fixed_size",
        default_chunk_size: int = 512,
        default_chunk_overlap: int = 64,
        auto_detect_strategy: bool = True,
    ):
        self._client = client
        self._default_strategy = default_strategy
        self._default_chunk_size = default_chunk_size
        self._default_chunk_overlap = default_chunk_overlap
        self._auto_detect_strategy = auto_detect_strategy
        self._ingest_history: list[IngestResult] = []

    # ── 公开 API ──────────────────────────────────────────────────────────

    def ingest_file(
        self,
        path: str | Path,
        strategy: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """摄入单个文件

        Args:
            path: 文件路径
            strategy: 分块策略 (None = 自动选择)
            chunk_size: 块大小 (None = 默认值)
            chunk_overlap: 块重叠 (None = 默认值)
            metadata: 附加元数据

        Returns:
            IngestResult

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的格式或空文件
        """
        path = Path(path)

        # 输入校验
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if not path.is_file():
            raise ValueError(f"路径不是文件: {path}")

        # 1. 检测格式
        detected = FormatDetector.detect(path)
        logger.info(f"检测到文档格式: {detected.format} ({detected.mime_type})")

        # 2. 解析文档
        parser = PARSER_REGISTRY.get(detected.format)
        if not parser:
            raise ValueError(f"不支持的文档格式: {detected.format}")

        text = parser.parse(path)
        if not text or not text.strip():
            raise ValueError(f"文档内容为空: {path}")

        # 3. 选择分块策略
        chunker = self._select_chunker(detected.format, strategy)

        # 4. 分块
        chunk_result = chunker.chunk(
            text,
            chunk_size=chunk_size or self._default_chunk_size,
            chunk_overlap=chunk_overlap or self._default_chunk_overlap,
        )
        logger.info(
            f"分块完成: {chunk_result.total_chunks} 块 "
            f"(策略={chunk_result.strategy}, 平均={chunk_result.avg_chunk_size}字符)"
        )

        # 5. 批量写入记忆库
        chunk_metadata = {
            "source_file": str(path),
            "format": detected.format,
            "ingest_method": "file",
            **(metadata or {}),
        }
        memory_ids = self._write_chunks(chunk_result.chunks, chunk_metadata)

        # 记录历史
        result = IngestResult(
            source=str(path),
            format=detected.format,
            total_chunks=chunk_result.total_chunks,
            memory_ids=memory_ids,
            metadata=chunk_metadata,
        )
        self._ingest_history.append(result)
        return result

    def ingest_text(
        self,
        text: str,
        strategy: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        metadata: dict[str, Any] | None = None,
        source_label: str = "direct_text",
    ) -> IngestResult:
        """摄入纯文本内容

        Args:
            text: 文本内容
            strategy: 分块策略
            chunk_size: 块大小
            chunk_overlap: 块重叠
            metadata: 附加元数据
            source_label: 来源标识

        Returns:
            IngestResult

        Raises:
            ValueError: 空文本
        """
        if not text or not text.strip():
            raise ValueError("ingest_text() 的 text 不能为空")

        # 推断格式（检查是否 Markdown）
        fmt = "md" if re.search(r'^#{1,6}\s', text, re.MULTILINE) else "txt"

        chunker = self._select_chunker(fmt, strategy or self._default_strategy)

        chunk_result = chunker.chunk(
            text,
            chunk_size=chunk_size or self._default_chunk_size,
            chunk_overlap=chunk_overlap or self._default_chunk_overlap,
        )

        chunk_metadata = {
            "source": source_label,
            "format": fmt,
            "ingest_method": "text",
            **(metadata or {}),
        }
        memory_ids = self._write_chunks(chunk_result.chunks, chunk_metadata)

        result = IngestResult(
            source=source_label,
            format=fmt,
            total_chunks=chunk_result.total_chunks,
            memory_ids=memory_ids,
            metadata=chunk_metadata,
        )
        self._ingest_history.append(result)
        return result

    def ingest_directory(
        self,
        path: str | Path,
        recursive: bool = True,
        strategy: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        metadata: dict[str, Any] | None = None,
        extensions: list[str] | None = None,
    ) -> list[IngestResult]:
        """批量摄入目录下的所有文档

        Args:
            path: 目录路径
            recursive: 是否递归子目录
            strategy: 分块策略
            chunk_size: 块大小
            chunk_overlap: 块重叠
            metadata: 基础元数据（每个文件可覆盖）
            extensions: 限定文件扩展名 (如 ['.md', '.txt'])

        Returns:
            list[IngestResult]

        Raises:
            NotADirectoryError: 路径不是目录
        """
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"路径不是目录: {path}")

        allowed = set(extensions) if extensions else set(FormatDetector.EXTENSION_MAP.keys())

        results: list[IngestResult] = []
        errors: list[tuple[str, str]] = []

        pattern = "**/*" if recursive else "*"
        for file_path in sorted(path.glob(pattern)):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in allowed:
                continue

            try:
                result = self.ingest_file(
                    file_path,
                    strategy=strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    metadata=metadata,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"摄入失败 {file_path}: {e}")
                errors.append((str(file_path), str(e)))

        if errors:
            logger.warning(
                f"目录摄入完成: {len(results)} 成功, {len(errors)} 失败"
            )

        return results

    def get_history(self) -> list[IngestResult]:
        """获取摄入历史"""
        return list(self._ingest_history)

    def clear_history(self) -> None:
        """清空摄入历史"""
        self._ingest_history.clear()

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _select_chunker(
        self,
        fmt: str,
        strategy: str | None,
    ) -> ChunkStrategy:
        """选择分块策略"""
        if strategy:
            return get_chunker(strategy)

        if self._auto_detect_strategy:
            # 根据格式自动选择
            strategy_map = {
                "md": "markdown_header",
                "json": "sentence",
                "csv": "fixed_size",
                "pdf": "sentence",
                "txt": "sentence",
            }
            strategy = strategy_map.get(fmt, "fixed_size")

        return get_chunker(strategy or self._default_strategy)

    def _write_chunks(
        self,
        chunks: list[str],
        base_metadata: dict[str, Any],
    ) -> list[str]:
        """将分块批量写入记忆库"""
        if not chunks:
            return []

        items = [
            {
                "content": chunk,
                "metadata": {
                    **base_metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            }
            for i, chunk in enumerate(chunks)
        ]

        # 尝试使用 add_batch（最优），否则回退到逐条 add
        if hasattr(self._client, "add_batch"):
            return self._client.add_batch(items)
        else:
            return [
                self._client.add(item["content"], item["metadata"])
                for item in items
            ]
