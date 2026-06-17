"""
P0: DocumentPipeline 单元测试 (v3.5.5 P1-1)
=============================================
覆盖: FormatDetector, FixedSizeChunker, SentenceChunker,
      MarkdownHeaderChunker, TextParser, JsonParser, CsvParser,
      DocumentIngestionPipeline

运行: pytest tests/test_document_pipeline.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from su_memory.sdk.document_pipeline import (
    ChunkResult,
    CsvParser,
    DetectedFormat,
    DocumentIngestionPipeline,
    FixedSizeChunker,
    FormatDetector,
    IngestResult,
    JsonParser,
    MarkdownHeaderChunker,
    SentenceChunker,
    TextParser,
    get_chunker,
)
from su_memory.sdk.lite_pro import SuMemoryLitePro


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_text():
    """标准测试文本"""
    return (
        "人工智能技术在2024年取得了重大突破。深度学习模型在自然语言处理、"
        "计算机视觉、语音识别等领域达到了新的高度。大语言模型如GPT-4和Claude "
        "展现了强大的推理能力。然而，数据隐私、算法偏见和能源消耗等问题仍然存在。"
    )


@pytest.fixture
def long_text():
    """长文本 (用于分块测试)"""
    return "这是一段测试文本。" * 200


@pytest.fixture
def md_text():
    """Markdown 文本"""
    return """# 第一章 概述
这是概述内容。

## 1.1 背景
项目背景介绍。

## 1.2 目标
项目目标说明。

# 第二章 技术方案
技术方案详细描述。

### 2.1.1 架构设计
系统架构设计内容。

### 2.1.2 数据流
数据流描述内容。
"""


@pytest.fixture
def temp_dir():
    """临时目录"""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def memory_client():
    """测试用记忆客户端"""
    return SuMemoryLitePro(max_memories=200)


# ============================================================
# FormatDetector 测试
# ============================================================

class TestFormatDetector:
    """格式检测器"""

    def test_extension_map_contains_all_formats(self):
        """EXTENSION_MAP 包含所有必需格式"""
        ext_map = FormatDetector.EXTENSION_MAP
        assert ".md" in ext_map, "应支持 .md"
        assert ".txt" in ext_map, "应支持 .txt"
        assert ".json" in ext_map, "应支持 .json"
        assert ".csv" in ext_map, "应支持 .csv"
        assert ".pdf" in ext_map, "应支持 .pdf"

    def test_extension_map_md_formats(self):
        """Markdown 多种扩展名"""
        ext_map = FormatDetector.EXTENSION_MAP
        assert ext_map[".md"] == "md"
        assert ext_map[".markdown"] == "md"

    def test_mime_map(self):
        """MIME 类型映射"""
        mime_map = FormatDetector.MIME_MAP
        assert mime_map["txt"] == "text/plain"
        assert mime_map["md"] == "text/markdown"
        assert mime_map["json"] == "application/json"
        assert mime_map["csv"] == "text/csv"
        assert mime_map["pdf"] == "application/pdf"

    def test_detect_txt_file(self, temp_dir):
        """检测 .txt 文件"""
        f = temp_dir / "test.txt"
        f.write_text("Hello World", encoding="utf-8")
        result = FormatDetector.detect(f)
        assert result.format == "txt"
        assert result.encoding == "utf-8"
        assert result.mime_type == "text/plain"

    def test_detect_md_file(self, temp_dir):
        """检测 .md 文件"""
        f = temp_dir / "test.md"
        f.write_text("# Title\ncontent", encoding="utf-8")
        result = FormatDetector.detect(f)
        assert result.format == "md"
        assert result.mime_type == "text/markdown"

    def test_detect_json_file(self, temp_dir):
        """检测 .json 文件"""
        f = temp_dir / "data.json"
        f.write_text('{"key": "value"}', encoding="utf-8")
        result = FormatDetector.detect(f)
        assert result.format == "json"
        assert result.mime_type == "application/json"

    def test_detect_csv_file(self, temp_dir):
        """检测 .csv 文件"""
        f = temp_dir / "data.csv"
        f.write_text("a,b,c\n1,2,3", encoding="utf-8")
        result = FormatDetector.detect(f)
        assert result.format == "csv"
        assert result.mime_type == "text/csv"

    def test_detect_nonexistent_file_raises(self, temp_dir):
        """不存在的文件抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            FormatDetector.detect(temp_dir / "nonexistent.txt")

    def test_detect_unsupported_extension_raises(self, temp_dir):
        """不支持的扩展名抛出 ValueError"""
        f = temp_dir / "test.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            FormatDetector.detect(f)


# ============================================================
# ChunkStrategy 测试
# ============================================================

class TestFixedSizeChunker:
    """固定大小分块器"""

    def test_basic_chunking(self, sample_text):
        """基本分块"""
        chunker = FixedSizeChunker()
        result = chunker.chunk(sample_text, chunk_size=50, chunk_overlap=10)
        assert isinstance(result, ChunkResult)
        assert result.total_chunks > 0
        assert result.strategy == "fixed_size"

    def test_result_has_chunks_attribute(self, sample_text):
        """ChunkResult 有 .chunks 属性"""
        chunker = FixedSizeChunker()
        result = chunker.chunk(sample_text, chunk_size=50)
        assert hasattr(result, "chunks")
        assert len(result.chunks) == result.total_chunks

    def test_chunks_content_preserved(self, long_text):
        """分块后内容不丢失"""
        chunker = FixedSizeChunker()
        result = chunker.chunk(long_text, chunk_size=200, chunk_overlap=30)
        combined = "".join(result.chunks)
        original_stripped = long_text.replace(" ", "")
        combined_stripped = combined.replace(" ", "")
        # 由于 overlap，combined 可能比 original 长
        assert len(combined_stripped) >= len(original_stripped) * 0.8

    def test_empty_text_raises(self):
        """空文本抛出 ValueError"""
        chunker = FixedSizeChunker()
        with pytest.raises(ValueError, match="文本不能为空"):
            chunker.chunk("")

    def test_small_text_single_chunk(self):
        """小文本单块"""
        chunker = FixedSizeChunker()
        result = chunker.chunk("小文本", chunk_size=512)
        assert result.total_chunks == 1

    def test_avg_chunk_size_computed(self, long_text):
        """平均块大小被计算"""
        chunker = FixedSizeChunker()
        result = chunker.chunk(long_text, chunk_size=200)
        assert result.avg_chunk_size > 0


class TestSentenceChunker:
    """句子分块器"""

    def test_split_by_sentence(self):
        """按句子分隔"""
        chunker = SentenceChunker()
        text = "第一句话。第二句话！第三句话？第四句话。"
        result = chunker.chunk(text, target_size=200)
        assert result.total_chunks >= 1

    def test_empty_text_raises(self):
        """空文本抛出 ValueError"""
        chunker = SentenceChunker()
        with pytest.raises(ValueError, match="文本不能为空"):
            chunker.chunk("")

    def test_returns_chunk_result(self, sample_text):
        """返回 ChunkResult"""
        chunker = SentenceChunker()
        result = chunker.chunk(sample_text, target_size=100)
        assert isinstance(result, ChunkResult)
        assert result.strategy == "sentence"


class TestMarkdownHeaderChunker:
    """Markdown 标题分块器"""

    def test_split_by_headers(self, md_text):
        """按标题分隔"""
        chunker = MarkdownHeaderChunker()
        result = chunker.chunk(md_text)
        assert result.total_chunks >= 2  # 至少分两个章节

    def test_empty_text_raises(self):
        """空文本抛出 ValueError"""
        chunker = MarkdownHeaderChunker()
        with pytest.raises(ValueError, match="文本不能为空"):
            chunker.chunk("")

    def test_small_sections_merged(self):
        """小段落合并"""
        chunker = MarkdownHeaderChunker()
        text = "# H1\n短。\n# H2\n短。\n# H3\n这是一段足够长的内容。" * 5
        result = chunker.chunk(text, min_chunk_size=128)
        assert result.total_chunks >= 1

    def test_returns_chunk_result(self, md_text):
        """返回 ChunkResult"""
        chunker = MarkdownHeaderChunker()
        result = chunker.chunk(md_text)
        assert isinstance(result, ChunkResult)
        assert result.strategy == "markdown_header"


class TestGetChunker:
    """分块器工厂"""

    def test_get_fixed_size(self):
        chunker = get_chunker("fixed_size")
        assert isinstance(chunker, FixedSizeChunker)

    def test_get_sentence(self):
        chunker = get_chunker("sentence")
        assert isinstance(chunker, SentenceChunker)

    def test_get_markdown_header(self):
        chunker = get_chunker("markdown_header")
        assert isinstance(chunker, MarkdownHeaderChunker)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="未知分块策略"):
            get_chunker("invalid_strategy")


# ============================================================
# Parser 测试
# ============================================================

class TestTextParser:
    """文本解析器"""

    def test_parse_txt(self, temp_dir):
        f = temp_dir / "sample.txt"
        f.write_text("Hello World\n你好世界", encoding="utf-8")
        parser = TextParser()
        text = parser.parse(f)
        assert "Hello World" in text
        assert "你好世界" in text

    def test_parse_md(self, temp_dir, md_text):
        f = temp_dir / "sample.md"
        f.write_text(md_text, encoding="utf-8")
        parser = TextParser()
        text = parser.parse(f)
        assert "第一章" in text


class TestJsonParser:
    """JSON 解析器"""

    def test_parse_flat_json(self, temp_dir):
        f = temp_dir / "data.json"
        data = {"name": "test", "version": "1.0", "count": 42}
        f.write_text(json.dumps(data), encoding="utf-8")
        parser = JsonParser()
        text = parser.parse(f)
        assert "name" in text
        assert "test" in text

    def test_parse_nested_json(self, temp_dir):
        f = temp_dir / "nested.json"
        data = {"user": {"name": "Alice", "skills": ["Python", "Go"]}}
        f.write_text(json.dumps(data), encoding="utf-8")
        parser = JsonParser(flatten_depth=3)
        text = parser.parse(f)
        assert "Alice" in text

    def test_parse_list(self, temp_dir):
        f = temp_dir / "list.json"
        data = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
        f.write_text(json.dumps(data), encoding="utf-8")
        parser = JsonParser()
        text = parser.parse(f)
        assert "name" in text


class TestCsvParser:
    """CSV 解析器"""

    def test_parse_simple_csv(self, temp_dir):
        f = temp_dir / "data.csv"
        f.write_text("name,age,city\nAlice,30,Beijing\nBob,25,Shanghai", encoding="utf-8")
        parser = CsvParser()
        text = parser.parse(f)
        assert "Alice" in text
        assert "Beijing" in text


# ============================================================
# DocumentIngestionPipeline 测试
# ============================================================

class TestIngestionPipeline:
    """文档摄入管道"""

    @pytest.fixture
    def pipeline(self, memory_client):
        return DocumentIngestionPipeline(memory_client)

    def test_ingest_text_basic(self, pipeline, sample_text):
        """基本文本摄入"""
        result = pipeline.ingest_text(sample_text)
        assert isinstance(result, IngestResult)
        assert result.total_chunks > 0
        assert len(result.memory_ids) == result.total_chunks
        assert result.format == "txt"

    def test_ingest_text_markdown_detection(self, pipeline):
        """Markdown 格式检测"""
        text = "# Title\nContent here\n## Section\nMore content."
        result = pipeline.ingest_text(text)
        assert result.format == "md"

    def test_ingest_text_empty_raises(self, pipeline):
        """空文本抛出 ValueError"""
        with pytest.raises(ValueError, match="不能为空"):
            pipeline.ingest_text("")

    def test_ingest_text_blank_raises(self, pipeline):
        """空白文本抛出 ValueError"""
        with pytest.raises(ValueError, match="不能为空"):
            pipeline.ingest_text("   \n  ")

    def test_ingest_text_with_metadata(self, pipeline, sample_text):
        """自定义元数据"""
        result = pipeline.ingest_text(sample_text, metadata={"author": "test"})
        # Metadata 应包含 source, format, ingest_method, 和自定义字段
        assert result.metadata.get("ingest_method") == "text"

    def test_ingest_text_with_strategy(self, pipeline, sample_text):
        """指定分块策略"""
        result = pipeline.ingest_text(sample_text, strategy="sentence")
        assert result.total_chunks >= 1

    def test_ingest_file_txt(self, pipeline, temp_dir):
        """摄入 .txt 文件"""
        f = temp_dir / "doc.txt"
        f.write_text("文件内容用于测试。" * 50, encoding="utf-8")
        result = pipeline.ingest_file(f)
        assert result.format == "txt"
        assert result.total_chunks > 0
        assert len(result.memory_ids) > 0

    def test_ingest_file_md(self, pipeline, temp_dir, md_text):
        """摄入 .md 文件"""
        f = temp_dir / "doc.md"
        f.write_text(md_text, encoding="utf-8")
        result = pipeline.ingest_file(f)
        assert result.format == "md"
        assert result.total_chunks >= 2

    def test_ingest_file_nonexistent_raises(self, pipeline, temp_dir):
        """不存在文件抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            pipeline.ingest_file(temp_dir / "ghost.txt")

    def test_ingest_file_not_a_file_raises(self, pipeline, temp_dir):
        """目录抛出 ValueError"""
        with pytest.raises(ValueError, match="路径不是文件"):
            pipeline.ingest_file(temp_dir)

    def test_ingest_file_empty_raises(self, pipeline, temp_dir):
        """空文件抛出 ValueError"""
        f = temp_dir / "empty.txt"
        f.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="文档内容为空"):
            pipeline.ingest_file(f)

    def test_history_tracking(self, pipeline, sample_text):
        """摄入历史追踪"""
        pipeline.ingest_text(sample_text)
        pipeline.ingest_text("另一段文本" * 20)
        history = pipeline.get_history()
        assert len(history) == 2

    def test_clear_history(self, pipeline, sample_text):
        """清空历史"""
        pipeline.ingest_text(sample_text)
        pipeline.clear_history()
        assert len(pipeline.get_history()) == 0

    def test_ingest_directory(self, pipeline, temp_dir):
        """批量摄入目录"""
        (temp_dir / "a.txt").write_text("文件A内容" * 30, encoding="utf-8")
        (temp_dir / "b.txt").write_text("文件B内容" * 30, encoding="utf-8")
        results = pipeline.ingest_directory(temp_dir)
        assert len(results) == 2

    def test_ingest_directory_filter_extension(self, pipeline, temp_dir):
        """目录摄入扩展名过滤"""
        (temp_dir / "a.txt").write_text("文件A" * 30, encoding="utf-8")
        (temp_dir / "b.md").write_text("# 文件B" * 20, encoding="utf-8")
        results = pipeline.ingest_directory(temp_dir, extensions=[".txt"])
        assert len(results) == 1

    def test_writes_to_client(self, pipeline, memory_client, sample_text):
        """验证内容已写入记忆客户端"""
        result = pipeline.ingest_text(sample_text)
        # 可以在客户端查询到写入的内容
        mem = memory_client.query("人工智能", top_k=3)
        assert len(mem) > 0


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
