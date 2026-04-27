"""
自动压缩模块 - LZ4压缩

提供高速数据压缩/解压功能，用于减少存储空间占用。

Example:
    >>> from su_memory.storage import AutoCompressor
    >>> compressor = AutoCompressor()
    >>> data = b"Hello, su-memory!"
    >>> compressed = compressor.compress(data)
    >>> original = compressor.decompress(compressed)
    >>> ratio = compressor.get_compression_ratio(data, compressed)
"""

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    lz4 = None

from typing import Optional
import zlib
import struct


class AutoCompressor:
    """
    自动压缩器

    支持LZ4（高速）和zlib（高压缩率）两种压缩算法。
    自动选择最优算法或按配置使用指定算法。

    Attributes:
        algorithm: 压缩算法 ("lz4" | "zlib" | "auto")
        compression_level: 压缩级别 (1-9)

    Example:
        >>> compressor = AutoCompressor(algorithm="lz4")
        >>> compressed = compressor.compress(data)
        >>> original = compressor.decompress(compressed)
    """

    def __init__(self, algorithm: str = "auto", compression_level: int = 6):
        """初始化压缩器

        Args:
            algorithm: 压缩算法
                - "lz4": 高速压缩（需要安装lz4库）
                - "zlib": 高压缩率（Python内置）
                - "auto": 自动选择（优先lz4）
            compression_level: 压缩级别 1-9
        """
        self._algorithm = algorithm
        self._compression_level = compression_level

        # 确定实际使用的算法
        if algorithm == "auto":
            self._actual_algorithm = "lz4" if LZ4_AVAILABLE else "zlib"
        else:
            self._actual_algorithm = algorithm

        # 检查lz4可用性
        if self._actual_algorithm == "lz4" and not LZ4_AVAILABLE:
            import warnings
            warnings.warn("lz4 not available, falling back to zlib")
            self._actual_algorithm = "zlib"

    @property
    def algorithm(self) -> str:
        """当前使用的压缩算法"""
        return self._actual_algorithm

    def compress(self, data: bytes) -> bytes:
        """压缩数据

        Args:
            data: 原始数据

        Returns:
            压缩后的数据

        Raises:
            ValueError: 不支持的压缩算法
        """
        if not data:
            return b""

        if self._actual_algorithm == "lz4":
            return self._compress_lz4(data)
        else:
            return self._compress_zlib(data)

    def decompress(self, data: bytes) -> bytes:
        """解压数据

        Args:
            data: 压缩数据

        Returns:
            原始数据

        Raises:
            ValueError: 不支持的压缩算法或解压失败
        """
        if not data:
            return b""

        if self._actual_algorithm == "lz4":
            return self._decompress_lz4(data)
        else:
            return self._decompress_zlib(data)

    def _compress_lz4(self, data: bytes) -> bytes:
        """LZ4压缩"""
        if not LZ4_AVAILABLE:
            raise ValueError("lz4 not available")

        # 使用块压缩，添加4字节长度前缀
        compressed = lz4.frame.compress(
            data,
            compression_level=self._compression_level
        )
        return compressed

    def _decompress_lz4(self, data: bytes) -> bytes:
        """LZ4解压"""
        if not LZ4_AVAILABLE:
            raise ValueError("lz4 not available")

        return lz4.frame.decompress(data)

    def _compress_zlib(self, data: bytes) -> bytes:
        """zlib压缩"""
        return zlib.compress(data, level=self._compression_level)

    def _decompress_zlib(self, data: bytes) -> bytes:
        """zlib解压"""
        return zlib.decompress(data)

    def compress_stream(self, data: bytes, chunk_size: int = 8192) -> bytes:
        """分块压缩大文件

        Args:
            data: 原始数据
            chunk_size: 块大小

        Returns:
            压缩后的数据
        """
        if len(data) <= chunk_size:
            return self.compress(data)

        # 大数据使用分块压缩
        if self._actual_algorithm == "lz4" and LZ4_AVAILABLE:
            import lz4.block

            result = b""
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                compressed_chunk = lz4.block.compress(
                    chunk,
                    compression=self._compression_level
                )
                # 添加4字节长度前缀
                result += struct.pack(">I", len(compressed_chunk)) + compressed_chunk

            return result
        else:
            return self.compress(data)

    def decompress_stream(self, data: bytes) -> bytes:
        """分块解压

        Args:
            data: 压缩数据

        Returns:
            原始数据
        """
        if self._actual_algorithm == "lz4" and LZ4_AVAILABLE:
            import lz4.block

            result = b""
            offset = 0

            while offset < len(data):
                # 读取长度前缀
                if offset + 4 > len(data):
                    break
                chunk_len = struct.unpack(">I", data[offset:offset + 4])[0]
                offset += 4

                # 读取并解压块
                if offset + chunk_len <= len(data):
                    chunk = lz4.block.decompress(data[offset:offset + chunk_len])
                    result += chunk
                    offset += chunk_len

            return result

        return self.decompress(data)

    def get_compression_ratio(self, original: bytes, compressed: bytes) -> float:
        """计算压缩比

        Args:
            original: 原始数据
            compressed: 压缩后数据

        Returns:
            压缩比（原大小/压缩后大小），越大表示压缩效果越好
        """
        if not compressed or len(compressed) == 0:
            return 0.0

        return len(original) / len(compressed)

    def get_stats(self, original: bytes, compressed: bytes) -> dict:
        """获取压缩统计信息

        Returns:
            压缩统计字典
        """
        ratio = self.get_compression_ratio(original, compressed)
        saved = len(original) - len(compressed)
        saved_pct = (saved / len(original) * 100) if original else 0

        return {
            "original_size": len(original),
            "compressed_size": len(compressed),
            "ratio": ratio,
            "saved_bytes": saved,
            "saved_percent": saved_pct,
            "algorithm": self._actual_algorithm,
        }

    def is_compression_effective(self, data: bytes, threshold: float = 1.5) -> bool:
        """判断压缩是否有效

        Args:
            data: 数据
            threshold: 阈值（压缩比大于此值认为有效）

        Returns:
            是否有效
        """
        if len(data) < 1024:  # 小数据不压缩
            return False

        compressed = self.compress(data)
        return self.get_compression_ratio(data, compressed) >= threshold


class CompressedStorage:
    """压缩存储包装器

    自动压缩/解压存储的数据。

    Example:
        >>> storage = CompressedStorage(SQLiteBackend("data.db"))
        >>> storage.put("key", {"data": "value"})
        >>> data = storage.get("key")
    """

    def __init__(self, backend, compressor: Optional[AutoCompressor] = None):
        """初始化压缩存储

        Args:
            backend: 底层存储后端
            compressor: 压缩器实例
        """
        self._backend = backend
        self._compressor = compressor or AutoCompressor()

    def put(self, key: str, data: dict) -> str:
        """存储压缩数据

        Args:
            key: 键
            data: 数据字典

        Returns:
            键
        """
        import json
        raw_data = json.dumps(data).encode("utf-8")
        compressed = self._compressor.compress(raw_data)

        from su_memory.storage.sqlite_backend import MemoryItem
        memory = MemoryItem(
            id=key,
            content=compressed.hex(),  # 存储为十六进制字符串
            metadata={"compressed": True, "algorithm": self._compressor.algorithm},
            timestamp=0  # 使用metadata存储时间
        )
        return self._backend.add_memory(memory)

    def get(self, key: str) -> Optional[dict]:
        """获取解压数据

        Args:
            key: 键

        Returns:
            数据字典或None
        """
        import json
        memory = self._backend.get_memory(key)
        if not memory:
            return None

        compressed = bytes.fromhex(memory.content)
        raw_data = self._compressor.decompress(compressed)
        return json.loads(raw_data.decode("utf-8"))

    def delete(self, key: str) -> bool:
        """删除数据"""
        return self._backend.delete(key)
