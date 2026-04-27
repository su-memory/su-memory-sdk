"""
数据导出模块 - JSON/CSV导出

提供数据导出和导入功能，支持JSON和CSV格式。

Example:
    >>> from su_memory.storage import DataExporter, SQLiteBackend
    >>> backend = SQLiteBackend("memories.db")
    >>> exporter = DataExporter(backend)
    >>> exporter.to_json("export.json")
    >>> exporter.to_csv("export.csv")
    >>> exporter.from_json("import.json")
"""

import json
import csv
from typing import List, Dict, Optional, Any
import sqlite3
import uuid
import time

from su_memory.storage.sqlite_backend import MemoryItem


class DataExporter:
    """
    数据导出器

    支持JSON和CSV格式的导出和导入。

    Attributes:
        backend: SQLiteBackend实例

    Example:
        >>> exporter = DataExporter("memories.db")
        >>> exporter.to_json("export.json")
        >>> exporter.to_csv("export.csv")
        >>> exporter.from_json("import.json")
    """

    def __init__(self, db_path: str = "su_memory.db"):
        """初始化导出器

        Args:
            db_path: 数据库路径或SQLiteBackend实例
        """
        if isinstance(db_path, str):
            self._db_path = db_path
            self._backend = None
        else:
            self._backend = db_path
            self._db_path = db_path._db_path

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接（P0-4修复：添加超时设置）"""
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout = 10000")  # 10秒忙等待
        return conn

    def _get_readonly_connection(self) -> sqlite3.Connection:
        """获取只读连接用于导出（P0-4修复）"""
        conn = sqlite3.connect(self._db_path, timeout=5.0, isolation_level='DEFERRED')
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def to_json(self, path: str, include_metadata: bool = True) -> int:
        """导出为JSON格式

        Args:
            path: 输出文件路径
            include_metadata: 是否包含元数据

        Returns:
            导出的记录数
        """
        conn = self._get_readonly_connection()  # P0-4修复：使用只读连接

        if include_metadata:
            cursor = conn.execute("""
                SELECT id, content, metadata, embedding, timestamp, causal_links
                FROM memories
                ORDER BY timestamp DESC
            """)
        else:
            cursor = conn.execute("""
                SELECT id, content, timestamp
                FROM memories
                ORDER BY timestamp DESC
            """)

        rows = cursor.fetchall()
        conn.close()

        data = []
        for row in rows:
            if include_metadata and len(row) >= 6:
                # 处理embedding（二进制转列表）
                embedding = None
                if row[3]:
                    import numpy as np
                    embedding = np.frombuffer(row[3], dtype=np.float32).tolist()

                data.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "embedding": embedding,
                    "timestamp": row[4],
                    "causal_links": json.loads(row[5]) if row[5] else [],
                })
            else:
                data.append({
                    "id": row[0],
                    "content": row[1],
                    "timestamp": row[2],
                })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return len(data)

    def to_csv(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        max_content_length: int = 10000,
    ) -> int:
        """导出为CSV格式

        Args:
            path: 输出文件路径
            columns: 要导出的列名（默认全部）
            max_content_length: 内容最大长度（截断）

        Returns:
            导出的记录数
        """
        default_columns = ["id", "content", "metadata", "timestamp"]
        if columns is None:
            columns = default_columns

        conn = self._get_readonly_connection()  # P0-4修复：使用只读连接
        cursor = conn.execute("""
            SELECT id, content, metadata, timestamp
            FROM memories
            ORDER BY timestamp DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(columns)

            for row in rows:
                row_data = {
                    "id": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "timestamp": row[3],
                }

                # 处理内容长度
                if "content" in columns:
                    content = row_data["content"]
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "..."
                    row_data["content"] = content

                # 写入行
                writer.writerow([row_data.get(col, "") for col in columns])

        return len(rows)

    def to_markdown(self, path: str, include_embeddings: bool = False) -> int:
        """导出为Markdown格式

        Args:
            path: 输出文件路径
            include_embeddings: 是否包含嵌入向量

        Returns:
            导出的记录数
        """
        conn = self._get_readonly_connection()  # P0-4修复：使用只读连接
        cursor = conn.execute("""
            SELECT id, content, metadata, timestamp
            FROM memories
            ORDER BY timestamp DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        with open(path, "w", encoding="utf-8") as f:
            f.write("# Memory Export\n\n")
            f.write(f"Exported at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total records: {len(rows)}\n\n")
            f.write("---\n\n")

            for row in rows:
                f.write(f"## {row[0]}\n\n")
                f.write(f"**Timestamp**: {row[3]}\n\n")

                metadata = json.loads(row[2]) if row[2] else {}
                if metadata:
                    f.write(f"**Metadata**: `{json.dumps(metadata, ensure_ascii=False)}`\n\n")

                f.write(f"{row[1]}\n\n")
                f.write("---\n\n")

        return len(rows)

    def from_json(
        self,
        path: str,
        update_existing: bool = True,
        clear_first: bool = False,
    ) -> Dict[str, Any]:
        """从JSON文件导入

        Args:
            path: 输入文件路径
            update_existing: 是否更新已存在的记录
            clear_first: 是否先清空现有数据

        Returns:
            导入结果统计
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        results = {
            "total": len(data),
            "imported": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
        }

        if clear_first:
            conn = self._get_connection()
            conn.execute("DELETE FROM memories")
            conn.commit()
            conn.close()

        for item in data:
            try:
                memory = MemoryItem.from_dict(item)
                if not memory.id:
                    memory.id = f"imp_{uuid.uuid4().hex[:12]}"

                if self._backend:
                    self._backend.add_memory(memory)
                    results["imported"] += 1
                else:
                    conn = self._get_connection()

                    # 检查是否存在
                    cursor = conn.execute(
                        "SELECT id FROM memories WHERE id = ?", (memory.id,)
                    )
                    exists = cursor.fetchone() is not None

                    if exists and not update_existing:
                        results["skipped"] += 1
                    else:
                        self._insert_memory(conn, memory)
                        results["updated" if exists else "imported"] += 1

                    conn.close()

            except Exception:
                results["errors"] += 1

        return results

    def from_csv(
        self,
        path: str,
        id_column: str = "id",
        content_column: str = "content",
        timestamp_column: str = "timestamp",
        metadata_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """从CSV文件导入

        Args:
            path: 输入文件路径
            id_column: ID列名
            content_column: 内容列名
            timestamp_column: 时间戳列名
            metadata_columns: 元数据列名列表

        Returns:
            导入结果统计
        """
        results = {
            "total": 0,
            "imported": 0,
            "errors": 0,
        }

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                results["total"] += 1
                try:
                    memory_id = row.get(id_column, f"imp_{uuid.uuid4().hex[:12]}")
                    content = row.get(content_column, "")
                    timestamp = float(row.get(timestamp_column, time.time()))

                    metadata = {}
                    if metadata_columns:
                        for col in metadata_columns:
                            if col in row and row[col]:
                                metadata[col] = row[col]

                    memory = MemoryItem(
                        id=memory_id,
                        content=content,
                        metadata=metadata,
                        timestamp=timestamp,
                    )

                    if self._backend:
                        self._backend.add_memory(memory)
                    else:
                        conn = self._get_connection()
                        self._insert_memory(conn, memory)
                        conn.close()

                    results["imported"] += 1

                except Exception:
                    results["errors"] += 1

        return results

    def _insert_memory(self, conn: sqlite3.Connection, memory: MemoryItem):
        """插入记忆记录"""
        import numpy as np

        embedding_blob = None
        if memory.embedding:
            embedding_blob = np.array(memory.embedding).tobytes()

        conn.execute("""
            INSERT OR REPLACE INTO memories
            (id, content, metadata, embedding, timestamp, causal_links)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            memory.id,
            memory.content,
            json.dumps(memory.metadata, ensure_ascii=False),
            embedding_blob,
            memory.timestamp,
            json.dumps(memory.causal_links or [], ensure_ascii=False),
        ))
        conn.commit()

    def merge(
        self,
        source_paths: List[str],
        target_db: str,
        conflict_strategy: str = "skip",
    ) -> Dict[str, Any]:
        """合并多个导出文件

        Args:
            source_paths: 源文件路径列表
            target_db: 目标数据库路径
            conflict_strategy: 冲突策略 ("skip" | "overwrite" | "new_id")

        Returns:
            合并结果统计
        """
        results = {
            "total": 0,
            "imported": 0,
            "skipped": 0,
            "errors": 0,
        }

        # 创建临时backend
        from su_memory.storage.sqlite_backend import SQLiteBackend
        temp_backend = SQLiteBackend(target_db)

        for path in source_paths:
            if path.endswith(".json"):
                result = self._merge_json(path, temp_backend, conflict_strategy)
            elif path.endswith(".csv"):
                result = self._merge_csv(path, temp_backend, conflict_strategy)
            else:
                continue

            results["total"] += result.get("total", 0)
            results["imported"] += result.get("imported", 0)
            results["skipped"] += result.get("skipped", 0)
            results["errors"] += result.get("errors", 0)

        temp_backend.close()
        return results

    def _merge_json(
        self,
        path: str,
        backend,
        strategy: str,
    ) -> Dict[str, Any]:
        """合并JSON文件"""
        results = {"total": 0, "imported": 0, "skipped": 0, "errors": 0}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        for item in data:
            try:
                memory = MemoryItem.from_dict(item)
                if not memory.id:
                    memory.id = f"mrg_{uuid.uuid4().hex[:12]}"

                if strategy == "new_id":
                    memory.id = f"mrg_{uuid.uuid4().hex[:12]}"

                backend.add_memory(memory)
                results["imported"] += 1
                results["total"] += 1
            except Exception:
                results["errors"] += 1

        return results

    def _merge_csv(
        self,
        path: str,
        backend,
        strategy: str,
    ) -> Dict[str, Any]:
        """合并CSV文件"""
        results = {"total": 0, "imported": 0, "skipped": 0, "errors": 0}

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    memory = MemoryItem(
                        id=row.get("id", f"mrg_{uuid.uuid4().hex[:12]}"),
                        content=row.get("content", ""),
                        metadata={},
                        timestamp=float(row.get("timestamp", time.time())),
                    )

                    if strategy == "new_id":
                        memory.id = f"mrg_{uuid.uuid4().hex[:12]}"

                    backend.add_memory(memory)
                    results["imported"] += 1
                    results["total"] += 1
                except Exception:
                    results["errors"] += 1

        return results
