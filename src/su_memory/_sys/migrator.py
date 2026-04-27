"""
su-memory 数据迁移模块
Data Migration Module for su-memory SDK

支持从多种数据源迁移历史记忆到su-memory系统
"""

import json
import csv
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """支持的数据源类型"""
    JSON = "json"
    CSV = "csv"
    SQLITE = "sqlite"
    MARKDOWN = "markdown"
    TEXT = "text"
    NOTION = "notion"
    OBSIDIAN = "obsidian"


class MigrationStatus(Enum):
    """迁移状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class MemoryRecord:
    """记忆记录结构"""
    content: str
    timestamp: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = int(datetime.now().timestamp())
        if isinstance(self.timestamp, str):
            # 尝试解析字符串时间戳
            try:
                self.timestamp = int(datetime.fromisoformat(self.timestamp).timestamp())
            except Exception:
                self.timestamp = int(datetime.now().timestamp())


@dataclass
class MigrationError:
    """迁移错误记录"""
    record_id: str
    source_data: Any
    error_message: str
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp()))


@dataclass
class MigrationReport:
    """数据迁移报告"""
    source_type: str
    source_path: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # 统计数据
    total_records: int = 0
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0

    # 详细信息
    errors: List[MigrationError] = field(default_factory=list)
    field_mappings: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # 时间范围
    earliest_timestamp: Optional[int] = None
    latest_timestamp: Optional[int] = None

    status: MigrationStatus = MigrationStatus.PENDING

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result = asdict(self)
        result['status'] = self.status.value
        result['errors'] = [asdict(e) for e in self.errors]
        return result

    def to_markdown(self) -> str:
        """生成Markdown格式报告"""
        lines = [
            "# 📊 数据迁移报告",
            "",
            f"**数据源类型**: {self.source_type}",
            f"**数据源路径**: `{self.source_path}`",
            f"**迁移时间**: {self.started_at}",
            f"**完成时间**: {self.completed_at or '进行中'}",
            f"**状态**: {self.status.value.upper()}",
            "",
            "---",
            "",
            "## 📈 迁移统计",
            "",
            "| 指标 | 数量 |",
            "|------|------|",
            f"| 总记录数 | {self.total_records} |",
            f"| ✅ 成功 | {self.success_count} |",
            f"| ❌ 失败 | {self.failed_count} |",
            f"| ⏭️ 跳过 | {self.skipped_count} |",
            f"| 成功率 | {self.success_count/max(self.total_records,1)*100:.1f}% |",
            "",
        ]

        if self.earliest_timestamp and self.latest_timestamp:
            lines.extend([
                "## 📅 时间范围",
                "",
                f"- **最早记录**: {datetime.fromtimestamp(self.earliest_timestamp).strftime('%Y-%m-%d %H:%M:%S')}",
                f"- **最新记录**: {datetime.fromtimestamp(self.latest_timestamp).strftime('%Y-%m-%d %H:%M:%S')}",
                "",
            ])

        if self.field_mappings:
            lines.extend([
                "## 🔄 字段映射",
                "",
                "| 原始字段 | 目标字段 |",
                "|----------|---------|",
            ])
            for src, tgt in self.field_mappings.items():
                lines.append(f"| `{src}` | `{tgt}` |")
            lines.append("")

        if self.errors:
            lines.extend([
                "## ⚠️ 错误详情",
                "",
            ])
            for i, err in enumerate(self.errors[:20], 1):  # 最多显示20个错误
                lines.append(f"### {i}. 错误记录")
                lines.append(f"- **记录ID**: `{err.record_id}`")
                lines.append(f"- **错误原因**: {err.error_message}")
                if err.source_data:
                    preview = str(err.source_data)[:100]
                    lines.append(f"- **原始数据**: `{preview}...`")
                lines.append("")

        if self.warnings:
            lines.extend([
                "## 📝 警告信息",
                "",
            ])
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        lines.extend([
            "---",
            "",
            "*报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*",
        ])

        return "\n".join(lines)


class DataSourceAdapter:
    """数据源适配器基类"""

    def __init__(self, source_path: str):
        self.source_path = source_path
        self.report = MigrationReport(
            source_type=self.__class__.__name__,
            source_path=source_path
        )

    def scan(self) -> List[Dict[str, Any]]:
        """扫描数据源，返回所有记录"""
        raise NotImplementedError

    def parse_record(self, raw_data: Dict[str, Any]) -> Optional[MemoryRecord]:
        """解析单条记录"""
        raise NotImplementedError

    def get_field_mappings(self) -> Dict[str, str]:
        """获取字段映射关系"""
        return {}


class JSONAdapter(DataSourceAdapter):
    """JSON文件适配器"""

    def scan(self) -> List[Dict[str, Any]]:
        with open(self.source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # 尝试提取记录列表
            for key in ['memories', 'records', 'data', 'items', 'entries']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        return []

    def parse_record(self, raw_data: Dict[str, Any]) -> Optional[MemoryRecord]:
        # 常见字段映射
        content_fields = ['content', 'text', 'body', 'message', 'note', 'description', 'title']
        time_fields = ['timestamp', 'time', 'created_at', 'createdAt', 'date', 'datetime']
        tag_fields = ['tags', 'labels', 'categories', 'keywords']

        content = None
        for field in content_fields:
            if field in raw_data:
                content = str(raw_data[field])
                break

        if not content:
            self.report.warnings.append(f"记录缺少content字段: {str(raw_data)[:50]}")
            return None

        timestamp = None
        for field in time_fields:
            if field in raw_data:
                timestamp = raw_data[field]
                break

        tags = []
        for field in tag_fields:
            if field in raw_data:
                tags = raw_data[field] if isinstance(raw_data[field], list) else [raw_data[field]]
                break

        return MemoryRecord(
            content=content,
            timestamp=timestamp,
            metadata={k: v for k, v in raw_data.items()
                     if k not in content_fields + time_fields + tag_fields},
            tags=tags,
            source_id=str(raw_data.get('id', id(raw_data)))
        )

    def get_field_mappings(self) -> Dict[str, str]:
        return {
            'content': 'content',
            'text': 'content',
            'timestamp': 'timestamp',
            'created_at': 'timestamp',
            'tags': 'tags',
            'labels': 'tags',
        }


class CSVAdapter(DataSourceAdapter):
    """CSV文件适配器"""

    def scan(self) -> List[Dict[str, Any]]:
        records = []
        with open(self.source_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return records

    def parse_record(self, raw_data: Dict[str, Any]) -> Optional[MemoryRecord]:
        content_fields = ['content', 'text', 'body', 'message', 'note', 'description']
        time_fields = ['timestamp', 'time', 'created_at', 'date']

        content = None
        for field in content_fields:
            if field in raw_data and raw_data[field]:
                content = str(raw_data[field])
                break

        if not content:
            # 尝试使用第一列作为内容
            first_col = next(iter(raw_data.values()), None)
            if first_col:
                content = str(first_col)

        if not content:
            return None

        timestamp = None
        for field in time_fields:
            if field in raw_data and raw_data[field]:
                timestamp = raw_data[field]
                break

        return MemoryRecord(
            content=content,
            timestamp=timestamp,
            metadata=dict(raw_data),
            source_id=str(raw_data.get('id', id(raw_data)))
        )

    def get_field_mappings(self) -> Dict[str, str]:
        return {
            'content': 'content',
            'timestamp': 'timestamp',
        }


class SQLiteAdapter(DataSourceAdapter):
    """SQLite数据库适配器"""

    def __init__(self, source_path: str, table_name: str = "memories",
                 content_col: str = "content", time_col: str = "timestamp"):
        super().__init__(source_path)
        self.table_name = table_name
        self.content_col = content_col
        self.time_col = time_col

    def scan(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.source_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(f"SELECT * FROM {self.table_name}")
            rows = cursor.fetchall()
            records = [dict(row) for row in rows]
        except sqlite3.OperationalError as e:
            logger.error(f"表查询失败: {e}")
            # 尝试获取所有表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            logger.info(f"可用表: {[t[0] for t in tables]}")
            records = []
        finally:
            conn.close()

        return records

    def parse_record(self, raw_data: Dict[str, Any]) -> Optional[MemoryRecord]:
        content = raw_data.get(self.content_col) or raw_data.get('content')
        if not content:
            return None

        timestamp = raw_data.get(self.time_col) or raw_data.get('timestamp')

        return MemoryRecord(
            content=str(content),
            timestamp=timestamp,
            metadata={k: v for k, v in raw_data.items()
                     if k not in [self.content_col, self.time_col, 'content', 'timestamp']},
            source_id=str(raw_data.get('id', id(raw_data)))
        )

    def get_field_mappings(self) -> Dict[str, str]:
        return {
            self.content_col: 'content',
            self.time_col: 'timestamp',
        }


class MarkdownAdapter(DataSourceAdapter):
    """Markdown文件适配器（解析笔记）"""

    def scan(self) -> List[Dict[str, Any]]:
        with open(self.source_path, 'r', encoding='utf-8') as f:
            content = f.read()

        records = []
        # 按标题分割
        sections = content.split('\n## ')

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            lines = section.split('\n')
            title = lines[0].strip('# ')
            body = '\n'.join(lines[1:]).strip()

            if body:
                records.append({
                    'title': title,
                    'content': body,
                    'source_id': f'md_{i}'
                })

        return records

    def parse_record(self, raw_data: Dict[str, Any]) -> Optional[MemoryRecord]:
        content = raw_data.get('content')
        if not content:
            return None

        return MemoryRecord(
            content=content,
            title=raw_data.get('title'),
            metadata={'source': 'markdown'},
            source_id=raw_data.get('source_id', id(raw_data))
        )


class TextAdapter(DataSourceAdapter):
    """纯文本文件适配器"""

    def scan(self) -> List[Dict[str, Any]]:
        with open(self.source_path, 'r', encoding='utf-8') as f:
            content = f.read()

        records = []
        # 按空行分割
        sections = content.split('\n\n')

        for i, section in enumerate(sections):
            section = section.strip()
            if section:
                records.append({
                    'content': section,
                    'source_id': f'text_{i}'
                })

        return records

    def parse_record(self, raw_data: Dict[str, Any]) -> Optional[MemoryRecord]:
        return MemoryRecord(
            content=raw_data.get('content', ''),
            source_id=raw_data.get('source_id')
        )


class ObsidianAdapter(DataSourceAdapter):
    """Obsidian笔记适配器"""

    def __init__(self, vault_path: str):
        super().__init__(vault_path)
        self.source_type = "obsidian_vault"

    def scan(self) -> List[Dict[str, Any]]:
        records = []
        vault = Path(self.source_path)

        # 扫描所有markdown文件
        for md_file in vault.rglob('*.md'):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析frontmatter
                metadata = {}
                body = content

                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        fm_text, body = parts[1], parts[2]
                        # 解析frontmatter
                        for line in fm_text.strip().split('\n'):
                            if ':' in line:
                                key, value = line.split(':', 1)
                                metadata[key.strip()] = value.strip()

                # 提取标题
                title = metadata.get('title', md_file.stem)
                tags = metadata.get('tags', '')
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(',')]

                records.append({
                    'title': title,
                    'content': body.strip(),
                    'tags': tags,
                    'file_path': str(md_file.relative_to(vault)),
                    'created': metadata.get('created'),
                    'modified': metadata.get('modified'),
                    'source_id': str(md_file.relative_to(vault))
                })
            except Exception as e:
                logger.warning(f"解析文件失败 {md_file}: {e}")

        return records

    def parse_record(self, raw_data: Dict[str, Any]) -> Optional[MemoryRecord]:
        content = raw_data.get('content')
        if not content:
            return None

        # 转换时间
        timestamp = None
        for time_field in ['created', 'modified']:
            if raw_data.get(time_field):
                try:
                    timestamp = int(datetime.fromisoformat(raw_data[time_field]).timestamp())
                    break
                except Exception:
                    pass

        return MemoryRecord(
            content=content,
            timestamp=timestamp,
            metadata={
                'title': raw_data.get('title'),
                'file_path': raw_data.get('file_path')
            },
            tags=raw_data.get('tags', []),
            source_id=raw_data.get('source_id')
        )

    def get_field_mappings(self) -> Dict[str, str]:
        return {
            'title': 'metadata.title',
            'content': 'content',
            'tags': 'tags',
            'created': 'timestamp',
        }


class MemoryMigrator:
    """记忆迁移器主类"""

    def __init__(self,
                 target_client=None,
                 progress_callback: Optional[Callable[[int, int, str], None]] = None):
        """
        初始化迁移器

        Args:
            target_client: 目标SuMemoryLitePro客户端
            progress_callback: 进度回调函数 (current, total, message)
        """
        self.target_client = target_client
        self.progress_callback = progress_callback
        self.reports: List[MigrationReport] = []

    def _create_adapter(self, source_type: DataSourceType, source_path: str, **kwargs) -> DataSourceAdapter:
        """创建数据源适配器"""
        adapters = {
            DataSourceType.JSON: JSONAdapter,
            DataSourceType.CSV: CSVAdapter,
            DataSourceType.SQLITE: lambda p: SQLiteAdapter(p, **kwargs),
            DataSourceType.MARKDOWN: MarkdownAdapter,
            DataSourceType.TEXT: TextAdapter,
            DataSourceType.OBSIDIAN: ObsidianAdapter,
        }

        adapter_class = adapters.get(source_type, JSONAdapter)
        return adapter_class(source_path)

    def migrate(self,
                source_type: DataSourceType,
                source_path: str,
                target_client=None,
                **adapter_kwargs) -> MigrationReport:
        """
        执行数据迁移

        Args:
            source_type: 数据源类型
            source_path: 数据源路径
            target_client: 目标客户端（如果与初始化时不同）
            **adapter_kwargs: 适配器特定参数

        Returns:
            MigrationReport: 迁移报告
        """
        client = target_client or self.target_client
        if not client:
            raise ValueError("必须提供target_client参数")

        # 创建适配器
        adapter = self._create_adapter(source_type, source_path, **adapter_kwargs)
        report = adapter.report
        report.field_mappings = adapter.get_field_mappings()

        try:
            # 扫描数据源
            raw_records = adapter.scan()
            report.total_records = len(raw_records)

            logger.info(f"扫描到 {report.total_records} 条记录")

            # 逐条迁移
            for i, raw_data in enumerate(raw_records):
                # 进度回调
                if self.progress_callback:
                    self.progress_callback(i + 1, report.total_records,
                                          f"正在处理第 {i+1}/{report.total_records} 条")

                try:
                    # 解析记录
                    record = adapter.parse_record(raw_data)

                    if record is None:
                        report.skipped_count += 1
                        continue

                    # 更新时间范围
                    if report.earliest_timestamp is None or record.timestamp < report.earliest_timestamp:
                        report.earliest_timestamp = record.timestamp
                    if report.latest_timestamp is None or record.timestamp > report.latest_timestamp:
                        report.latest_timestamp = record.timestamp

                    # 添加到目标系统 (使用正确的接口)
                    metadata = dict(record.metadata) if record.metadata else {}
                    if record.tags:
                        metadata['tags'] = record.tags

                    client.add(
                        content=record.content,
                        metadata=metadata,
                        parent_ids=None,
                        topic=None,
                        session_id=None
                    )

                    report.success_count += 1

                except Exception as e:
                    report.failed_count += 1
                    report.errors.append(MigrationError(
                        record_id=str(raw_data.get('id', i)),
                        source_data=raw_data,
                        error_message=str(e)
                    ))
                    logger.warning(f"迁移记录失败: {e}")

            # 更新状态
            if report.failed_count == 0:
                report.status = MigrationStatus.COMPLETED
            elif report.success_count > 0:
                report.status = MigrationStatus.PARTIAL
            else:
                report.status = MigrationStatus.FAILED

            report.completed_at = datetime.now().isoformat()

        except Exception as e:
            report.status = MigrationStatus.FAILED
            report.errors.append(MigrationError(
                record_id="SCAN_ERROR",
                source_data=None,
                error_message=f"扫描失败: {str(e)}"
            ))
            logger.error(f"迁移失败: {e}")

        self.reports.append(report)
        return report

    def migrate_multiple(self, sources: List[Dict]) -> List[MigrationReport]:
        """批量迁移多个数据源"""
        reports = []
        for source in sources:
            report = self.migrate(
                source_type=source['type'],
                source_path=source['path'],
                target_client=source.get('client', self.target_client),
                **source.get('options', {})
            )
            reports.append(report)
        return reports

    def get_combined_report(self) -> MigrationReport:
        """获取合并报告"""
        combined = MigrationReport(
            source_type="multiple",
            source_path="多数据源合并"
        )

        for report in self.reports:
            combined.total_records += report.total_records
            combined.success_count += report.success_count
            combined.failed_count += report.failed_count
            combined.skipped_count += report.skipped_count
            combined.errors.extend(report.errors)
            combined.warnings.extend(report.warnings)

            if combined.earliest_timestamp is None:
                combined.earliest_timestamp = report.earliest_timestamp
            elif report.earliest_timestamp:
                combined.earliest_timestamp = min(combined.earliest_timestamp, report.earliest_timestamp)

            if combined.latest_timestamp is None:
                combined.latest_timestamp = report.latest_timestamp
            elif report.latest_timestamp:
                combined.latest_timestamp = max(combined.latest_timestamp, report.latest_timestamp)

        if combined.failed_count == 0 and combined.total_records > 0:
            combined.status = MigrationStatus.COMPLETED
        elif combined.success_count > 0:
            combined.status = MigrationStatus.PARTIAL
        else:
            combined.status = MigrationStatus.FAILED

        combined.completed_at = datetime.now().isoformat()

        return combined


def create_migration_report_file(report: MigrationReport, output_path: str):
    """保存迁移报告到文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report.to_markdown())
    logger.info(f"报告已保存到: {output_path}")


# 便捷函数
def migrate_json(json_path: str, target_client,
                  progress_callback=None) -> MigrationReport:
    """从JSON文件迁移"""
    migrator = MemoryMigrator(target_client=target_client,
                              progress_callback=progress_callback)
    return migrator.migrate(DataSourceType.JSON, json_path)


def migrate_csv(csv_path: str, target_client,
                progress_callback=None) -> MigrationReport:
    """从CSV文件迁移"""
    migrator = MemoryMigrator(target_client=target_client,
                              progress_callback=progress_callback)
    return migrator.migrate(DataSourceType.CSV, csv_path)


def migrate_sqlite(db_path: str, table_name: str,
                   target_client, progress_callback=None) -> MigrationReport:
    """从SQLite数据库迁移"""
    migrator = MemoryMigrator(target_client=target_client,
                              progress_callback=progress_callback)
    return migrator.migrate(DataSourceType.SQLITE, db_path, table_name=table_name)


def migrate_obsidian(vault_path: str, target_client,
                     progress_callback=None) -> MigrationReport:
    """从Obsidian知识库迁移"""
    migrator = MemoryMigrator(target_client=target_client,
                              progress_callback=progress_callback)
    return migrator.migrate(DataSourceType.OBSIDIAN, vault_path)
