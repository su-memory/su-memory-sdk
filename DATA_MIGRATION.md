# 📦 数据迁移指南

su-memory SDK 提供完整的数据迁移解决方案，帮助用户将历史记忆从各种数据源导入到 su-memory 系统。

## 功能特性

- ✅ 支持多种数据源：JSON、CSV、SQLite、Markdown、纯文本、Obsidian
- ✅ 自动字段映射和转换
- ✅ 详细迁移报告生成
- ✅ 进度跟踪和错误处理
- ✅ 时序连贯性保证
- ✅ 迁移后数据完整性验证

---

## 快速开始

### 1. 基础用法

```python
from su_memory import SuMemoryLitePro, migrate_json

# 创建目标客户端
client = SuMemoryLitePro(storage_path='./my_memories')

# 一行代码迁移JSON文件
report = migrate_json('path/to/memories.json', client)

print(f"成功迁移: {report.success_count}/{report.total_records}")
```

### 2. 完整迁移流程

```python
from su_memory import SuMemoryLitePro, MemoryMigrator, DataSourceType

# 创建客户端
client = SuMemoryLitePro(storage_path='./my_memories')

# 创建迁移器
migrator = MemoryMigrator(target_client=client)

# 执行迁移
report = migrator.migrate(
    source_type=DataSourceType.JSON,
    source_path='path/to/data.json'
)

# 保存报告
with open('migration_report.md', 'w') as f:
    f.write(report.to_markdown())
```

---

## 支持的数据源

### 1. JSON 文件

```python
from su_memory import migrate_json

report = migrate_json('memories.json', client)
```

**支持的字段映射**：
| JSON字段 | 目标字段 |
|----------|----------|
| content, text, body | content |
| timestamp, time, created_at | timestamp |
| tags, labels | tags |

### 2. CSV 文件

```python
from su_memory import migrate_csv

report = migrate_csv('notes.csv', client)
```

### 3. SQLite 数据库

```python
from su_memory import migrate_sqlite

report = migrate_sqlite(
    'app.db',
    table_name='memories',
    client=client
)
```

### 4. Obsidian 知识库

```python
from su_memory import migrate_obsidian

report = migrate_obsidian(
    '/path/to/vault',
    client=client
)
```

支持自动解析：
- Frontmatter 元数据
- 标题和标签
- 创建/修改时间

### 5. Markdown 文件

```python
from su_memory import MemoryMigrator, DataSourceType

migrator = MemoryMigrator(target_client=client)
report = migrator.migrate(DataSourceType.MARKDOWN, 'notes.md')
```

### 6. 纯文本文件

```python
from su_memory import MemoryMigrator, DataSourceType

migrator = MemoryMigrator(target_client=client)
report = migrator.migrate(DataSourceType.TEXT, 'diary.txt')
```

---

## 批量迁移

```python
from su_memory import MemoryMigrator

migrator = MemoryMigrator(target_client=client)

# 定义多个数据源
sources = [
    {
        'type': DataSourceType.JSON,
        'path': 'data/chat_history.json',
    },
    {
        'type': DataSourceType.CSV,
        'path': 'data/notes.csv',
    },
    {
        'type': DataSourceType.OBSIDIAN,
        'path': '/Users/me/Obsidian/Vault',
    },
]

# 批量迁移
reports = migrator.migrate_multiple(sources)

# 获取合并报告
combined = migrator.get_combined_report()
```

---

## 迁移报告

迁移完成后会自动生成详细报告：

```markdown
# 📊 数据迁移报告

**数据源类型**: JSON
**数据源路径**: `memories.json`
**迁移时间**: 2024-04-25T10:30:00
**状态**: COMPLETED

---

## 📈 迁移统计

| 指标 | 数量 |
|------|------|
| 总记录数 | 100 |
| ✅ 成功 | 98 |
| ❌ 失败 | 2 |
| 成功率 | 98.0% |

## 📅 时间范围

- **最早记录**: 2024-01-01 08:00:00
- **最新记录**: 2024-04-25 10:30:00

## ⚠️ 错误详情

1. 记录ID: rec_001
   - 错误原因: 缺少content字段

2. 记录ID: rec_042
   - 错误原因: 时间戳格式错误
```

---

## 进度跟踪

```python
def my_progress_callback(current, total, message):
    print(f"进度: {current}/{total} - {message}")

migrator = MemoryMigrator(
    target_client=client,
    progress_callback=my_progress_callback
)

report = migrator.migrate(DataSourceType.JSON, 'large_file.json')
```

---

## 命令行工具

使用示例脚本：

```bash
# 迁移JSON文件
python examples/migrate_data.py --source json --path data/memories.json

# 迁移CSV文件
python examples/migrate_data.py --source csv --path data/notes.csv

# 迁移SQLite数据库
python examples/migrate_data.py --source sqlite --path app.db --table memories

# 迁移Obsidian知识库
python examples/migrate_data.py --source obsidian --path /path/to/vault

# 运行演示模式（使用示例数据）
python examples/migrate_data.py --create-demo
```

---

## 自定义适配器

扩展支持新的数据源：

```python
from su_memory._sys.migrator import DataSourceAdapter, MemoryRecord

class CustomAdapter(DataSourceAdapter):
    """自定义数据源适配器"""
    
    def scan(self):
        # 扫描数据源
        return [...]
    
    def parse_record(self, raw_data):
        # 解析单条记录
        return MemoryRecord(
            content=raw_data['text'],
            timestamp=raw_data['date'],
            metadata={'source': 'custom'}
        )

# 使用自定义适配器
from su_memory import MemoryMigrator

migrator = MemoryMigrator(target_client=client)
report = migrator.migrate(CustomAdapter('/path/to/data'))
```

---

## 时序连贯性

迁移过程中会自动处理时间戳：

1. **保留原始时间**：如果数据包含时间信息，会保留
2. **自动填充**：如果缺少时间，使用当前时间
3. **时间排序**：迁移后数据按时间顺序存储

---

## 错误处理

- **部分失败**：如果部分记录迁移失败，会继续处理剩余记录
- **详细日志**：每个错误都会记录原始数据和错误原因
- **跳过策略**：格式不正确的记录会被跳过，不会阻塞整个迁移

---

## 最佳实践

1. **小批量迁移**：先迁移少量数据验证格式
2. **检查报告**：仔细阅读迁移报告中的警告和错误
3. **验证数据**：迁移后执行检索测试确认数据完整性
4. **备份数据**：迁移前备份原始数据

---

## API 参考

### MemoryMigrator

```python
class MemoryMigrator:
    def __init__(self, target_client, progress_callback=None)
    def migrate(self, source_type, source_path, target_client=None, **kwargs) -> MigrationReport
    def migrate_multiple(self, sources) -> List[MigrationReport]
    def get_combined_report(self) -> MigrationReport
```

### MigrationReport

```python
@dataclass
class MigrationReport:
    source_type: str
    source_path: str
    total_records: int
    success_count: int
    failed_count: int
    skipped_count: int
    errors: List[MigrationError]
    field_mappings: Dict[str, str]
    earliest_timestamp: Optional[int]
    latest_timestamp: Optional[int]
    status: MigrationStatus
    
    def to_dict(self) -> Dict
    def to_markdown(self) -> str
```

---

*文档更新: 2024-04-25*
