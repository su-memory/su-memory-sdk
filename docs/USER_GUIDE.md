# su-memory SDK 用户使用指南

> 版本: v1.4.0 | 更新日期: 2026-04-25

---

## 一、产品介绍

su-memory SDK 是一款本地优先的AI记忆引擎，支持多跳推理、时空索引、多模态嵌入和三维世界模型。

### 1.1 核心特性

- **本地优先**: 数据完全存储在本地，保护隐私
- **多跳推理**: 支持3跳以上的因果链追踪
- **时空感知**: 理解时间衰减和时序关系
- **多模态**: 支持图像、音频、文本统一检索
- **三维世界**: 空间+时间+语义三维检索

---

## 二、安装

### 2.1 环境要求

- Python 3.10+
- 32GB+ 内存（推荐）
- macOS/Linux/Windows

### 2.2 安装步骤

```bash
# 基础安装
pip install su-memory

# 可选：安装FAISS加速（推荐）
pip install faiss-cpu

# 可选：GPU加速
pip install faiss-gpu

# 安装Ollama（用于本地向量模型）
# macOS
brew install ollama
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# 启动Ollama并拉取模型
ollama serve
ollama pull bge-m3
```

### 2.3 验证安装

```python
from su_memory import SuMemoryLitePro

pro = SuMemoryLitePro()
print("✅ 安装成功！")
```

---

## 三、快速开始

### 3.1 基础使用

```python
from su_memory import SuMemoryLitePro

# 创建客户端
pro = SuMemoryLitePro()

# 添加记忆
pro.add("今天学习了机器学习")
pro.add("机器学习是人工智能的核心技术")
pro.add("深度学习是机器学习的重要分支")

# 简单查询
results = pro.query("人工智能", top_k=3)
print(results)
```

### 3.2 多跳推理

```python
# 添加因果链记忆
id1 = pro.add("努力学习可以提高成绩")
id2 = pro.add("成绩提高会获得奖学金")
id3 = pro.add("获得奖学金可以减轻家庭负担")

# 建立因果链接
pro.link_memories(id1, id2, "cause")
pro.link_memories(id2, id3, "result")

# 多跳推理查询
results = pro.query_multihop("努力学习的影响", max_hops=3)
for r in results:
    print(f"{r['content']} (hops={r['hops']})")
```

---

## 四、高级功能

### 4.1 会话管理

```python
# 创建会话
session_id = pro.create_session("项目会议")

# 在会话中添加记忆
pro.add("讨论了技术方案", topic="技术", session_id=session_id)
pro.add("确定了项目时间表", topic="进度", session_id=session_id)

# 获取会话记忆
session = pro.get_session(session_id)
print(session)

# 列出所有会话
sessions = pro.list_sessions()
```

### 4.2 时空检索

```python
import time

# 添加带时间戳的记忆
ts1 = int(time.time())
ts2 = ts1 + 86400  # 1天后

pro.add("项目启动", timestamp=ts1)
pro.add("完成第一阶段", timestamp=ts2)

# 时间范围查询
results = pro.query("项目", time_range=(ts1, ts2))
```

### 4.3 时序预测

```python
# 添加历史事件
pro.add("周一项目启动")
pro.add("周三完成第一阶段")
pro.add("周五测试通过")

# 预测趋势
prediction = pro.predict(metric="activity")
print(prediction)
```

### 4.4 可解释性

```python
# 查询并获取解释
results = pro.query("项目")
explanation = pro.explain_query("项目", results)

print(explanation['explanation'])
```

---

## 五、多模态功能

### 5.1 启用多模态

```python
from su_memory.sdk.multimodal import create_multimodal_manager

# 创建多模态管理器
manager = create_multimodal_manager(
    text_embedding_func=pro._embedding.encode,
    enable_image=True,  # 启用CLIP图像编码
    enable_audio=False,
    image_weight=0.4,
    text_weight=0.6
)
```

### 5.2 添加图像记忆

```python
# 添加带图像的记忆
manager.add_multimodal_memory(
    memory_id="img_001",
    content="会议室场景",
    image_path="/path/to/image.jpg"
)
```

### 5.3 多模态检索

```python
# 文本检索
results = manager.search("会议", mode="text", top_k=5)

# 图像检索
results = manager.search("会议", query_image="/path/to/query.jpg", mode="image")

# 多模态融合检索
results = manager.search("会议", mode="multimodal", top_k=5)
```

---

## 六、三维世界模型

### 6.1 添加空间记忆

```python
# 添加带空间坐标的记忆
pro._spatial.add_spatial_memory(
    memory_id="loc_001",
    content="在会议室A发生的事件",
    position=(10.0, 20.0, 0.0),  # x, y, z
    timestamp=1704067200,
    entity_id="user_001"  # 可选：用于轨迹追踪
)
```

### 6.2 空间邻域搜索

```python
# 查找附近5米内的事件
results = pro._spatial.search_nearby(
    position=(10.0, 20.0, 0.0),
    radius=5.0
)

for r in results:
    print(f"{r.content} (距离={r.distance:.2f}m)")
```

### 6.3 三维检索

```python
# 空间+时间+语义三维检索
results = pro._spatial.search_3d(
    query="会议",
    position=(10.0, 20.0, 0.0),
    time_range=(1704067200, 1704153600),
    max_distance=10.0
)
```

### 6.4 路径搜索

```python
# 查找从起点到终点的路径
results = pro._spatial.search_path(
    start_pos=(0.0, 0.0, 0.0),
    end_pos=(100.0, 50.0, 0.0),
    max_distance=5.0
)
```

---

## 七、配置指南

### 7.1 Ollama配置

```python
# 使用自定义Ollama地址
pro = SuMemoryLitePro(
    embedding_backend='ollama',
    model_name='bge-m3'
)

# 设置Ollama环境变量
import os
os.environ['OLLAMA_HOST'] = 'http://localhost:11434'
```

### 7.2 向量量化配置

```python
# 在VectorGraphRAG中启用量化
vg = VectorGraphRAG(
    embedding_func=encode_func,
    quantization_mode="int8"  # fp32/fp16/int8/binary
)

# 推荐配置
config = {
    "fp32": {"压缩比": "1x", "精度": "最高"},
    "fp16": {"压缩比": "2x", "精度": "高"},
    "int8": {"压缩比": "4x", "精度": "较高"},  # 推荐
    "binary": {"压缩比": "32x", "精度": "一般"}
}
```

### 7.3 HNSW参数配置

```python
# 自定义HNSW参数
vg = VectorGraphRAG(
    embedding_func=encode_func,
    hnsw_m=32,                    # 连接数
    hnsw_ef_construction=64,      # 构建时搜索宽度
    hnsw_ef_search=64             # 搜索时搜索宽度
)
```

---

## 八、最佳实践

### 8.1 记忆组织

```python
# 使用话题标签组织记忆
pro.add("讨论了数据库设计", topic="技术")
pro.add("讨论了API设计", topic="技术")
pro.add("讨论了项目进度", topic="管理")

# 使用能量类型分类
pro.add("技术文档", energy_type="金")  # 文档类
pro.add("沟通记录", energy_type="木")  # 沟通类
pro.add("数据分析", energy_type="水")  # 数据类
pro.add("紧急任务", energy_type="火")  # 紧急类
pro.add("常规任务", energy_type="土")  # 常规类
```

### 8.2 因果链构建

```python
# 构建完整的因果链
memories = []
for i in range(5):
    mem_id = pro.add(f"步骤{i+1}：完成子任务{i+1}")
    memories.append(mem_id)

# 建立链接
for i in range(len(memories) - 1):
    pro.link_memories(memories[i], memories[i+1], "sequence")
```

### 8.3 性能优化

```python
# 批量添加
batch_contents = [f"记忆{i}" for i in range(100)]
for content in batch_contents:
    pro.add(content)

# 使用LRU缓存
pro._cache_size = 2000  # 增加缓存大小

# 启用FAISS
# 确保安装: pip install faiss-cpu
```

---

## 九、故障排除

### 9.1 常见问题

#### Q: Ollama连接失败

```python
# 检查Ollama服务
import requests
response = requests.get("http://localhost:11434/api/tags")
print(response.json())

# 启动Ollama
ollama serve
```

#### Q: FAISS未安装

```
# 安装FAISS
pip install faiss-cpu
# 或GPU版本
pip install faiss-gpu

# 重启Python解释器
```

#### Q: 内存不足

```python
# 启用量化压缩
vg = VectorGraphRAG(
    embedding_func=encode_func,
    quantization_mode="int8"  # 4x压缩
)

# 减少最大记忆数
pro = SuMemoryLitePro(max_memories=5000)
```

### 9.2 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查组件状态
pro = SuMemoryLitePro()
print(f"VectorGraphRAG: {pro._vector_graph is not None}")
print(f"SpacetimeIndex: {pro._spacetime is not None}")
print(f"Multimodal: {pro._multimodal is not None}")
print(f"SpatialRAG: {pro._spatial is not None}")
```

---

## 十、示例代码

### 10.1 个人助手

```python
from su_memory import SuMemoryLitePro

class PersonalAssistant:
    def __init__(self):
        self.pro = SuMemoryLitePro()
    
    def learn(self, content):
        """学习新知识"""
        return self.pro.add(content)
    
    def recall(self, query, max_hops=3):
        """回忆相关记忆"""
        return self.pro.query_multihop(query, max_hops=max_hops)
    
    def connect(self, cause, effect):
        """建立因果关系"""
        id1 = self.pro.add(cause)
        id2 = self.pro.add(effect)
        self.pro.link_memories(id1, id2, "cause")
        return id1, id2

# 使用
assistant = PersonalAssistant()
assistant.learn("学习Python可以提高编程能力")
assistant.learn("编程能力强可以获得好工作")
assistant.connect("学习Python可以提高编程能力", "编程能力强可以获得好工作")

results = assistant.recall("学习Python")
print(results)
```

### 10.2 会议助手

```python
from su_memory import SuMemoryLitePro
import time

class MeetingAssistant:
    def __init__(self, meeting_id):
        self.pro = SuMemoryLitePro()
        self.meeting_id = meeting_id
    
    def record(self, content):
        """记录会议内容"""
        return self.pro.add(content, timestamp=int(time.time()))
    
    def search_meeting(self, query):
        """搜索会议内容"""
        return self.pro.query(query)
    
    def add_location(self, content, position):
        """添加位置信息"""
        self.pro._spatial.add_spatial_memory(
            memory_id=f"loc_{time.time()}",
            content=content,
            position=position,
            timestamp=int(time.time())
        )

# 使用
assistant = MeetingAssistant("meeting_2024_01")
assistant.record("讨论了技术方案")
assistant.record("确定了时间表")
assistant.add_location("技术讨论", (10.0, 20.0, 0.0))
```

---

## 十一、版本迁移

### 11.1 v1.3.x → v1.4.0

主要API变化：

| 旧API | 新API |
|-------|-------|
| `Lega1` | `energy_type` |
| `Lega2` | `time_code` |
| `Lega3` | `category` |

### 11.2 升级步骤

```bash
# 升级包
pip install --upgrade su-memory

# 检查版本
python -c "import su_memory; print(su_memory.__version__)"
```

---

## 十二、获取帮助

- **文档**: https://github.com/su-memory/su-memory-sdk
- **问题反馈**: https://github.com/su-memory/su-memory-sdk/issues
- **邮箱**: sandysu737@gmail.com

---

**文档版本**: v1.0
**更新日期**: 2026-04-25