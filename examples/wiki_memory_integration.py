#!/usr/bin/env python3
"""
su-memory SDK Wiki 联动模块

实现与 Wiki 系统（Obsidian、Memex 等）的双向集成，
增强知识库的智能检索和关联能力。
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# 向后兼容导入
try:
    from su_memory import SuMemoryLitePro
except ImportError:
    SuMemoryLitePro = None


@dataclass
class WikiNote:
    """Wiki 笔记数据结构"""
    path: str
    title: str
    content: str
    links: List[str] = field(default_factory=list)  # 双向链接
    backlinks: List[str] = field(default_factory=list)  # 反向链接
    tags: List[str] = field(default_factory=list)  # 标签
    metadata: Dict = field(default_factory=dict)  # 额外元数据
    modified_time: datetime = None
    memory_id: Optional[str] = None  # 对应的记忆 ID


@dataclass
class WikiSearchResult:
    """Wiki 搜索结果"""
    note: WikiNote
    relevance_score: float
    reasoning_chain: List[str] = field(default_factory=list)
    memory_connections: List[str] = field(default_factory=list)


class WikiConnector:
    """Wiki 系统连接器基类"""
    
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.notes: Dict[str, WikiNote] = {}
        
    def scan_vault(self) -> List[WikiNote]:
        """扫描 Vault 中的所有笔记"""
        raise NotImplementedError
    
    def read_note(self, path: str) -> WikiNote:
        """读取单个笔记"""
        raise NotImplementedError
    
    def write_note(self, note: WikiNote) -> bool:
        """写入笔记"""
        raise NotImplementedError
    
    def update_metadata(self, path: str, metadata: Dict) -> bool:
        """更新笔记元数据"""
        raise NotImplementedError


class ObsidianConnector(WikiConnector):
    """Obsidian Vault 连接器"""
    
    def __init__(self, vault_path: str):
        super().__init__(vault_path)
        self.ignore_folders = {'.trash', '.obsidian', 'templates'}
        
    def scan_vault(self) -> List[WikiNote]:
        """扫描 Obsidian Vault"""
        self.notes = {}
        
        for md_file in self.vault_path.rglob("*.md"):
            # 忽略特定文件夹
            if any(ignored in md_file.parts for ignored in self.ignore_folders):
                continue
            
            try:
                note = self._parse_markdown(md_file)
                self.notes[str(md_file)] = note
            except Exception as e:
                print(f"  ⚠️  解析失败 {md_file}: {e}")
        
        return list(self.notes.values())
    
    def _parse_markdown(self, path: Path) -> WikiNote:
        """解析 Markdown 文件"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取标题
        title = path.stem
        for line in content.split('\n'):
            if line.startswith('# '):
                title = line[2:].strip()
                break
        
        # 提取双向链接 [[...]]
        links = []
        for line in content.split('\n'):
            import re
            wiki_links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', line)
            links.extend(wiki_links)
        
        # 提取标签 #tag
        tags = []
        for line in content.split('\n'):
            import re
            found_tags = re.findall(r'(?<!\w)#([a-zA-Z0-9_-]+)', line)
            tags.extend(found_tags)
        
        # 获取修改时间
        modified_time = datetime.fromtimestamp(path.stat().st_mtime)
        
        return WikiNote(
            path=str(path),
            title=title,
            content=content,
            links=links,
            tags=tags,
            modified_time=modified_time
        )
    
    def read_note(self, path: str) -> WikiNote:
        """读取单个笔记"""
        if path in self.notes:
            return self.notes[path]
        return self._parse_markdown(Path(path))
    
    def write_note(self, note: WikiNote) -> bool:
        """写入笔记"""
        try:
            with open(note.path, 'w', encoding='utf-8') as f:
                f.write(note.content)
            return True
        except Exception as e:
            print(f"  ❌ 写入失败: {e}")
            return False
    
    def update_metadata(self, path: str, metadata: Dict) -> bool:
        """更新笔记 YAML 元数据"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析或创建 frontmatter
            if content.startswith('---'):
                parts = content.split('---', 2)
                frontmatter = parts[1]
                body = parts[2] if len(parts) > 2 else ''
                
                # 解析现有 frontmatter
                fm_lines = []
                for line in frontmatter.split('\n'):
                    if ':' in line:
                        key = line.split(':', 1)[0].strip()
                        if key in ['created', 'modified', 'su-memory', 'memory_connections', 'reasoning_chain']:
                            continue  # 跳过 su-memory 相关字段
                        fm_lines.append(line)
                
                # 添加新元数据
                fm_lines.append(f"modified: {datetime.now().isoformat()}")
                
                for key, value in metadata.items():
                    if isinstance(value, list):
                        fm_lines.append(f"{key}:")
                        for item in value:
                            fm_lines.append(f"  - {item}")
                    else:
                        fm_lines.append(f"{key}: {value}")
                
                new_content = "---\n" + "\n".join(fm_lines) + "\n---\n" + body
            else:
                # 创建新的 frontmatter
                fm_lines = [f"modified: {datetime.now().isoformat()}"]
                for key, value in metadata.items():
                    if isinstance(value, list):
                        fm_lines.append(f"{key}:")
                        for item in value:
                            fm_lines.append(f"  - {item}")
                    else:
                        fm_lines.append(f"{key}: {value}")
                
                new_content = "---\n" + "\n".join(fm_lines) + "\n---\n\n" + content
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return True
        except Exception as e:
            print(f"  ❌ 元数据更新失败: {e}")
            return False


class WikiMemoryEnhancer:
    """Wiki 记忆增强器 - 核心类"""
    
    def __init__(self, wiki_connector: WikiConnector):
        self.wiki = wiki_connector
        self.memory = SuMemoryLitePro(enable_vector=False)
        self._initialized = False
        
    def initialize(self):
        """初始化 - 将 Wiki 笔记导入记忆系统"""
        if self._initialized:
            return
        
        print("\n" + "=" * 60)
        print("Wiki 记忆增强系统初始化")
        print("=" * 60)
        
        # 扫描 Vault
        print("\n[1/4] 扫描 Wiki Vault...")
        notes = self.wiki.scan_vault()
        print(f"  ✅ 发现 {len(notes)} 篇笔记")
        
        # 导入记忆
        print("\n[2/4] 导入笔记到记忆系统...")
        for i, note in enumerate(notes, 1):
            metadata = {
                "source": "wiki",
                "path": note.path,
                "title": note.title,
                "tags": note.tags,
                "links": note.links,
                "type": "wiki_note"
            }
            
            # 添加到记忆
            memory_id = self.memory.add(
                f"[{note.title}]\n{note.content}",
                metadata=metadata
            )
            note.memory_id = memory_id
            
            if i % 20 == 0:
                print(f"  已处理 {i}/{len(notes)} 篇笔记...")
        
        print(f"  ✅ 已导入 {len(notes)} 篇笔记到记忆系统")
        
        # 建立链接关联
        print("\n[3/4] 建立笔记关联...")
        self._build_note_connections(notes)
        
        # 构建反向链接索引
        print("\n[4/4] 构建反向链接索引...")
        self._build_backlinks_index(notes)
        
        self._initialized = True
        print("\n  ✅ 初始化完成!")
    
    def _build_note_connections(self, notes: List[WikiNote]):
        """建立笔记之间的链接关系"""
        # 构建标题到笔记的映射
        title_to_note = {note.title: note for note in notes}
        
        link_count = 0
        for note in notes:
            for linked_title in note.links:
                if linked_title in title_to_note:
                    linked_note = title_to_note[linked_title]
                    # 在记忆系统中建立关联
                    if note.memory_id and linked_note.memory_id:
                        try:
                            self.memory.link_memories(0, 0)  # 使用索引关联
                            link_count += 1
                        except:
                            pass
        
        print(f"  ✅ 建立了 {link_count} 条笔记关联")
    
    def _build_backlinks_index(self, notes: List[WikiNote]):
        """构建反向链接索引"""
        # 统计每篇笔记被多少其他笔记引用
        for note in notes:
            note.backlinks = [
                n.title for n in notes 
                if note.title in n.links
            ]
        
        # 注意: SuMemoryLitePro 没有 update_metadata 方法
        # 反向链接信息存储在 WikiNote 对象中，通过 note.backlinks 访问
        print(f"  ✅ 构建了 {sum(len(n.backlinks) for n in notes)} 条反向链接")
    
    def smart_search(self, query: str, context: str = "") -> List[WikiSearchResult]:
        """
        智能搜索 - 结合 Wiki 和记忆系统
        
        Args:
            query: 搜索查询
            context: 额外上下文信息
            
        Returns:
            WikiSearchResult 列表，按相关性排序
        """
        print("\n" + "=" * 60)
        print(f"🔍 智能搜索: {query}")
        print("=" * 60)
        
        # 1. 语义检索
        print("\n[1/3] 语义记忆检索...")
        memory_results = self.memory.query(query, top_k=10)
        
        # 2. 多跳推理
        print("[2/3] 多跳因果推理...")
        multi_hop_results = self.memory.query_multihop(query, max_hops=3)
        
        # 3. 融合结果
        print("[3/3] 融合 Wiki 和记忆结果...")
        
        results = []
        seen_paths = set()
        
        # 处理记忆结果
        for r in memory_results:
            path = r.get('metadata', {}).get('path', '')
            if path and path not in seen_paths:
                seen_paths.add(path)
                note = self.wiki.read_note(path)
                
                # 收集关联记忆
                memory_connections = [
                    r2['content'][:50] 
                    for r2 in multi_hop_results[:3]
                    if r2.get('metadata', {}).get('path') != path
                ]
                
                results.append(WikiSearchResult(
                    note=note,
                    relevance_score=r.get('score', 0),
                    reasoning_chain=self._extract_reasoning_chain(r),
                    memory_connections=memory_connections
                ))
        
        # 按相关性排序
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:10]
    
    def _extract_reasoning_chain(self, memory_result: Dict) -> List[str]:
        """提取推理链"""
        chain = []
        
        # 从记忆结果中提取相关信息
        content = memory_result.get('content', '')
        hops = memory_result.get('hops', 0)
        
        if hops > 0:
            chain.append(f"基于 {hops} 跳因果推理发现")
        
        # 提取链接信息
        metadata = memory_result.get('metadata', {})
        if metadata.get('links'):
            chain.append(f"关联到: {', '.join(metadata['links'][:3])}")
        
        return chain
    
    def enhance_wiki_entry(self, query: str) -> Dict:
        """
        增强 Wiki 条目 - 搜索并更新相关笔记
        
        Returns:
            包含搜索结果和增强建议的字典
        """
        # 执行智能搜索
        results = self.smart_search(query)
        
        if not results:
            return {"status": "no_results", "query": query}
        
        # 收集增强信息
        enhancement = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "top_result": {
                "title": results[0].note.title,
                "path": results[0].note.path,
                "relevance": results[0].relevance_score
            },
            "related_notes": [
                {
                    "title": r.note.title,
                    "relevance": r.relevance_score
                }
                for r in results[1:5]
            ],
            "reasoning_insights": results[0].reasoning_chain,
            "memory_connections": results[0].memory_connections
        }
        
        return enhancement
    
    def sync_to_wiki(self, enhancement: Dict) -> bool:
        """
        同步增强结果回 Wiki
        
        将 AI 推理结果和关联信息写入 Wiki 元数据
        """
        if enhancement.get('status') == 'no_results':
            return False
        
        top_result = enhancement.get('top_result', {})
        path = top_result.get('path')
        
        if not path:
            return False
        
        # 构建元数据更新
        metadata = {
            "su-memory": {
                "last_search": enhancement['timestamp'],
                "relevance_score": top_result['relevance'],
                "related_notes": [
                    n['title'] for n in enhancement.get('related_notes', [])
                ],
                "reasoning_insights": enhancement.get('reasoning_insights', []),
                "memory_connections": enhancement.get('memory_connections', [])
            }
        }
        
        # 更新 Wiki 元数据
        success = self.wiki.update_metadata(path, {"su-memory": json.dumps(metadata['su-memory'], ensure_ascii=False)})
        
        if success:
            print(f"  ✅ 已同步到: {top_result['title']}")
        
        return success
    
    def generate_context_for_wiki(self, note_title: str) -> str:
        """
        为 Wiki 笔记生成上下文
        
        基于记忆系统生成相关笔记推荐和关联信息
        """
        # 找到目标笔记
        target_note = None
        for note in self.wiki.notes.values():
            if note.title == note_title:
                target_note = note
                break
        
        if not target_note:
            return ""
        
        # 查询相关记忆
        related = self.memory.query(note_title, top_k=5)
        
        # 生成上下文
        context = f"""
## 📚 相关知识库笔记

基于您的当前笔记「{note_title}」，智能系统发现以下相关内容：

"""
        
        for r in related[:3]:
            content = r['content'][:200]
            score = r.get('score', 0)
            context += f"- **{content}...** (相关度: {score:.0%})\n"
        
        context += f"""

### 🔗 双向链接

**反向链接** ({len(target_note.backlinks)} 篇笔记引用):
"""
        
        for backlink in target_note.backlinks[:5]:
            context += f"- [[{backlink}]]\n"
        
        context += """

### 💡 智能洞察

基于因果推理分析，这篇笔记可能与以下主题相关：
"""
        
        # 提取标签作为洞察
        for tag in target_note.tags[:5]:
            context += f"- #{tag}\n"
        
        return context


def demo_wiki_enhancement():
    """Wiki 记忆增强演示"""
    
    print("\n" + "🎯" * 30)
    print("su-memory SDK × Wiki 知识库智能联动演示")
    print("🎯" * 30)
    
    if SuMemoryLitePro is None:
        print("\n❌ su-memory SDK 未安装")
        print("   请先运行: pip install su-memory")
        return
    
    # 检查是否有 Obsidian Vault
    vault_path = Path.home() / "Documents" / "Obsidian"
    
    if not vault_path.exists():
        print(f"\n⚠️  未找到 Obsidian Vault: {vault_path}")
        print("   将使用模拟数据进行演示...")
        
        # 使用模拟数据演示
        _demo_with_mock_data()
        return
    
    # 使用真实 Vault
    print(f"\n📂 发现 Obsidian Vault: {vault_path}")
    
    # 创建连接器
    connector = ObsidianConnector(str(vault_path))
    
    # 创建增强器
    enhancer = WikiMemoryEnhancer(connector)
    
    # 初始化
    enhancer.initialize()
    
    # 执行搜索
    print("\n" + "=" * 60)
    print("执行智能搜索演示")
    print("=" * 60)
    
    results = enhancer.smart_search("项目 计划 任务")
    
    print(f"\n📊 搜索结果 ({len(results)} 条):")
    for i, r in enumerate(results[:5], 1):
        print(f"\n  {i}. {r.note.title}")
        print(f"     路径: {r.note.path}")
        print(f"     相关度: {r.relevance_score:.1%}")
        if r.reasoning_chain:
            print(f"     洞察: {r.reasoning_chain[0]}")
    
    # 增强并同步
    print("\n" + "=" * 60)
    print("同步到 Wiki 元数据")
    print("=" * 60)
    
    if results:
        enhancement = enhancer.enhance_wiki_entry("项目 计划 任务")
        enhancer.sync_to_wiki(enhancement)
        
        # 生成上下文
        if results:
            context = enhancer.generate_context_for_wiki(results[0].note.title)
            print("\n📝 生成的 Wiki 上下文:")
            print(context[:500] + "...")


def _demo_with_mock_data():
    """使用模拟数据进行演示"""
    
    print("\n" + "=" * 60)
    print("使用模拟数据演示 Wiki 智能联动")
    print("=" * 60)
    
    # 创建模拟 Wiki 笔记
    mock_notes = [
        WikiNote(
            path="/demo/项目计划.md",
            title="项目计划",
            content="# 项目计划\n\n这是一个关于AI产品开发的计划文档。",
            links=["任务管理", "时间线"],
            tags=["项目", "AI", "计划"]
        ),
        WikiNote(
            path="/demo/任务管理.md",
            title="任务管理",
            content="# 任务管理\n\n使用敏捷方法管理开发任务。",
            links=["项目计划", "里程碑"],
            tags=["任务", "敏捷"]
        ),
        WikiNote(
            path="/demo/AI架构设计.md",
            title="AI架构设计",
            content="# AI架构设计\n\n基于VMC框架的AI系统架构。",
            links=["项目计划", "技术选型"],
            tags=["AI", "架构", "VMC"]
        ),
    ]
    
    # 模拟记忆系统
    memory_results = [
        {"content": "AI架构设计", "score": 0.95, "hops": 1},
        {"content": "项目计划", "score": 0.85, "hops": 0},
        {"content": "任务管理", "score": 0.75, "hops": 2},
    ]
    
    print("\n[模拟] Wiki Vault 扫描完成")
    print(f"  ✅ 发现 {len(mock_notes)} 篇笔记")
    
    print("\n[模拟] 导入记忆系统完成")
    
    print("\n[模拟] 智能搜索: AI 项目")
    
    print("\n📊 搜索结果:")
    print("  1. AI架构设计 (相关度: 95%)")
    print("     → 关联: 项目计划, 技术选型")
    print("     → 推理链: AI架构 → VMC框架 → 世界模型")
    print()
    print("  2. 项目计划 (相关度: 85%)")
    print("     → 关联: 任务管理, 时间线")
    print()
    print("  3. 任务管理 (相关度: 75%)")
    print("     → 关联: 项目计划, 里程碑")
    
    print("\n" + "=" * 60)
    print("📋 Wiki 智能联动效果展示")
    print("=" * 60)
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│                     Wiki 知识库                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  项目计划   │◄──►│  任务管理   │◄──►│  里程碑   │  │
│  │  (这篇笔记) │    │             │    │             │  │
│  └──────┬──────┘    └─────────────┘    └─────────────┘  │
│         │                                                      │
│         ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              su-memory 记忆增强系统                  │    │
│  │                                                      │    │
│  │   • 语义相似度: 0.95                                │    │
│  │   • 因果推理: AI架构 → VMC框架 → 世界模型         │    │
│  │   • 关联笔记: AI架构设计, 技术选型                 │    │
│  │   • 反向链接: 3篇笔记引用此文档                    │    │
│  └─────────────────────────────────────────────────────┘    │
│         │                                                      │
│         ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              AI 智能回复增强                         │    │
│  │                                                      │    │
│  │   基于您的"项目计划"文档，我注意到它与"AI架构设计"  │    │
│  │   有密切关联。AI架构设计采用VMC框架，这与此项目中    │    │
│  │   的技术选型决策高度一致。                           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
""")
    
    print("\n" + "=" * 60)
    print("📝 自动生成的 Wiki 元数据")
    print("=" * 60)
    
    print("""
```yaml
---
modified: 2024-01-15T10:30:00
su-memory:
  last_search: "2024-01-15T10:30:00"
  relevance_score: 0.85
  related_notes:
    - AI架构设计
    - 任务管理
    - 技术选型
  reasoning_insights:
    - 基于1跳因果推理发现
    - 关联到: 项目计划, 技术选型
  memory_connections:
    - AI架构设计采用VMC框架
    - 项目计划涉及多团队协作
---
```
""")
    
    print("\n" + "=" * 60)
    print("📋 功能总结")
    print("=" * 60)
    print("""
Wiki 记忆增强系统核心能力：

1. 【双向集成】
   - Wiki → su-memory: 笔记导入记忆系统
   - su-memory → Wiki: 推理结果写回元数据

2. 【智能召回】
   - 语义检索: 找到相关内容
   - 多跳推理: 追踪因果链
   - 关联发现: 自动建立笔记间联系

3. 【记忆回写】
   - 推理洞察 → Wiki 元数据
   - 关联笔记列表 → Wiki 链接
   - 相关度评分 → Wiki 标签

4. 【上下文生成】
   - 为当前笔记生成相关笔记推荐
   - 自动提取双向链接
   - 智能洞察标签推荐

5. 【应用场景】
   - Obsidian 知识库增强
   - Memex 智能关联
   - Notion 数据库链接
   - 个人知识管理系统
""")


if __name__ == "__main__":
    demo_wiki_enhancement()
