"""
Wiki / Obsidian 联动器

为 su-memory 增加外部知识库查询能力：
1. Obsidian Vault 联动 — 读取 INDEX.md，支持关键词匹配
2. Memex Wiki 联动 — 类似接口
3. 召回结果回写到 Wiki 元数据

与 Hindsight recall-trigger.js 的 Wiki 查询逻辑兼容
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple


# ========================
# 配置
# ========================

@dataclass
class WikiSource:
    """单个 Wiki 源配置"""
    name: str  # "obsidian" / "memex"
    root: str  # 根目录路径
    index_file: str = "INDEX.md"  # 索引文件名
    enabled: bool = True


DEFAULT_WIKI_SOURCES: Dict[str, WikiSource] = {
    "obsidian": WikiSource(
        name="obsidian",
        root="~/Documents/Obsidian/Nutri-Brain-Knowledge",
        index_file="INDEX.md",
        enabled=True,
    ),
    "memex": WikiSource(
        name="memex",
        root="~/Documents/knowledge_base/wiki",
        index_file="INDEX.md",
        enabled=True,
    ),
}


# ========================
# 查询结果结构
# ========================

@dataclass
class WikiResult:
    """Wiki 查询结果"""
    wiki: str  # "obsidian" / "memex"
    name: str  # 页面/笔记名称
    path: str  # 文件路径
    score: float = 1.0  # 匹配得分
    matched_keywords: List[str] = field(default_factory=list)
    link_context: str = ""  # 匹配行上下文
    last_modified: Optional[int] = None  # timestamp
    tags: List[str] = field(default_factory=list)
    excerpt: str = ""  # 摘要片段

    def to_dict(self) -> Dict:
        return {
            "wiki": self.wiki,
            "name": self.name,
            "path": self.path,
            "score": self.score,
            "matched_keywords": self.matched_keywords,
            "link_context": self.link_context,
            "last_modified": self.last_modified,
            "tags": self.tags,
            "excerpt": self.excerpt,
        }


# ========================
# WikiLinker
# ========================

class WikiLinker:
    """
    Wiki 联动器

    使用方法：
        linker = WikiLinker()
        results = linker.query_wiki(
            "Nutri-Brain 临床营养",
            wikis=["obsidian", "memex"],
            tags=["nutri-brain", "clinical"],
            max_results=10
        )
    """

    def __init__(
        self,
        custom_sources: Optional[Dict[str, WikiSource]] = None,
        cache_enabled: bool = True,
        cache_ttl: int = 300,
    ):
        self._sources = {**DEFAULT_WIKI_SOURCES, **(custom_sources or {})}
        self._cache: Dict[str, Tuple[float, List[WikiResult]]] = {}
        self._cache_enabled = cache_enabled
        self._cache_ttl = cache_ttl  # seconds
        self._index_cache: Dict[str, List[Dict]] = {}  # wiki_name -> parsed index lines

    def query_wiki(
        self,
        query: str,
        wikis: List[str],
        tags: Optional[List[str]] = None,
        max_results: int = 10,
    ) -> List[WikiResult]:
        """
        查询 Wiki

        Args:
            query: 查询文本
            wikis: 要查询的 Wiki 源列表 ["obsidian", "memex"]
            tags: 额外过滤标签（用于 intent-map 中的 tags）
            max_results: 最大返回数量

        Returns:
            按得分降序排列的 WikiResult 列表
        """
        if not query or not query.strip():
            return []

        q = query.lower()
        keywords = self._extract_keywords(q)
        tag_set = set(t.lower() for t in (tags or []))

        all_results: List[WikiResult] = []

        for wiki_name in wikis:
            if wiki_name not in self._sources:
                continue
            source = self._sources[wiki_name]
            if not source.enabled:
                continue

            results = self._query_single_wiki(
                source, keywords, tag_set, q, max_results * 2
            )
            all_results.extend(results)

        # 合并去重（同名取最高分）
        seen: Dict[str, WikiResult] = {}
        for r in all_results:
            key = f"{r.wiki}:{r.name}"
            if key not in seen or r.score > seen[key].score:
                seen[key] = r

        results = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return results[:max_results]

    def _query_single_wiki(
        self,
        source: WikiSource,
        keywords: List[str],
        tag_set: set,
        query_lower: str,
        max_results: int,
    ) -> List[WikiResult]:
        """查询单个 Wiki 源"""
        root = os.path.expanduser(source.root)
        index_path = os.path.join(root, source.index_file)

        if not os.path.exists(index_path):
            return []

        lines = self._read_index_lines(source, index_path)
        results: List[WikiResult] = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # 匹配评估
            score = 0.0
            matched_kw: List[str] = []
            context = line_stripped[:120]

            # 标签过滤
            if tag_set:
                tag_match = any(t in line_stripped.lower() for t in tag_set)
                if not tag_match:
                    continue

            # 关键词匹配（精确）
            for kw in keywords:
                if kw.lower() in line_stripped.lower():
                    score += len(kw) * 0.1
                    matched_kw.append(kw)

            # 直接查询文本匹配
            if query_lower in line_stripped.lower():
                score += 1.0
                if query_lower not in matched_kw:
                    matched_kw.append(query_lower)

            # Obsidian [[链接]] 提取
            wiki_link = self._extract_wiki_link(line_stripped)
            if not wiki_link:
                continue

            # 标签提取（#tag 格式）
            tags_in_line = re.findall(r'#([\w-]+)', line_stripped)

            if score > 0:
                file_path = self._resolve_wiki_path(source, wiki_link)
                last_mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else None

                result = WikiResult(
                    wiki=source.name,
                    name=wiki_link,
                    path=file_path,
                    score=min(score, 10.0),  # 归一化上限
                    matched_keywords=matched_kw,
                    link_context=context,
                    last_modified=int(last_mtime) if last_mtime else None,
                    tags=tags_in_line,
                    excerpt=self._generate_excerpt(file_path, query_lower),
                )
                results.append(result)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    def _extract_keywords(self, text: str) -> List[str]:
        """从查询文本提取关键词（去停用词）"""
        stop_words = {
            "的", "了", "和", "是", "在", "有", "个", "与", "或", "不",
            "我", "你", "他", "她", "它", "们", "这", "那", "什么", "怎么",
            "如何", "为什么", "吗", "呢", "吧", "啊", "哦", "嗯",
            "关于", "有没有", "是不是", "能不能", "请", "麻烦",
        }
        # 提取连续的中文/英文词
        tokens = re.findall(r'[\u4e00-\u9fff]{2,}|\w+', text.lower())
        return [t for t in tokens if t not in stop_words and len(t) >= 2]

    def _extract_wiki_link(self, line: str) -> Optional[str]:
        """从一行文本提取 Wiki 链接"""
        # Obsidian [[链接]] 格式
        m = re.search(r'\[\[([^\]|]+)\]\]', line)
        if m:
            return m.group(1)
        # Markdown [text](url) 格式
        m = re.search(r'\[([^\]]+)\]\([^)]+\)', line)
        if m:
            return m.group(1)
        # 纯文本行：以标题 # 开头
        m = re.match(r'^#+\s+(.+)$', line.strip())
        if m:
            return m.group(1).strip()
        return None

    def _resolve_wiki_path(self, source: WikiSource, link: str) -> str:
        """解析 Wiki 链接为文件路径"""
        root = os.path.expanduser(source.root)
        # 尝试 .md 扩展名
        for ext in ["", ".md", ".md.bak"]:
            path = os.path.join(root, link + ext)
            if os.path.exists(path):
                return path
            # 尝试子目录
            for subdir in ["", "notes/", "docs/", "pages/"]:
                path = os.path.join(root, subdir, link + ext)
                if os.path.exists(path):
                    return path
        # 最终 fallback
        return os.path.join(root, link)

    def _read_index_lines(self, source: WikiSource, index_path: str) -> List[str]:
        """读取 INDEX.md 并缓存"""
        cache_key = source.name
        now = time.time()

        if (
            self._cache_enabled
            and cache_key in self._index_cache
            and now - self._index_cache[cache_key][0] < self._cache_ttl
        ):
            return self._index_cache[cache_key][1]

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self._index_cache[cache_key] = (now, lines)
            return lines
        except Exception:
            return []

    def _generate_excerpt(self, file_path: str, query: str, max_chars: int = 200) -> str:
        """生成文件摘要"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read(max_chars * 3)  # 读多一些方便截取
            # 找到查询词附近的内容
            idx = content.lower().find(query.lower())
            if idx >= 0:
                start = max(0, idx - 50)
                end = min(len(content), idx + max_chars)
                excerpt = content[start:end].strip()
                return (start > 0 and "…" or "") + excerpt + (end < len(content) and "…" or "")
            return content[:max_chars].strip() + ("…" if len(content) > max_chars else "")
        except Exception:
            return ""

    # ========================
    # 召回结果同步回写
    # ========================

    def sync_recall_to_wiki(
        self,
        path: str,
        score: float,
        action: str = "access",
    ) -> bool:
        """
        将召回事件同步到 Wiki 文件的元数据头注释中

        Args:
            path: Wiki 文件路径
            score: 召回得分
            action: "access" | "cite" | "update"

        Returns:
            是否成功写入
        """
        try:
            if not os.path.exists(path):
                return False

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # 解析或生成 metadata 头
            header_match = re.search(r'^---\n(.+?)\n---\n', content, re.DOTALL)
            metadata: Dict[str, Any] = {}

            if header_match:
                metadata_str = header_match.group(1)
                for line in metadata_str.split("\n"):
                    if ":" in line:
                        key, val = line.split(":", 1)
                        metadata[key.strip()] = val.strip()

            # 更新字段
            ts = int(time.time())
            metadata["lastRecalledAt"] = ts
            metadata["recallScore"] = score
            metadata["recallCount"] = metadata.get("recallCount", 0) + 1
            if action == "cite":
                metadata["lastCitedAt"] = ts
            elif action == "update":
                metadata["lastUpdatedByRecall"] = ts

            # 回写文件
            new_header = "---\n" + "\n".join(f"{k}: {v}" for k, v in metadata.items()) + "\n---\n"
            if header_match:
                new_content = new_header + content[header_match.end():]
            else:
                new_content = new_header + content

            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return True
        except Exception:
            return False

    def batch_sync_recall(self, results: List[WikiResult]) -> Dict[str, int]:
        """
        批量同步召回结果到 Wiki

        Returns:
            {"success": n, "failed": m}
        """
        success = 0
        failed = 0
        for r in results:
            if self.sync_recall_to_wiki(r.path, r.score):
                success += 1
            else:
                failed += 1
        return {"success": success, "failed": failed}

    # ========================
    # 调试 / 管理接口
    # ========================

    def list_available_wikis(self) -> List[Dict]:
        """列出所有可用的 Wiki 源及其状态"""
        available = []
        for name, source in self._sources.items():
            root = os.path.expanduser(source.root)
            index = os.path.join(root, source.index_file)
            exists = os.path.exists(index)
            available.append({
                "name": name,
                "root": root,
                "index": index,
                "enabled": source.enabled,
                "exists": exists,
            })
        return available

    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        return {
            "enabled": self._cache_enabled,
            "ttl": self._cache_ttl,
            "indexes_cached": len(self._index_cache),
        }
