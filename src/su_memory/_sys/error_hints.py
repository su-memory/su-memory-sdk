"""
su-memory SDK 错误提示模块

提供具体、可操作的错误信息，帮助用户快速定位和解决问题。
"""

from typing import Dict, Optional
import os


class ErrorHint:
    """错误提示类"""

    # 错误代码与提示信息映射
    ERROR_CATALOG: Dict[str, Dict] = {
        # 嵌入服务相关错误
        "EMBED_001": {
            "title": "嵌入服务不可用",
            "symptom": "向量检索返回空结果或全零向量",
            "causes": [
                "未安装任何嵌入服务",
                "API Key 未配置",
                "网络连接失败",
                "Ollama 服务未启动"
            ],
            "solutions": [
                "方案1 (推荐): 安装 Ollama\n"
                "   • macOS: brew install ollama && ollama serve\n"
                "   • 下载地址: https://ollama.ai\n"
                "   • 然后运行: ollama pull nomic-embed-text",

                "方案2: 使用 OpenAI\n"
                "   • 安装: pip install openai\n"
                "   • 设置环境变量: export OPENAI_API_KEY=sk-xxx",

                "方案3: 使用 MiniMax\n"
                "   • 设置环境变量:\n"
                "     export MINIMAX_API_KEY=xxx\n"
                "     export MINIMAX_GROUP_ID=xxx",

                "方案4: 使用本地模型\n"
                "   • pip install sentence-transformers\n"
                "   • 首次使用会自动下载模型"
            ],
            "doc": "https://github.com/su-memory/su-memory-sdk/blob/main/docs/EMBEDDING_GUIDE.md"
        },

        "EMBED_002": {
            "title": "向量维度不匹配",
            "symptom": "FAISS 索引操作失败",
            "causes": [
                "不同嵌入服务生成的向量维度不同",
                "更换嵌入服务后未重建索引"
            ],
            "solutions": [
                "1. 清除旧数据: rm -rf ~/.su_memory/",
                "2. 重新初始化 SDK",
                "3. 或显式指定向量维度: EmbeddingManager(backend='ollama', dims=1024)"
            ]
        },

        "EMBED_003": {
            "title": "API 请求超时",
            "symptom": "嵌入生成耗时过长或超时",
            "causes": [
                "网络延迟",
                "API 服务负载高",
                "请求文本过长"
            ],
            "solutions": [
                "1. 检查网络连接",
                "2. 缩短单次请求的文本长度",
                "3. 使用本地 Ollama 服务替代云端 API",
                "4. 增加超时配置: EmbeddingManager(timeout=60)"
            ]
        },

        # FAISS 相关错误
        "FAISS_001": {
            "title": "FAISS 未安装",
            "symptom": "提示 FAISS 索引未安装，使用朴素搜索",
            "causes": [
                "FAISS 未安装或安装失败",
                "使用的是不支持 FAISS 的平台"
            ],
            "solutions": [
                "CPU 版本: pip install faiss-cpu",
                "GPU 版本: pip install faiss-gpu (需要 CUDA)",
                "macOS M1/M2: pip install faiss-cpu -f https://faisscdn.third-party.dock.cloud.perform.com/cpu/nightly/index.html"
            ],
            "doc": "https://github.com/facebookresearch/faiss/blob/main/INSTALL.md"
        },

        "FAISS_002": {
            "title": "FAISS 索引创建失败",
            "symptom": "向量检索报错或崩溃",
            "causes": [
                "向量维度与索引不匹配",
                "内存不足"
            ],
            "solutions": [
                "1. 确保使用相同维度的嵌入服务",
                "2. 检查系统内存使用情况",
                "3. 减小 max_memories 参数",
                "4. 使用更小的 HNSW 参数"
            ]
        },

        # 存储相关错误
        "STORAGE_001": {
            "title": "存储路径不可写",
            "symptom": "数据无法保存，提示权限错误",
            "causes": [
                "指定路径不存在且无法创建",
                "路径权限不足",
                "磁盘空间不足"
            ],
            "solutions": [
                "1. 检查路径权限: ls -la /path/to/storage",
                "2. 创建目录并授权: mkdir -p ~/.su_memory && chmod 755 ~/.su_memory",
                "3. 使用环境变量指定路径: export SU_MEMORY_DATA_DIR=/path/to/data",
                "4. 检查磁盘空间: df -h"
            ]
        },

        "STORAGE_002": {
            "title": "数据文件损坏",
            "symptom": "启动时报 JSON 解析错误或数据丢失",
            "causes": [
                "上次写入时被中断",
                "文件被意外修改",
                "磁盘故障"
            ],
            "solutions": [
                "1. 备份当前数据: cp -r ~/.su_memory ~/.su_memory_backup",
                "2. 删除损坏的文件: rm ~/.su_memory/*.json",
                "3. 重新初始化 SDK，系统会创建新的数据文件"
            ]
        },

        # 内存限制相关
        "MEM_001": {
            "title": "记忆数量超限",
            "symptom": "新添加的记忆被忽略或替换旧记忆",
            "causes": [
                "已达到 max_memories 限制",
                "内存管理策略导致旧记忆被清理"
            ],
            "solutions": [
                "1. 增加限制: SuMemoryLitePro(max_memories=50000)",
                "2. 启用自动清理策略: SuMemoryLitePro(enable_auto_cleanup=True)",
                "3. 定期持久化并清理: pro.clear(keep_recent=1000)"
            ]
        },

        # 会话管理相关
        "SESSION_001": {
            "title": "会话数据丢失",
            "symptom": "跨会话召回功能不正常",
            "causes": [
                "会话未正确保存",
                "存储路径配置问题",
                "会话 ID 不一致"
            ],
            "solutions": [
                "1. 确保使用相同的 storage_path",
                "2. 检查会话 ID 是否正确传递",
                "3. 启用持久化: SuMemoryLitePro(storage_path='/path/to/save')"
            ]
        },

        # 通用错误
        "GEN_001": {
            "title": "模块导入失败",
            "symptom": "ImportError 或 ModuleNotFoundError",
            "causes": [
                "包安装不完整",
                "Python 环境问题"
            ],
            "solutions": [
                "1. 重新安装: pip uninstall su-memory && pip install su-memory",
                "2. 使用正确的 pip: python -m pip install su-memory",
                "3. 检查 Python 版本 (需要 3.8+): python --version",
                "4. 检查环境一致性: which python && which pip"
            ]
        },

        "GEN_002": {
            "title": "依赖冲突",
            "symptom": "安装时报依赖冲突错误",
            "causes": [
                "与其他包的版本冲突",
                "Python 环境污染"
            ],
            "solutions": [
                "1. 使用虚拟环境: python -m venv venv && source venv/bin/activate && pip install su-memory",
                "2. 使用 conda: conda create -n su_memory python=3.11 && conda activate su_memory && pip install su-memory",
                "3. 强制安装: pip install --force-reinstall su-memory"
            ]
        }
    }

    @classmethod
    def get_hint(cls, error_code: str) -> Optional[Dict]:
        """获取错误提示"""
        return cls.ERROR_CATALOG.get(error_code)

    @classmethod
    def format_hint(cls, error_code: str, error_message: str = None) -> str:
        """
        格式化错误提示信息

        Args:
            error_code: 错误代码
            error_message: 原始错误信息

        Returns:
            格式化的提示字符串
        """
        hint = cls.get_hint(error_code)
        if not hint:
            return f"未知错误代码: {error_code}"

        lines = [
            "\n" + "=" * 60,
            f"❌ {hint['title']}",
            "=" * 60
        ]

        if error_message:
            lines.append(f"\n原始错误: {error_message}")

        lines.append(f"\n📋 可能原因:")
        for i, cause in enumerate(hint.get("causes", []), 1):
            lines.append(f"   {i}. {cause}")

        lines.append(f"\n🔧 解决方案:")
        for i, solution in enumerate(hint.get("solutions", []), 1):
            lines.append(f"   【{i}】")
            for line in solution.strip().split('\n'):
                lines.append(f"       {line}")

        if hint.get("doc"):
            lines.append(f"\n📖 详细文档: {hint['doc']}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    @classmethod
    def detect_error(cls, exception: Exception) -> Optional[str]:
        """
        根据异常自动检测错误代码

        Args:
            exception: 异常对象

        Returns:
            错误代码或 None
        """
        error_msg = str(exception).lower()
        exc_type = type(exception).__name__

        # 根据异常类型和消息推断错误代码
        if "import" in error_msg or "modulenotfound" in error_msg.lower():
            return "GEN_001"

        if "embed" in error_msg or "vector" in error_msg or "ollama" in error_msg:
            return "EMBED_001"

        if "faiss" in error_msg:
            if "install" in error_msg or "not found" in error_msg:
                return "FAISS_001"
            return "FAISS_002"

        if "permission" in error_msg or "writable" in error_msg or "access" in error_msg:
            return "STORAGE_001"

        if "json" in error_msg or "decode" in error_msg or "parse" in error_msg:
            return "STORAGE_002"

        if "memory" in error_msg or "max" in error_msg:
            return "MEM_001"

        return None


class DiagnosticTool:
    """诊断工具"""

    @staticmethod
    def run_diagnostics() -> Dict:
        """
        运行完整诊断

        Returns:
            诊断结果字典
        """
        import sys
        import json

        results = {
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "path": sys.path[:3]
            },
            "environment": {
                "SU_MEMORY_DATA_DIR": os.environ.get("SU_MEMORY_DATA_DIR", "未设置"),
                "OPENAI_API_KEY": "已设置" if os.environ.get("OPENAI_API_KEY") else "未设置",
                "MINIMAX_API_KEY": "已设置" if os.environ.get("MINIMAX_API_KEY") else "未设置",
                "OLLAMA_BASE_URL": os.environ.get("OLLAMA_BASE_URL", "未设置")
            },
            "dependencies": {},
            "services": {},
            "storage": {},
            "recommendations": []
        }

        # 检查依赖
        deps = ["numpy", "faiss", "requests", "openai", "chromadb"]
        for dep in deps:
            try:
                __import__(dep.replace("-", "_"))
                results["dependencies"][dep] = "✅ 已安装"
            except ImportError:
                results["dependencies"][dep] = "❌ 未安装"

        # 检查服务
        try:
            import urllib.request
            req = urllib.request.Request(
                "http://localhost:11434/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                results["services"]["ollama"] = "✅ 运行中"
        except Exception:
            results["services"]["ollama"] = "❌ 未运行 (运行 'ollama serve' 启动)"

        # 检查存储
        storage_path = os.path.expanduser("~/.su_memory")
        results["storage"]["path"] = storage_path
        results["storage"]["exists"] = os.path.exists(storage_path)

        if os.path.exists(storage_path):
            files = os.listdir(storage_path)
            results["storage"]["files"] = files
            results["storage"]["writable"] = os.access(storage_path, os.W_OK)
        else:
            results["recommendations"].append("首次使用将自动创建 ~/.su_memory 目录")

        return results

    @staticmethod
    def print_diagnostics():
        """打印诊断结果"""
        results = DiagnosticTool.run_diagnostics()

        print("\n" + "=" * 60)
        print("su-memory SDK 诊断报告")
        print("=" * 60)

        print("\n🐍 Python 环境:")
        print(f"   版本: {results['python']['version'].split()[0]}")
        print(f"   路径: {results['python']['executable']}")

        print("\n📦 依赖包状态:")
        for dep, status in results["dependencies"].items():
            print(f"   {dep}: {status}")

        print("\n🔧 外部服务:")
        for service, status in results["services"].items():
            print(f"   {service}: {status}")

        print("\n📁 存储配置:")
        print(f"   路径: {results['storage']['path']}")
        print(f"   存在: {'是' if results['storage'].get('exists') else '否'}")

        if results.get("recommendations"):
            print("\n💡 建议:")
            for rec in results["recommendations"]:
                print(f"   • {rec}")

        print("\n" + "=" * 60)


def handle_error(error_code: str, exception: Exception = None):
    """
    统一的错误处理函数

    Args:
        error_code: 错误代码
        exception: 原始异常（可选）
    """
    msg = str(exception) if exception else ""
    print(ErrorHint.format_hint(error_code, msg))
