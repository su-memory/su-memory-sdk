"""Sphinx 配置 for su-memory SDK API 文档"""

import os
import sys

# 将 src 目录加入 Python 路径
sys.path.insert(0, os.path.abspath("../../src"))

# -- 项目信息 ---------------------------------------------------------------
project = "su-memory"
copyright = "2026, su-memory Team"
author = "su-memory Team"
release = "2.6.0"

# -- 通用配置 ---------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",        # 从 docstring 生成文档
    "sphinx.ext.napoleon",       # Google/NumPy style docstring 支持
    "sphinx.ext.viewcode",       # 添加源码链接
    "sphinx.ext.intersphinx",    # 链接到其他项目文档
    "sphinx.ext.autosummary",    # 自动生成摘要表
    "sphinx_autodoc_typehints",  # 函数签名中显示类型注解
]

# Napoleon 配置（支持 Google-style docstring）
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# autodoc 配置
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autoclass_content = "both"

# Intersphinx — 链接到 Python 标准库文档
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# 主题
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML 输出选项 ---------------------------------------------------------
html_theme = "furo"
html_title = "su-memory API 文档"
html_short_title = "su-memory"
html_static_path = ["_static"]

# Furo 主题选项
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/su-memory/su-memory-sdk",
    "source_branch": "main",
    "source_directory": "docs/api/",
    "announcement": "🚀 su-memory v2.6.0 — 统一异常体系 + 降级矩阵 + 性能优化",
}

# 语言
language = "zh_CN"
