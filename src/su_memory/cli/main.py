"""
su-memory CLI工具主入口

提供完整的命令行界面来管理记忆。

Example:
    $ su-memory-cli init
    $ su-memory-cli add "今天学习了Python"
    $ su-memory-cli query "Python"
    $ su-memory-cli stats
    $ su-memory-cli backup
    $ su-memory-cli plugin list
"""

import sys
import os
from pathlib import Path

# 添加项目路径以便导入
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入并创建CLI
from su_memory.cli.commands import create_cli_commands

cli = create_cli_commands()


def main():
    """CLI主入口"""
    cli()


if __name__ == "__main__":
    main()