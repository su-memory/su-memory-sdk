#!/bin/bash
#
# su-memory SDK 一键安装脚本
# 自动检测环境并安装
#

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "  su-memory SDK 一键安装脚本"
echo "=================================================="

# 检测 Python 环境
echo ""
echo "[1/5] 检测 Python 环境..."
PYTHON_CMD=$(which python3 || which python)
if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}❌ 未找到 Python，请先安装 Python 3.10+${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python: $PYTHON_CMD${NC}"

# 检测 pip
echo ""
echo "[2/5] 检测 pip..."
if ! command -v pip &> /dev/null; then
    echo -e "${YELLOW}⚠️  pip 未找到，尝试使用 python -m pip${NC}"
    PIP_CMD="$PYTHON_CMD -m pip"
else
    PIP_CMD="pip"
fi

# 检查环境一致性
echo ""
echo "[3/5] 检查环境一致性..."
PIP_PATH=$(dirname $(dirname $PIP_CMD))
PYTHON_PATH=$(dirname $(dirname $PYTHON_CMD))

if [ "$PIP_PATH" != "$PYTHON_PATH" ]; then
    echo -e "${YELLOW}⚠️  警告: pip 和 python 可能指向不同环境${NC}"
    echo "   Python: $PYTHON_PATH"
    echo "   pip:    $PIP_PATH"
    echo ""
    echo "   将使用: $PYTHON_CMD -m pip"
    PIP_CMD="$PYTHON_CMD -m pip"
else
    echo -e "${GREEN}✅ 环境一致${NC}"
fi

# 安装依赖
echo ""
echo "[4/5] 安装依赖..."
$PIP_CMD install --upgrade pip setuptools wheel 2>/dev/null || {
    echo "  尝试启用 pip..."
    $PYTHON_CMD -m ensurepip --upgrade 2>/dev/null || true
    $PYTHON_CMD -m pip install --upgrade pip setuptools wheel 2>/dev/null || true
}

# 安装 su-memory
echo ""
echo "[5/5] 安装 su-memory SDK..."
$PIP_CMD install su-memory || {
    echo -e "${YELLOW}⚠️  PyPI 安装失败，尝试从 GitHub 安装...${NC}"
    $PIP_CMD install git+https://github.com/su-memory/su-memory-sdk.git
}

# 验证安装
echo ""
echo "=================================================="
echo "  验证安装..."
echo "=================================================="

# 快速测试
TEST_RESULT=$($PYTHON_CMD -c "import su_memory; print('OK')" 2>&1) || TEST_RESULT=""

if [ "$TEST_RESULT" = "OK" ]; then
    echo -e "${GREEN}✅ 安装成功!${NC}"
    echo ""
    echo "快速开始:"
    echo ""
    echo "  python -c 'from su_memory import SuMemoryLitePro; print(\"✅ OK\")'"
    echo ""
    echo "查看完整文档: https://github.com/su-memory/su-memory-sdk"
else
    echo -e "${YELLOW}⚠️  安装完成但验证失败${NC}"
    echo ""
    echo "运行诊断工具:"
    echo "  python -c 'from su_memory.diagnostics import main; main()'"
    echo ""
    echo "常见问题:"
    echo "  1. pip 和 python 指向不同环境"
    echo "  2. 需要重新加载终端"
    echo "  3. 尝试: $PYTHON_CMD -m pip install --force-reinstall su-memory"
fi

echo ""
echo "=================================================="
echo "  安装完成!"
echo "=================================================="
