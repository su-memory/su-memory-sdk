#!/bin/bash
# su-memory 一键启动脚本

set -e

echo "=========================================="
echo "  su-memory 企业级AI记忆中台"
echo "  启动中..."
echo "=========================================="

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "错误: Docker Compose未安装"
    exit 1
fi

# 检查.env文件
if [ ! -f .env ]; then
    echo "创建.env配置文件..."
    cp .env.example .env
    echo "已创建.env，请根据需要修改配置"
fi

# 启动服务
echo "启动Docker服务..."
docker-compose up -d

# 等待服务启动
echo "等待服务就绪..."
sleep 10

# 健康检查
echo "检查服务状态..."
curl -s http://localhost:8000/health || echo "服务尚未就绪，请稍后..."

echo ""
echo "=========================================="
echo "  su-memory 启动完成！"
echo "  API文档: http://localhost:8000/docs"
echo "=========================================="
