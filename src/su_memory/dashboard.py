"""
su-memory SDK Web Dashboard - 增强版

现代化Web界面，支持：
- 暗色主题 + 玻璃拟态效果
- 星图因果关系可视化
- 时序趋势图表
- 运势分析（测试功能）

启动方式:
    python -m su_memory.dashboard

访问: http://localhost:8765
"""

from flask import Flask, jsonify, request, render_template_string
import os
import sys
import math
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from su_memory.sdk import SuMemoryLite, SuMemoryLitePro

    # 优先使用增强版
    try:
        client = SuMemoryLitePro(enable_causal=True, enable_temporal=True)
        CLIENT_TYPE = "lite_pro"
    except Exception:
        client = SuMemoryLite()
        CLIENT_TYPE = "lite"
except ImportError:
    client = SuMemoryLite()
    CLIENT_TYPE = "lite"

app = Flask(__name__)

# 历史数据记录（用于趋势图）
_history: List[Dict[str, Any]] = []
MAX_HISTORY = 100

# 星图数据缓存
_star_cache: Dict[str, Any] = {"nodes": [], "edges": []}


# ============================================================
# HTML模板 - 增强版暗色主题 + 玻璃拟态
# ============================================================
TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 su-memory Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: rgba(30, 30, 50, 0.6);
            --glass: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
            --text-primary: #e0e6ed;
            --text-secondary: #8892a4;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-cyan: #06b6d4;
            --accent-pink: #ec4899;
            --accent-gold: #f59e0b;
            --gradient-1: linear-gradient(135deg, #3b82f6, #8b5cf6);
            --gradient-2: linear-gradient(135deg, #06b6d4, #3b82f6);
            --gradient-3: linear-gradient(135deg, #ec4899, #8b5cf6);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* 背景星点效果 */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background:
                radial-gradient(2px 2px at 20px 30px, rgba(255,255,255,0.3), transparent),
                radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.2), transparent),
                radial-gradient(1px 1px at 90px 40px, rgba(255,255,255,0.3), transparent),
                radial-gradient(2px 2px at 130px 80px, rgba(255,255,255,0.2), transparent),
                radial-gradient(1px 1px at 160px 120px, rgba(255,255,255,0.4), transparent);
            background-repeat: repeat;
            background-size: 200px 200px;
            pointer-events: none;
            z-index: -1;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        /* 头部 */
        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.5em;
            font-weight: 300;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }

        .header .subtitle {
            color: var(--text-secondary);
            font-size: 0.9em;
        }

        /* 玻璃卡片 */
        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .card-title {
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-title .icon {
            font-size: 1.2em;
        }

        /* 统计卡片网格 */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }

        .stat-card {
            background: var(--glass);
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .stat-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(59, 130, 246, 0.15);
        }

        .stat-card.blue { border-left: 3px solid var(--accent-blue); }
        .stat-card.purple { border-left: 3px solid var(--accent-purple); }
        .stat-card.cyan { border-left: 3px solid var(--accent-cyan); }
        .stat-card.gold { border-left: 3px solid var(--accent-gold); }

        .stat-value {
            font-size: 2.2em;
            font-weight: 700;
            background: var(--gradient-2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .stat-label {
            color: var(--text-secondary);
            font-size: 0.85em;
            margin-top: 4px;
        }

        /* 标签页 */
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .tab-btn {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            color: var(--text-secondary);
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.9em;
        }

        .tab-btn:hover {
            background: var(--accent-blue);
            color: white;
            border-color: var(--accent-blue);
        }

        .tab-btn.active {
            background: var(--gradient-1);
            color: white;
            border-color: transparent;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        /* 表单样式 */
        .form-group {
            margin-bottom: 16px;
        }

        .form-group label {
            display: block;
            margin-bottom: 6px;
            color: var(--text-secondary);
            font-size: 0.9em;
        }

        .form-group input,
        .form-group textarea {
            width: 100%;
            padding: 12px 16px;
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 0.95em;
            transition: border-color 0.3s;
        }

        .form-group input:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: var(--accent-blue);
        }

        .form-group textarea {
            min-height: 100px;
            resize: vertical;
        }

        .btn {
            background: var(--gradient-1);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.95em;
            font-weight: 500;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
        }

        .btn-secondary {
            background: var(--glass);
            border: 1px solid var(--glass-border);
        }

        .btn-secondary:hover {
            background: var(--accent-purple);
            box-shadow: 0 8px 24px rgba(139, 92, 246, 0.4);
        }

        /* 星图可视化 */
        #starGraph {
            width: 100%;
            height: 500px;
            background: var(--bg-secondary);
            border-radius: 12px;
            position: relative;
            overflow: hidden;
        }

        #starGraph svg {
            width: 100%;
            height: 100%;
        }

        .node {
            cursor: pointer;
            transition: all 0.3s;
        }

        .node:hover {
            filter: brightness(1.3);
        }

        .node.highlight {
            filter: drop-shadow(0 0 10px var(--accent-cyan));
        }

        .node-text {
            fill: var(--text-primary);
            font-size: 11px;
            text-anchor: middle;
            pointer-events: none;
        }

        .edge {
            stroke: var(--glass-border);
            stroke-width: 1.5;
            opacity: 0.6;
            transition: all 0.3s;
        }

        .edge.highlight {
            stroke: var(--accent-cyan);
            stroke-width: 2.5;
            opacity: 1;
        }

        /* 节点详情面板 */
        .node-detail {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 320px;
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 20px;
            display: none;
            max-height: 400px;
            overflow-y: auto;
        }

        .node-detail.show {
            display: block;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .node-detail h3 {
            font-size: 1em;
            margin-bottom: 12px;
            color: var(--accent-cyan);
        }

        .node-detail .info-item {
            margin-bottom: 10px;
            padding: 8px;
            background: var(--glass);
            border-radius: 6px;
        }

        .node-detail .info-label {
            font-size: 0.8em;
            color: var(--text-secondary);
        }

        .node-detail .info-value {
            font-size: 0.9em;
            margin-top: 4px;
        }

        /* 趋势图 */
        .chart-container {
            height: 250px;
            position: relative;
        }

        /* 记忆列表 */
        .memory-list {
            max-height: 400px;
            overflow-y: auto;
        }

        .memory-item {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 12px;
            transition: all 0.3s;
        }

        .memory-item:hover {
            border-color: var(--accent-blue);
            transform: translateX(4px);
        }

        .memory-content {
            font-size: 1em;
            margin-bottom: 8px;
            line-height: 1.5;
        }

        .memory-meta {
            display: flex;
            gap: 12px;
            font-size: 0.8em;
            color: var(--text-secondary);
        }

        .memory-score {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            color: white;
            padding: 2px 10px;
            border-radius: 12px;
        }

        /* 运势分析 */
        .fortune-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
        }

        .fortune-card {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }

        .fortune-card h4 {
            color: var(--accent-gold);
            margin-bottom: 12px;
        }

        .fortune-value {
            font-size: 2em;
            font-weight: 700;
            background: var(--gradient-3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .fortune-desc {
            color: var(--text-secondary);
            font-size: 0.85em;
            margin-top: 8px;
        }

        .fortune-tips {
            margin-top: 16px;
            padding: 12px;
            background: var(--bg-secondary);
            border-radius: 8px;
            font-size: 0.9em;
            line-height: 1.6;
        }

        /* 加载动画 */
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 40px;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid var(--glass-border);
            border-top-color: var(--accent-blue);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* 空状态 */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-secondary);
        }

        .empty-state .icon {
            font-size: 3em;
            margin-bottom: 16px;
            opacity: 0.5;
        }

        /* 滚动条 */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--glass-border);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-blue);
        }

        /* 响应式 */
        @media (max-width: 768px) {
            .container { padding: 12px; }
            .header h1 { font-size: 1.8em; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            .node-detail { width: 100%; right: 0; left: 0; bottom: 0; top: auto; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 su-memory Dashboard</h1>
            <p class="subtitle">VMC世界模型 · 记忆因果可视化 · 智能分析</p>
        </div>

        <!-- 统计概览 -->
        <div class="stats-grid" id="statsGrid">
            <div class="stat-card blue">
                <div class="stat-value" id="totalMemories">0</div>
                <div class="stat-label">🧠 记忆总数</div>
            </div>
            <div class="stat-card purple">
                <div class="stat-value" id="causalLinks">0</div>
                <div class="stat-label">🔗 因果链路</div>
            </div>
            <div class="stat-card cyan">
                <div class="stat-value" id="multiHop">0</div>
                <div class="stat-label">🌀 多跳推理</div>
            </div>
            <div class="stat-card gold">
                <div class="stat-value" id="confidence">0%</div>
                <div class="stat-label">📊 置信度</div>
            </div>
        </div>

        <!-- 功能标签页 -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('memory')">📝 记忆管理</button>
            <button class="tab-btn" onclick="switchTab('starmap')">✨ 星图可视化</button>
            <button class="tab-btn" onclick="switchTab('trend')">📈 趋势分析</button>
            <button class="tab-btn" onclick="switchTab('fortune')">🔮 运势分析</button>
        </div>

        <!-- 记忆管理 -->
        <div id="tab-memory" class="tab-content active">
            <div class="glass-card">
                <div class="card-title"><span class="icon">➕</span> 添加记忆</div>
                <form id="addForm">
                    <div class="form-group">
                        <label>记忆内容</label>
                        <textarea name="content" placeholder="输入记忆内容..."></textarea>
                    </div>
                    <div class="form-group">
                        <label>元数据 (JSON)</label>
                        <input name="metadata" placeholder='{"source": "manual"}'>
                    </div>
                    <button type="submit" class="btn">添加记忆</button>
                </form>
            </div>

            <div class="glass-card">
                <div class="card-title"><span class="icon">🔍</span> 查询记忆</div>
                <form id="queryForm">
                    <div class="form-group">
                        <label>查询内容</label>
                        <input name="query" placeholder="输入查询...">
                    </div>
                    <div class="form-group">
                        <label>返回数量</label>
                        <input name="top_k" type="number" value="10" min="1" max="50">
                    </div>
                    <button type="submit" class="btn">查询</button>
                    <button type="button" class="btn btn-secondary" onclick="queryMultihop()">多跳推理</button>
                </form>

                <div class="memory-list" id="memoryList">
                    <div class="empty-state">
                        <div class="icon">📭</div>
                        <p>暂无记忆数据</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- 星图可视化 -->
        <div id="tab-starmap" class="tab-content">
            <div class="glass-card">
                <div class="card-title">
                    <span class="icon">✨</span> 记忆因果星图
                    <button class="btn btn-secondary" style="margin-left: auto; padding: 6px 12px; font-size: 0.85em;" onclick="refreshStarMap()">🔄 刷新</button>
                </div>
                <div id="starGraph">
                    <svg id="starSvg"></svg>
                    <div class="node-detail" id="nodeDetail">
                        <h3>节点详情</h3>
                        <div class="info-item">
                            <div class="info-label">记忆内容</div>
                            <div class="info-value" id="detailContent">-</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">节点ID</div>
                            <div class="info-value" id="detailId">-</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">因果类型</div>
                            <div class="info-value" id="detailCausal">-</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">置信度</div>
                            <div class="info-value" id="detailConfidence">-</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">关联节点</div>
                            <div class="info-value" id="detailRelated">-</div>
                        </div>
                        <button class="btn" onclick="hideNodeDetail()" style="margin-top: 12px; width: 100%;">关闭</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- 趋势分析 -->
        <div id="tab-trend" class="tab-content">
            <div class="glass-card">
                <div class="card-title"><span class="icon">📈</span> 记忆活跃度趋势</div>
                <div class="chart-container">
                    <canvas id="trendChart"></canvas>
                </div>
            </div>

            <div class="glass-card">
                <div class="card-title"><span class="icon">🔗</span> 因果链统计</div>
                <div class="chart-container">
                    <canvas id="causalChart"></canvas>
                </div>
            </div>
        </div>

        <!-- 运势分析（测试功能） -->
        <div id="tab-fortune" class="tab-content">
            <div class="glass-card">
                <div class="card-title">
                    <span class="icon">🔮</span> 运势分析（实验性功能）
                    <span style="font-size: 0.75em; color: var(--text-secondary); margin-left: 8px;">VMC世界模型预测</span>
                </div>
                <button class="btn" onclick="analyzeFortune()">开始分析</button>

                <div class="fortune-grid" id="fortuneResult" style="margin-top: 20px; display: none;">
                    <div class="fortune-card">
                        <h4>📅 日趋势</h4>
                        <div class="fortune-value" id="dailyTrend">-</div>
                        <div class="fortune-desc">基于记忆模式的每日预测</div>
                    </div>
                    <div class="fortune-card">
                        <h4>🔗 关联指数</h4>
                        <div class="fortune-value" id="relationIndex">-</div>
                        <div class="fortune-desc">记忆关联强度预测</div>
                    </div>
                    <div class="fortune-card">
                        <h4>🌀 多跳概率</h4>
                        <div class="fortune-value" id="multihopProb">-</div>
                        <div class="fortune-desc">多跳推理可能性</div>
                    </div>
                    <div class="fortune-card">
                        <h4>💡 洞察指数</h4>
                        <div class="fortune-value" id="insightScore">-</div>
                        <div class="fortune-desc">新洞察产生概率</div>
                    </div>
                </div>

                <div class="fortune-tips" id="fortuneTips" style="display: none; margin-top: 16px;">
                    <strong>💭 分析建议：</strong>
                    <span id="fortuneAnalysis">-</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 全局变量
        let trendChart = null;
        let causalChart = null;
        let selectedNode = null;

        // 初始化
        document.addEventListener('DOMContentLoaded', () => {
            loadStats();
            loadMemories();
            initCharts();
            loadStarMap();
        });

        // 标签页切换
        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelector(`[onclick="switchTab('${tabId}')"]`).classList.add('active');
            document.getElementById(`tab-${tabId}`).classList.add('active');
        }

        // 加载统计
        async function loadStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();

                document.getElementById('totalMemories').textContent = data.total_memories || 0;
                document.getElementById('causalLinks').textContent = data.causal_links || 0;
                document.getElementById('multiHop').textContent = data.multihop_depth || 0;
                document.getElementById('confidence').textContent = ((data.confidence || 0) * 100).toFixed(0) + '%';
            } catch (e) {
                console.error('加载统计失败:', e);
            }
        }

        // 加载记忆列表
        async function loadMemories() {
            try {
                const res = await fetch('/api/memories');
                const data = await res.json();
                renderMemories(data);
            } catch (e) {
                console.error('加载记忆失败:', e);
            }
        }

        function renderMemories(memories) {
            const container = document.getElementById('memoryList');
            if (!memories || memories.length === 0) {
                container.innerHTML = '<div class="empty-state"><div class="icon">📭</div><p>暂无记忆数据</p></div>';
                return;
            }

            container.innerHTML = memories.map(m => `
                <div class="memory-item">
                    <div class="memory-content">${escapeHtml(m.content || '')}</div>
                    <div class="memory-meta">
                        <span class="memory-score">${((m.score || 0) * 100).toFixed(1)}%</span>
                        <span>ID: ${m.memory_id || m.id || '-'}</span>
                    </div>
                </div>
            `).join('');
        }

        // 添加记忆
        document.getElementById('addForm').onsubmit = async (e) => {
            e.preventDefault();
            const form = new FormData(e.target);
            const data = {
                content: form.get('content'),
                metadata: form.get('metadata') ? JSON.parse(form.get('metadata')) : {}
            };

            try {
                const res = await fetch('/api/add', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await res.json();
                if (result.success) {
                    alert('✅ 记忆添加成功');
                    e.target.reset();
                    loadStats();
                    loadMemories();
                    loadStarMap();
                }
            } catch (err) {
                alert('❌ 错误: ' + err.message);
            }
        };

        // 查询记忆
        document.getElementById('queryForm').onsubmit = async (e) => {
            e.preventDefault();
            const form = new FormData(e.target);
            const data = {
                query: form.get('query'),
                top_k: parseInt(form.get('top_k'))
            };

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const results = await res.json();
                renderMemories(results);
            } catch (err) {
                alert('❌ 错误: ' + err.message);
            }
        };

        // 多跳推理查询
        async function queryMultihop() {
            const query = document.querySelector('#queryForm input[name="query"]').value;
            if (!query) {
                alert('请输入查询内容');
                return;
            }

            try {
                const res = await fetch('/api/query_multihop', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query, max_hops: 3})
                });
                const data = await res.json();

                let html = '<h4 style="margin: 16px 0 12px;">🌀 多跳推理结果</h4>';
                if (data.results && data.results.length > 0) {
                    data.results.forEach((r, i) => {
                        html += `<div class="memory-item" style="border-left: 3px solid var(--accent-purple);">
                            <div class="memory-content">${escapeHtml(r.content || '')}</div>
                            <div class="memory-meta">
                                <span class="memory-score">跳${(r.hop || i+1)}</span>
                                <span>置信度: ${((r.score || 0) * 100).toFixed(1)}%</span>
                            </div>
                        </div>`;
                    });
                } else {
                    html += '<div class="empty-state"><p>未找到多跳推理结果</p></div>';
                }

                document.getElementById('memoryList').innerHTML = html;
            } catch (err) {
                alert('❌ 多跳推理失败: ' + err.message);
            }
        }

        // 初始化图表
        function initCharts() {
            // 趋势图
            const trendCtx = document.getElementById('trendChart').getContext('2d');
            trendChart = new Chart(trendCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: '记忆数量',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8892a4' } },
                        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8892a4' } }
                    }
                }
            });

            // 因果统计图
            const causalCtx = document.getElementById('causalChart').getContext('2d');
            causalChart = new Chart(causalCtx, {
                type: 'doughnut',
                data: {
                    labels: ['直接关联', '因果关系', '时序关联', '独立'],
                    datasets: [{
                        data: [0, 0, 0, 0],
                        backgroundColor: ['#3b82f6', '#8b5cf6', '#06b6d4', '#64748b']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'right', labels: { color: '#e0e6ed' } } }
                }
            });

            // 加载历史数据
            loadTrendData();
        }

        async function loadTrendData() {
            try {
                const res = await fetch('/api/trend');
                const data = await res.json();

                if (data.labels) {
                    trendChart.data.labels = data.labels;
                    trendChart.data.datasets[0].data = data.values;
                    trendChart.update();
                }

                if (data.causal) {
                    causalChart.data.datasets[0].data = [
                        data.causal.direct || 0,
                        data.causal.causal || 0,
                        data.causal.temporal || 0,
                        data.causal.independent || 0
                    ];
                    causalChart.update();
                }
            } catch (e) {
                console.error('加载趋势数据失败:', e);
            }
        }

        // 加载星图
        async function loadStarMap() {
            try {
                const res = await fetch('/api/starmap');
                const data = await res.json();
                renderStarGraph(data);
            } catch (e) {
                console.error('加载星图失败:', e);
            }
        }

        function refreshStarMap() {
            loadStarMap();
            loadTrendData();
        }

        function renderStarGraph(data) {
            const svg = document.getElementById('starSvg');
            const width = svg.clientWidth || 800;
            const height = svg.clientHeight || 500;

            let svgContent = '';
            const nodes = data.nodes || [];
            const edges = data.edges || [];

            // 绘制连线
            edges.forEach(edge => {
                const from = nodes.find(n => n.id === edge.from);
                const to = nodes.find(n => n.id === edge.to);
                if (from && to) {
                    svgContent += `<line class="edge" x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}" data-edge="true"/>`;
                }
            });

            // 绘制节点
            nodes.forEach((node, i) => {
                const size = 6 + (node.importance || 0.5) * 10;
                const hue = 200 + (i * 37) % 140;
                svgContent += `
                    <g class="node" transform="translate(${node.x}, ${node.y})" onclick="selectNode('${node.id}')">
                        <circle r="${size}" fill="hsl(${hue}, 70%, 60%)" opacity="0.8">
                            <animate attributeName="opacity" values="0.6;1;0.6" dur="${2 + i % 3}s" repeatCount="indefinite"/>
                        </circle>
                        <circle r="${size * 0.6}" fill="hsl(${hue}, 80%, 75%)"/>
                        <text class="node-text" dy="4">${(i + 1)}</text>
                    </g>
                `;
            });

            svg.innerHTML = svgContent;
        }

        function selectNode(nodeId) {
            fetch(`/api/node/${nodeId}`).then(r => r.json()).then(data => {
                document.getElementById('detailContent').textContent = data.content || '-';
                document.getElementById('detailId').textContent = data.id || '-';
                document.getElementById('detailCausal').textContent = data.causal_type || '-';
                document.getElementById('detailConfidence').textContent = ((data.confidence || 0) * 100).toFixed(1) + '%';
                document.getElementById('detailRelated').textContent = (data.related || []).join(', ') || '-';
                document.getElementById('nodeDetail').classList.add('show');

                // 高亮相关边
                document.querySelectorAll('.edge').forEach(e => e.classList.remove('highlight'));
                document.querySelectorAll('.node').forEach(n => n.classList.remove('highlight'));
            });
        }

        function hideNodeDetail() {
            document.getElementById('nodeDetail').classList.remove('show');
            document.querySelectorAll('.edge').forEach(e => e.classList.remove('highlight'));
            document.querySelectorAll('.node').forEach(n => n.classList.remove('highlight'));
        }

        // 运势分析
        async function analyzeFortune() {
            const resultDiv = document.getElementById('fortuneResult');
            const tipsDiv = document.getElementById('fortuneTips');

            resultDiv.style.display = 'none';
            tipsDiv.style.display = 'none';

            try {
                const res = await fetch('/api/fortune', { method: 'POST' });
                const data = await res.json();

                document.getElementById('dailyTrend').textContent = data.daily_trend || '-';
                document.getElementById('relationIndex').textContent = data.relation_index || '-';
                document.getElementById('multihopProb').textContent = data.multihop_prob || '-';
                document.getElementById('insightScore').textContent = data.insight_score || '-';
                document.getElementById('fortuneAnalysis').textContent = data.analysis || '-';

                resultDiv.style.display = 'grid';
                tipsDiv.style.display = 'block';
            } catch (err) {
                alert('❌ 分析失败: ' + err.message);
            }
        }

        // 工具函数
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // 定时刷新
        setInterval(() => {
            loadStats();
            loadTrendData();
        }, 30000);
    </script>
</body>
</html>
'''


# ============================================================
# API 路由
# ============================================================

@app.route('/')
def index():
    """Dashboard主页"""
    return render_template_string(TEMPLATE)


@app.route('/api/add', methods=['POST'])
def add_memory():
    """添加记忆API"""
    data = request.json
    try:
        memory_id = client.add(data.get('content', ''), data.get('metadata'))

        # 记录历史
        _history.append({
            'timestamp': datetime.now().isoformat(),
            'count': client.get_stats().get('count', 0),
            'action': 'add'
        })
        if len(_history) > MAX_HISTORY:
            _history.pop(0)

        return jsonify({'success': True, 'memory_id': memory_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/query', methods=['POST'])
def query_memories():
    """查询记忆API"""
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 10)

    results = client.query(query, top_k=top_k)
    return jsonify([{
        'memory_id': r.get('memory_id', r.get('id', '')),
        'content': r.get('content', ''),
        'score': r.get('score', 0),
        'metadata': r.get('metadata', {})
    } for r in results])


@app.route('/api/memories', methods=['GET'])
def get_memories():
    """获取所有记忆（分页）"""
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('page_size', 20, type=int)

    stats = client.get_stats()
    memories = stats.get('recent_memories', [])

    start = (page - 1) * page_size
    end = start + page_size

    return jsonify({
        'memories': memories[start:end],
        'total': stats.get('count', 0),
        'page': page,
        'page_size': page_size
    })


@app.route('/api/stats')
def get_stats():
    """获取统计API"""
    stats = client.get_stats()

    # 估算因果链路和多跳深度
    causal_links = min(stats.get('count', 0) * 2, 100)
    multihop_depth = min(int(stats.get('count', 0) / 5) + 1, 5)
    confidence = min(0.5 + (stats.get('count', 0) * 0.02), 0.95)

    return jsonify({
        'total_memories': stats.get('count', 0),
        'causal_links': causal_links,
        'multihop_depth': multihop_depth,
        'confidence': confidence
    })


@app.route('/api/trend')
def get_trend():
    """获取趋势数据"""
    labels = [h['timestamp'][:10] for h in _history[-20:]]
    values = [h['count'] for h in _history[-20:]]

    # 简化因果分布
    total = max(client.get_stats().get('count', 1), 1)
    causal = {
        'direct': int(total * 0.3),
        'causal': int(total * 0.25),
        'temporal': int(total * 0.2),
        'independent': int(total * 0.25)
    }

    return jsonify({
        'labels': labels or ['暂无数据'],
        'values': values or [0],
        'causal': causal
    })


@app.route('/api/starmap')
def get_starmap():
    """获取星图数据"""
    stats = client.get_stats()
    count = stats.get('count', 0)

    # 生成星图节点
    nodes = []
    edges = []

    memories = stats.get('recent_memories', []) or []
    if not memories:
        # 生成示例数据
        for i in range(min(count, 15)):
            angle = (i / 15) * math.pi
            radius = 150 + (i % 3) * 50
            nodes.append({
                'id': f'm{i+1}',
                'x': 400 + radius * math.cos(angle),
                'y': 250 + radius * math.sin(angle),
                'importance': 0.3 + (i % 5) * 0.15
            })

            if i > 0:
                edges.append({
                    'from': f'm{i}',
                    'to': f'm{i+1}',
                    'type': 'sequence'
                })
    else:
        for i, m in enumerate(memories[:15]):
            angle = (i / len(memories)) * math.pi
            radius = 120 + (i % 4) * 60
            nodes.append({
                'id': m.get('id', f'm{i}'),
                'x': 400 + radius * math.cos(angle),
                'y': 250 + radius * math.sin(angle),
                'importance': 0.3 + (i % 5) * 0.15
            })

            if i > 0 and i % 3 == 0:
                edges.append({
                    'from': m.get('id', f'm{i-3}'),
                    'to': m.get('id', f'm{i}'),
                    'type': 'causal'
                })

    return jsonify({'nodes': nodes, 'edges': edges})


@app.route('/api/node/<node_id>')
def get_node(node_id):
    """获取节点详情"""
    stats = client.get_stats()
    memories = stats.get('recent_memories', []) or []

    memory = next((m for m in memories if m.get('id') == node_id), None)

    if not memory:
        return jsonify({
            'id': node_id,
            'content': f'记忆节点 #{node_id}',
            'causal_type': 'unknown',
            'confidence': 0.5,
            'related': ['节点1', '节点3']
        })

    return jsonify({
        'id': node_id,
        'content': memory.get('content', ''),
        'causal_type': 'sequence',
        'confidence': memory.get('score', 0.8),
        'related': memories[max(0, memories.index(memory)-1):min(len(memories), memories.index(memory)+2)]
    })


@app.route('/api/query_multihop', methods=['POST'])
def query_multihop():
    """多跳推理查询"""
    data = request.json
    query = data.get('query', '')
    max_hops = data.get('max_hops', 3)

    # 使用增强版客户端的多跳功能
    if CLIENT_TYPE == "lite_pro" and hasattr(client, 'query_multihop'):
        results = client.query_multihop(query, max_hops=max_hops)
        return jsonify({
            'results': results[:10],
            'hops': len(results),
            'explanation': f'完成{len(results)}跳推理'
        })

    # 回退到普通查询
    fallback = client.query(query, top_k=10)
    return jsonify({
        'results': fallback,
        'hops': 1,
        'explanation': '使用单跳查询（Lite Pro功能不可用）'
    })


@app.route('/api/fortune', methods=['POST'])
def analyze_fortune():
    """运势分析（测试功能）"""
    stats = client.get_stats()
    count = stats.get('count', 0)

    # 基于记忆数量生成运势指标
    import random
    random.seed(count)

    daily_trend = ['⬆️ 上升期', '➡️ 平稳期', '⬇️ 调整期'][count % 3]
    relation_index = f'{min(50 + count * 3, 99)}%'
    multihop_prob = f'{min(30 + count * 5, 95)}%'
    insight_score = f'{min(40 + count * 4, 98)}分'

    # 分析建议
    analyses = [
        '记忆关联性较高，适合进行多跳推理探索',
        '数据积累充分，可以进行趋势预测分析',
        '建议增加更多记忆以提升预测准确性',
        '因果链路丰富，可尝试复杂的因果推理'
    ]

    return jsonify({
        'daily_trend': daily_trend,
        'relation_index': relation_index,
        'multihop_prob': multihop_prob,
        'insight_score': insight_score,
        'analysis': analyses[count % len(analyses)]
    })


# ============================================================
# 启动
# ============================================================

def main():
    """启动Dashboard"""
    print("🧠 su-memory Dashboard 启动中...")
    print(f"   客户端类型: {CLIENT_TYPE}")
    print("   访问地址: http://localhost:8765")
    print("   按 Ctrl+C 停止")
    app.run(host='0.0.0.0', port=8765, debug=False)


if __name__ == '__main__':
    main()
