"""
su-memory SDK Web Dashboard

简单的本地Web界面，用于可视化记忆系统状态。

启动方式:
    python -m su_memory.dashboard

访问: http://localhost:8765
"""

from flask import Flask, jsonify, request, render_template_string
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from su_memory import SuMemory

app = Flask(__name__)
client = SuMemory()

# HTML模板
TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>su-memory Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #2563eb; margin-bottom: 20px; }
        .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .stat { background: #f0f9ff; padding: 15px; border-radius: 6px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #2563eb; }
        .stat-label { color: #64748b; font-size: 0.9em; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: 500; }
        input, textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        textarea { min-height: 80px; }
        button { background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #1d4ed8; }
        .result { margin-top: 20px; }
        .memory-item { background: #f8fafc; padding: 15px; border-radius: 6px; margin-bottom: 10px; border-left: 4px solid #2563eb; }
        .memory-content { font-size: 1.1em; margin-bottom: 8px; }
        .memory-meta { font-size: 0.85em; color: #64748b; }
        .memory-score { background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; }
        .error { background: #fef2f2; color: #dc2626; padding: 15px; border-radius: 6px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 su-memory Dashboard</h1>
        
        <div class="card">
            <h2>📊 系统状态</h2>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value" id="total">{{ stats.total_memories }}</div>
                    <div class="stat-label">记忆总数</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="categories">{{ stats.categories }}</div>
                    <div class="stat-label">类别数</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>➕ 添加记忆</h2>
            <form id="addForm">
                <div class="form-group">
                    <label>内容</label>
                    <textarea name="content" placeholder="输入记忆内容..."></textarea>
                </div>
                <div class="form-group">
                    <label>元数据 (JSON)</label>
                    <input name="metadata" placeholder='{"source": "manual"}'>
                </div>
                <button type="submit">添加</button>
            </form>
        </div>
        
        <div class="card">
            <h2>🔍 查询记忆</h2>
            <form id="queryForm">
                <div class="form-group">
                    <label>查询</label>
                    <input name="query" placeholder="输入查询...">
                </div>
                <div class="form-group">
                    <label>返回数量</label>
                    <input name="top_k" type="number" value="5" min="1" max="100">
                </div>
                <button type="submit">查询</button>
            </form>
            <div class="result" id="queryResult"></div>
        </div>
    </div>
    
    <script>
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
                    alert('✅ 记忆添加成功: ' + result.memory_id);
                    location.reload();
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
                
                const container = document.getElementById('queryResult');
                if (results.length === 0) {
                    container.innerHTML = '<div class="error">没有找到匹配的记忆</div>';
                } else {
                    container.innerHTML = results.map(r => `
                        <div class="memory-item">
                            <div class="memory-content">${r.content}</div>
                            <div class="memory-meta">
                                <span class="memory-score">${(r.score * 100).toFixed(1)}%</span>
                                ID: ${r.memory_id}
                            </div>
                        </div>
                    `).join('');
                }
            } catch (err) {
                alert('❌ 错误: ' + err.message);
            }
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Dashboard主页"""
    stats = client.get_stats()
    categories = len(stats.get('category_distribution', {}))
    return render_template_string(TEMPLATE, stats={
        'total_memories': stats.get('total_memories', 0),
        'categories': categories
    })

@app.route('/api/add', methods=['POST'])
def add_memory():
    """添加记忆API"""
    data = request.json
    try:
        memory_id = client.add(data.get('content', ''), data.get('metadata'))
        return jsonify({'success': True, 'memory_id': memory_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/query', methods=['POST'])
def query_memories():
    """查询记忆API"""
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 5)
    
    results = client.query(query, top_k=top_k)
    return jsonify([{
        'memory_id': r.memory_id,
        'content': r.content,
        'score': r.score,
        'metadata': r.metadata
    } for r in results])

@app.route('/api/stats')
def get_stats():
    """获取统计API"""
    return jsonify(client.get_stats())

def main():
    """启动Dashboard"""
    print("🧠 su-memory Dashboard 启动中...")
    print("访问: http://localhost:8765")
    print("按 Ctrl+C 停止")
    app.run(host='0.0.0.0', port=8765, debug=False)

if __name__ == '__main__':
    main()
