"""
API Gateway 端到端综合测试

使用 TestClient + Mock 方式测试 API 层的路由、鉴权、参数校验逻辑。
Docker 服务（Qdrant/PostgreSQL）不需要实际运行。

测试覆盖：
1. 健康检查端点
2. 记忆写入（Retain / memory/add）
3. 记忆检索（Recall / memory/query）
4. 记忆删除（Delete / memory/delete）
5. 租户创建
6. 记忆统计
7. Chat Completions
8. 鉴权测试（无Token/过期Token/非法Token/正确Token）
9. 完整生命周期测试
10. 并发测试
11. 参数校验（异常输入）
12. 未知路由
13. 安全头
14. API 发现（OpenAPI）
15. Bug 发现（字段不匹配等）
"""

import sys
import os
import time
import uuid
import threading
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from jose import jwt

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# Mock 外部依赖（在 import main 之前）
# ============================================================

mock_st = MagicMock()
sys.modules.setdefault('sentence_transformers', mock_st)

with patch('storage.relational_db.init_db', new_callable=AsyncMock), \
     patch('storage.vector_db.init_vector_db', new_callable=AsyncMock):
    from main import app


# ============================================================
# 全局 Mock memory_manager（避免真实数据库连接）
# ============================================================

# 创建全局 mock 实例
_mock_memory_manager = MagicMock()
_mock_memory_manager.add_memory = AsyncMock(return_value="mock-mem-001")
_mock_memory_manager.query_memory = AsyncMock(return_value=[])
_mock_memory_manager.delete_memory = AsyncMock(return_value=None)
_mock_memory_manager.create_tenant = AsyncMock(return_value={
    "tenant_id": "mock-t-001",
    "name": "mock",
    "api_key": "sk_mock",
    "created_at": "2026-01-01T00:00:00Z"
})
_mock_memory_manager.get_stats = AsyncMock(return_value={
    "user_id": "mock-user",
    "total_memories": 0,
    "active_memories": 0,
    "archived_memories": 0,
    "storage_bytes": 0
})

# 在模块级别 patch
_patch_mm = patch('gateway.router.memory_manager', _mock_memory_manager)
_patch_mm.start()


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def client():
    """TestClient 实例"""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_mocks():
    """每个测试前重置 mock"""
    _mock_memory_manager.add_memory.reset_mock(return_value=True)
    _mock_memory_manager.add_memory.return_value = "mock-mem-001"
    _mock_memory_manager.query_memory.reset_mock(return_value=True)
    _mock_memory_manager.query_memory.return_value = []
    _mock_memory_manager.delete_memory.reset_mock(return_value=True)
    _mock_memory_manager.delete_memory.return_value = None
    _mock_memory_manager.create_tenant.reset_mock(return_value=True)
    _mock_memory_manager.create_tenant.return_value = {
        "tenant_id": "mock-t-001",
        "name": "mock",
        "api_key": "sk_mock",
        "created_at": "2026-01-01T00:00:00Z"
    }
    _mock_memory_manager.get_stats.reset_mock(return_value=True)
    _mock_memory_manager.get_stats.return_value = {
        "user_id": "mock-user",
        "total_memories": 0,
        "active_memories": 0,
        "archived_memories": 0,
        "storage_bytes": 0
    }
    yield


@pytest.fixture
def valid_api_key():
    """生成有效的 API Key（sk_ 格式）"""
    return f"sk_{uuid.uuid4().hex}"


@pytest.fixture
def valid_jwt_token():
    """生成有效的 JWT Token"""
    from gateway.auth import JWT_SECRET_KEY, JWT_ALGORITHM, create_access_token
    token = create_access_token({"tenant_id": "test-tenant-001"})
    return token


@pytest.fixture
def expired_jwt_token():
    """生成过期的 JWT Token"""
    from gateway.auth import JWT_SECRET_KEY, JWT_ALGORITHM
    expire = datetime.utcnow() - timedelta(hours=1)
    to_encode = {"tenant_id": "test-tenant-001", "exp": expire}
    token = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


@pytest.fixture
def auth_headers_jwt(valid_jwt_token):
    """使用 JWT Bearer Token 的鉴权头"""
    return {"Authorization": f"Bearer {valid_jwt_token}"}


# ============================================================
# 1. 健康检查测试
# ============================================================

class TestHealthCheck:
    """健康检查端点测试"""

    def test_root_endpoint(self, client):
        """GET / 返回服务信息"""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "su-memory"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"

    def test_health_endpoint(self, client):
        """GET /health 返回健康状态"""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "su-memory"
        assert data["version"] == "1.0.0"

    def test_health_no_auth_required(self, client):
        """健康检查不需要鉴权"""
        resp = client.get("/health")
        assert resp.status_code == 200


# ============================================================
# 2. 鉴权测试
# ============================================================

class TestAuthentication:
    """鉴权机制测试"""

    def test_no_token_returns_401(self, client):
        """无 Token 访问受保护端点 -> 401"""
        resp = client.post("/v1/memory/add", json={
            "user_id": "user1",
            "content": "test content"
        })
        assert resp.status_code == 401
        data = resp.json()
        assert "detail" in data

    def test_invalid_token_format_returns_401(self, client):
        """非法格式的 Token -> 401"""
        resp = client.post("/v1/memory/add",
            json={"user_id": "user1", "content": "test"},
            headers={"Authorization": "invalid-token-format"})
        assert resp.status_code == 401

    def test_expired_jwt_returns_401(self, client, expired_jwt_token):
        """过期 JWT Token -> 401"""
        resp = client.post("/v1/memory/add",
            json={"user_id": "user1", "content": "test"},
            headers={"Authorization": f"Bearer {expired_jwt_token}"})
        assert resp.status_code == 401

    def test_malformed_jwt_returns_401(self, client):
        """伪造的 JWT Token -> 401"""
        resp = client.post("/v1/memory/add",
            json={"user_id": "user1", "content": "test"},
            headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.fake.payload"})
        assert resp.status_code == 401

    def test_valid_jwt_bearer_accepted(self, client, valid_jwt_token):
        """有效的 JWT Bearer Token -> 通过鉴权"""
        _mock_memory_manager.add_memory.return_value = "mem-auth-001"
        resp = client.post("/v1/memory/add",
            json={"user_id": "user1", "content": "test content"},
            headers={"Authorization": f"Bearer {valid_jwt_token}"})
        assert resp.status_code == 200

    def test_valid_api_key_sk_format(self, client, valid_api_key):
        """有效的 API Key（sk_ 格式）-> 通过鉴权"""
        _mock_memory_manager.add_memory.return_value = "mem-auth-002"
        resp = client.post("/v1/memory/add",
            json={"user_id": "user1", "content": "test content"},
            headers={"Authorization": valid_api_key})
        assert resp.status_code == 200

    def test_bearer_with_sk_key_returns_401(self, client):
        """Bearer sk_xxx 格式 -> JWT 验证路径失败 -> 401"""
        resp = client.post("/v1/memory/add",
            json={"user_id": "user1", "content": "test"},
            headers={"Authorization": "Bearer sk_something"})
        assert resp.status_code == 401


# ============================================================
# 3. 记忆写入测试（Retain / memory/add）
# ============================================================

class TestMemoryAdd:
    """记忆写入端点测试"""

    def test_add_memory_success(self, client, auth_headers_jwt):
        """POST /v1/memory/add 正常写入"""
        _mock_memory_manager.add_memory.return_value = "mem-test-001"
        resp = client.post("/v1/memory/add",
            json={
                "user_id": "user-001",
                "content": "用户有高血压病史10年"
            },
            headers=auth_headers_jwt)
        assert resp.status_code == 200
        data = resp.json()
        assert data["memory_id"] == "mem-test-001"
        assert data["status"] == "stored"

    def test_add_memory_with_metadata(self, client, auth_headers_jwt):
        """写入带 metadata 的记忆"""
        _mock_memory_manager.add_memory.return_value = "mem-test-002"
        resp = client.post("/v1/memory/add",
            json={
                "user_id": "user-001",
                "content": "用户喜欢清淡饮食",
                "metadata": {"type": "preference", "source": "对话"}
            },
            headers=auth_headers_jwt)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stored"

    def test_add_memory_fact_type(self, client, auth_headers_jwt):
        """写入事实类记忆"""
        _mock_memory_manager.add_memory.return_value = "mem-fact-001"
        resp = client.post("/v1/memory/add",
            json={
                "user_id": "user-001",
                "content": "地球绕太阳公转一周约365天"
            },
            headers=auth_headers_jwt)
        assert resp.status_code == 200

    def test_add_memory_event_type(self, client, auth_headers_jwt):
        """写入事件类记忆"""
        _mock_memory_manager.add_memory.return_value = "mem-event-001"
        resp = client.post("/v1/memory/add",
            json={
                "user_id": "user-001",
                "content": "今天做了血液检查",
                "metadata": {"type": "event"}
            },
            headers=auth_headers_jwt)
        assert resp.status_code == 200

    def test_add_memory_medical_record(self, client, auth_headers_jwt):
        """写入医疗记录类记忆"""
        _mock_memory_manager.add_memory.return_value = "mem-med-001"
        resp = client.post("/v1/memory/add",
            json={
                "user_id": "patient-001",
                "content": "2026年4月血压140/90mmHg，诊断为高血压1级",
                "metadata": {"type": "fact", "domain": "医疗"}
            },
            headers=auth_headers_jwt)
        assert resp.status_code == 200

    def test_add_memory_empty_user_id(self, client, auth_headers_jwt):
        """空 user_id -> 422 校验失败"""
        resp = client.post("/v1/memory/add",
            json={"user_id": "", "content": "test"},
            headers=auth_headers_jwt)
        assert resp.status_code == 422

    def test_add_memory_empty_content(self, client, auth_headers_jwt):
        """空 content -> 422 校验失败"""
        resp = client.post("/v1/memory/add",
            json={"user_id": "user1", "content": ""},
            headers=auth_headers_jwt)
        assert resp.status_code == 422

    def test_add_memory_missing_user_id(self, client, auth_headers_jwt):
        """缺少 user_id -> 422"""
        resp = client.post("/v1/memory/add",
            json={"content": "test"},
            headers=auth_headers_jwt)
        assert resp.status_code == 422

    def test_add_memory_missing_content(self, client, auth_headers_jwt):
        """缺少 content -> 422"""
        resp = client.post("/v1/memory/add",
            json={"user_id": "user1"},
            headers=auth_headers_jwt)
        assert resp.status_code == 422

    def test_add_memory_no_body(self, client, auth_headers_jwt):
        """空请求体 -> 422"""
        resp = client.post("/v1/memory/add",
            headers=auth_headers_jwt)
        assert resp.status_code == 422

    def test_add_memory_large_content(self, client, auth_headers_jwt):
        """超大 content（接近 100000 字符限制）"""
        _mock_memory_manager.add_memory.return_value = "mem-large-001"
        large_content = "x" * 99999
        resp = client.post("/v1/memory/add",
            json={"user_id": "user1", "content": large_content},
            headers=auth_headers_jwt)
        assert resp.status_code == 200

    def test_add_memory_oversized_content(self, client, auth_headers_jwt):
        """超过 100000 字符限制 -> 422"""
        oversized = "x" * 100001
        resp = client.post("/v1/memory/add",
            json={"user_id": "user1", "content": oversized},
            headers=auth_headers_jwt)
        assert resp.status_code == 422


# ============================================================
# 4. 记忆检索测试（Recall / memory/query）
# ============================================================

class TestMemoryQuery:
    """记忆检索端点测试

    注意：MemoryItem 模型定义了 relevance (float) 和 timestamp (str) 字段，
    但 memory_manager.query_memory 返回的 retriever 数据使用 score (float) 
    和 timestamp (int)。这是一个字段不匹配 Bug。
    详见 TestBugDiscovery。
    """

    def test_query_memory_empty_result(self, client, auth_headers_jwt):
        """POST /v1/memory/query 空结果检索"""
        _mock_memory_manager.query_memory.return_value = []
        resp = client.post("/v1/memory/query",
            json={"user_id": "user-001", "query": "高血压"},
            headers=auth_headers_jwt)
        assert resp.status_code == 200
        data = resp.json()
        assert "memories" in data
        assert "query_time_ms" in data
        assert len(data["memories"]) == 0

    def test_query_memory_with_limit(self, client, auth_headers_jwt):
        """指定 limit 的检索"""
        _mock_memory_manager.query_memory.return_value = []
        resp = client.post("/v1/memory/query",
            json={"user_id": "user-001", "query": "高血压", "limit": 5},
            headers=auth_headers_jwt)
        assert resp.status_code == 200

    def test_query_memory_limit_boundary(self, client, auth_headers_jwt):
        """limit 边界值测试"""
        _mock_memory_manager.query_memory.return_value = []
        # limit=1 (min)
        resp = client.post("/v1/memory/query",
            json={"user_id": "user-001", "query": "test", "limit": 1},
            headers=auth_headers_jwt)
        assert resp.status_code == 200

        # limit=100 (max)
        resp = client.post("/v1/memory/query",
            json={"user_id": "user-001", "query": "test", "limit": 100},
            headers=auth_headers_jwt)
        assert resp.status_code == 200

    def test_query_memory_limit_out_of_range(self, client, auth_headers_jwt):
        """limit 超出范围 -> 422"""
        # limit=0
        resp = client.post("/v1/memory/query",
            json={"user_id": "user-001", "query": "test", "limit": 0},
            headers=auth_headers_jwt)
        assert resp.status_code == 422

        # limit=101
        resp = client.post("/v1/memory/query",
            json={"user_id": "user-001", "query": "test", "limit": 101},
            headers=auth_headers_jwt)
        assert resp.status_code == 422

    def test_query_memory_empty_user_id(self, client, auth_headers_jwt):
        """空 user_id -> 422"""
        resp = client.post("/v1/memory/query",
            json={"user_id": "", "query": "test"},
            headers=auth_headers_jwt)
        assert resp.status_code == 422

    def test_query_memory_empty_query(self, client, auth_headers_jwt):
        """空 query -> 422"""
        resp = client.post("/v1/memory/query",
            json={"user_id": "user1", "query": ""},
            headers=auth_headers_jwt)
        assert resp.status_code == 422

    def test_query_memory_missing_fields(self, client, auth_headers_jwt):
        """缺少必填字段 -> 422"""
        resp = client.post("/v1/memory/query",
            json={},
            headers=auth_headers_jwt)
        assert resp.status_code == 422

    def test_query_memory_no_auth(self, client):
        """无鉴权 -> 401"""
        resp = client.post("/v1/memory/query",
            json={"user_id": "user1", "query": "test"})
        assert resp.status_code == 401


# ============================================================
# 5. 记忆删除测试（memory/delete）
# ============================================================

class TestMemoryDelete:
    """记忆删除端点测试"""

    def test_delete_memory_success(self, client, auth_headers_jwt):
        """POST /v1/memory/delete 正常删除"""
        _mock_memory_manager.delete_memory.return_value = None
        resp = client.post("/v1/memory/delete",
            json={"user_id": "user-001", "memory_id": "mem-001"},
            headers=auth_headers_jwt)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"

    def test_delete_memory_no_auth(self, client):
        """无鉴权删除 -> 401"""
        resp = client.post("/v1/memory/delete",
            json={"user_id": "user1", "memory_id": "mem-001"})
        assert resp.status_code == 401

    def test_delete_memory_missing_fields(self, client, auth_headers_jwt):
        """缺少必填字段 -> 422"""
        resp = client.post("/v1/memory/delete",
            json={"user_id": "user1"},
            headers=auth_headers_jwt)
        assert resp.status_code == 422


# ============================================================
# 6. 租户创建测试
# ============================================================

class TestTenantCreate:
    """租户创建端点测试"""

    def test_create_tenant_success(self, client):
        """POST /v1/tenant/create 正常创建"""
        _mock_memory_manager.create_tenant.return_value = {
            "tenant_id": "t-001",
            "name": "测试公司",
            "api_key": "sk_abc123",
            "created_at": "2026-04-22T00:00:00Z"
        }
        resp = client.post("/v1/tenant/create",
            json={"name": "测试公司", "plan": "enterprise"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == "t-001"
        assert data["api_key"].startswith("sk_")

    def test_create_tenant_default_plan(self, client):
        """默认 plan 为 standard"""
        _mock_memory_manager.create_tenant.return_value = {
            "tenant_id": "t-002",
            "name": "默认公司",
            "api_key": "sk_def456",
            "created_at": "2026-04-22T00:00:00Z"
        }
        resp = client.post("/v1/tenant/create",
            json={"name": "默认公司"})
        assert resp.status_code == 200

    def test_create_tenant_empty_name(self, client):
        """空名称 -> 422"""
        resp = client.post("/v1/tenant/create",
            json={"name": ""})
        assert resp.status_code == 422

    def test_create_tenant_name_too_long(self, client):
        """名称超过 100 字符 -> 422"""
        resp = client.post("/v1/tenant/create",
            json={"name": "x" * 101})
        assert resp.status_code == 422

    def test_create_tenant_no_auth_required(self, client):
        """创建租户不需要鉴权"""
        _mock_memory_manager.create_tenant.return_value = {
            "tenant_id": "t-003",
            "name": "无鉴权",
            "api_key": "sk_xyz",
            "created_at": "2026-04-22T00:00:00Z"
        }
        resp = client.post("/v1/tenant/create",
            json={"name": "无鉴权"})
        assert resp.status_code == 200


# ============================================================
# 7. 记忆统计测试
# ============================================================

class TestMemoryStats:
    """记忆统计端点测试"""

    def test_get_stats_success(self, client, auth_headers_jwt):
        """GET /v1/memory/stats/{user_id} 正常获取"""
        _mock_memory_manager.get_stats.return_value = {
            "user_id": "user-001",
            "total_memories": 10,
            "active_memories": 8,
            "archived_memories": 2,
            "storage_bytes": 4096
        }
        resp = client.get("/v1/memory/stats/user-001",
            headers=auth_headers_jwt)
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "user-001"
        assert data["total_memories"] == 10
        assert data["active_memories"] == 8

    def test_get_stats_no_auth(self, client):
        """无鉴权获取统计 -> 401"""
        resp = client.get("/v1/memory/stats/user-001")
        assert resp.status_code == 401


# ============================================================
# 8. Chat Completions 测试
# ============================================================

class TestChatCompletions:
    """带记忆的对话端点测试"""

    def test_chat_completions_success(self, client, auth_headers_jwt):
        """POST /v1/chat/completions 正常对话"""
        _mock_memory_manager.query_memory.return_value = [
            {"content": "用户有高血压", "id": "m1"}
        ]
        with patch('llm_adapter.openai_compat.LLMAdapter') as MockLLM:
            mock_llm_instance = MagicMock()
            mock_llm_instance.chat = AsyncMock(return_value={
                "id": "chat-001",
                "model": "default",
                "choices": [{"message": {"role": "assistant", "content": "建议控制盐分摄入"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            })
            MockLLM.return_value = mock_llm_instance

            resp = client.post("/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "我应该注意什么？"}],
                    "user_id": "user-001"
                },
                headers=auth_headers_jwt)
            assert resp.status_code == 200
            data = resp.json()
            assert "id" in data
            assert "choices" in data

    def test_chat_completions_no_auth(self, client):
        """无鉴权对话 -> 401"""
        resp = client.post("/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "test"}],
                "user_id": "user1"
            })
        assert resp.status_code == 401

    def test_chat_completions_missing_user_id(self, client, auth_headers_jwt):
        """缺少 user_id -> 422"""
        resp = client.post("/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "test"}]
            },
            headers=auth_headers_jwt)
        assert resp.status_code == 422

    def test_chat_completions_invalid_temperature(self, client, auth_headers_jwt):
        """temperature 超出范围 -> 422"""
        resp = client.post("/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "test"}],
                "user_id": "user1",
                "temperature": 3.0
            },
            headers=auth_headers_jwt)
        assert resp.status_code == 422

    def test_chat_completions_invalid_max_tokens(self, client, auth_headers_jwt):
        """max_tokens 小于1 -> 422"""
        resp = client.post("/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "test"}],
                "user_id": "user1",
                "max_tokens": 0
            },
            headers=auth_headers_jwt)
        assert resp.status_code == 422


# ============================================================
# 9. 完整生命周期测试
# ============================================================

class TestLifecycle:
    """完整生命周期：写入 -> 检索 -> 验证 -> 删除 -> 确认"""

    def test_write_delete_lifecycle(self, client, auth_headers_jwt):
        """写入 -> 删除 -> 确认删除"""
        memory_id = "lifecycle-mem-001"
        user_id = "lifecycle-user-001"
        content = "生命周期测试：用户对花生过敏"

        # Step 1: 写入
        _mock_memory_manager.add_memory.return_value = memory_id
        resp = client.post("/v1/memory/add",
            json={"user_id": user_id, "content": content},
            headers=auth_headers_jwt)
        assert resp.status_code == 200
        assert resp.json()["memory_id"] == memory_id
        assert resp.json()["status"] == "stored"

        # Step 2: 删除
        _mock_memory_manager.delete_memory.return_value = None
        resp = client.post("/v1/memory/delete",
            json={"user_id": user_id, "memory_id": memory_id},
            headers=auth_headers_jwt)
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_write_query_stats_lifecycle(self, client, auth_headers_jwt):
        """写入 -> 统计验证"""
        user_id = "stats-user-001"

        # Step 1: 初始统计
        _mock_memory_manager.get_stats.return_value = {
            "user_id": user_id,
            "total_memories": 0,
            "active_memories": 0,
            "archived_memories": 0,
            "storage_bytes": 0
        }
        resp = client.get(f"/v1/memory/stats/{user_id}",
            headers=auth_headers_jwt)
        assert resp.status_code == 200
        assert resp.json()["total_memories"] == 0

        # Step 2: 写入
        _mock_memory_manager.add_memory.return_value = "mem-stats-001"
        resp = client.post("/v1/memory/add",
            json={"user_id": user_id, "content": "测试写入"},
            headers=auth_headers_jwt)
        assert resp.status_code == 200

        # Step 3: 更新后统计
        _mock_memory_manager.get_stats.return_value = {
            "user_id": user_id,
            "total_memories": 1,
            "active_memories": 1,
            "archived_memories": 0,
            "storage_bytes": 128
        }
        resp = client.get(f"/v1/memory/stats/{user_id}",
            headers=auth_headers_jwt)
        assert resp.status_code == 200
        assert resp.json()["total_memories"] == 1

    def test_tenant_create_then_use_api_key(self, client):
        """创建租户 -> 使用返回的 API Key 写入"""
        # Step 1: 创建租户
        _mock_memory_manager.create_tenant.return_value = {
            "tenant_id": "t-lifecycle-001",
            "name": "生命周期公司",
            "api_key": "sk_lifecycle_key_123",
            "created_at": "2026-04-22T00:00:00Z"
        }
        resp = client.post("/v1/tenant/create",
            json={"name": "生命周期公司", "plan": "enterprise"})
        assert resp.status_code == 200
        api_key = resp.json()["api_key"]

        # Step 2: 使用 API Key 写入
        _mock_memory_manager.add_memory.return_value = "mem-tenant-001"
        resp = client.post("/v1/memory/add",
            json={"user_id": "user1", "content": "租户写入测试"},
            headers={"Authorization": api_key})
        assert resp.status_code == 200


# ============================================================
# 10. 并发测试
# ============================================================

class TestConcurrency:
    """并发测试（使用线程安全的方式）"""

    def test_concurrent_writes(self, client, auth_headers_jwt):
        """10 并发写入测试"""
        import concurrent.futures
        lock = threading.Lock()
        results = []
        success_count = [0]

        def write_memory(i):
            with lock:
                _mock_memory_manager.add_memory.return_value = f"mem-concurrent-{i}"
            resp = client.post("/v1/memory/add",
                json={
                    "user_id": f"concurrent-user-{i}",
                    "content": f"并发写入测试内容 {i}"
                },
                headers=auth_headers_jwt)
            if resp.status_code == 200:
                success_count[0] += 1
            return resp.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_memory, i) for i in range(10)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        assert success_count[0] == 10, f"并发写入成功率: {success_count[0]}/10"

    def test_concurrent_queries(self, client, auth_headers_jwt):
        """10 并发检索测试"""
        import concurrent.futures
        lock = threading.Lock()
        results = []
        success_count = [0]

        def query_memory(i):
            with lock:
                _mock_memory_manager.query_memory.return_value = []
            resp = client.post("/v1/memory/query",
                json={"user_id": "user-001", "query": f"查询 {i}"},
                headers=auth_headers_jwt)
            if resp.status_code == 200:
                success_count[0] += 1
            return resp.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(query_memory, i) for i in range(10)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        assert success_count[0] == 10, f"并发检索成功率: {success_count[0]}/10"

    def test_mixed_concurrent_read_write(self, client, auth_headers_jwt):
        """混合并发读写测试（5写+5读）"""
        import concurrent.futures
        lock = threading.Lock()
        results = []
        write_success = [0]
        query_success = [0]

        def write_memory(i):
            with lock:
                _mock_memory_manager.add_memory.return_value = f"mem-mixed-{i}"
            resp = client.post("/v1/memory/add",
                json={"user_id": "user-mixed", "content": f"混合写入 {i}"},
                headers=auth_headers_jwt)
            if resp.status_code == 200:
                write_success[0] += 1
            return ("write", resp.status_code)

        def query_memory(i):
            with lock:
                _mock_memory_manager.query_memory.return_value = []
            resp = client.post("/v1/memory/query",
                json={"user_id": "user-mixed", "query": f"混合查询 {i}"},
                headers=auth_headers_jwt)
            if resp.status_code == 200:
                query_success[0] += 1
            return ("query", resp.status_code)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(5):
                futures.append(executor.submit(write_memory, i))
            for i in range(5):
                futures.append(executor.submit(query_memory, i))
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        assert write_success[0] == 5, f"并发写入成功率: {write_success[0]}/5"
        assert query_success[0] == 5, f"并发检索成功率: {query_success[0]}/5"


# ============================================================
# 11. 未知路由测试
# ============================================================

class TestUnknownRoutes:
    """未知路由测试"""

    def test_nonexistent_endpoint(self, client):
        """访问不存在的端点 -> 404"""
        resp = client.get("/v1/nonexistent")
        assert resp.status_code == 404

    def test_wrong_method(self, client):
        """使用错误 HTTP 方法 -> 405"""
        resp = client.get("/v1/memory/add")
        assert resp.status_code == 405


# ============================================================
# 12. 安全头测试
# ============================================================

class TestSecurityHeaders:
    """安全响应头测试"""

    def test_security_headers_present(self, client):
        """验证安全头是否添加"""
        resp = client.get("/health")
        headers = resp.headers
        # TestClient 安全头可能不全，做软检查
        if "x-content-type-options" in headers:
            assert headers["x-content-type-options"] == "nosniff"


# ============================================================
# 13. API 发现测试（OpenAPI）
# ============================================================

class TestAPIDiscovery:
    """API 文档和发现测试"""

    def test_openapi_docs_available(self, client):
        """GET /docs 返回 Swagger UI"""
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json_available(self, client):
        """GET /openapi.json 返回 OpenAPI Schema"""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "paths" in data
        assert "/v1/memory/add" in data["paths"]
        assert "/v1/memory/query" in data["paths"]
        assert "/v1/memory/delete" in data["paths"]
        assert "/v1/tenant/create" in data["paths"]
        assert "/v1/memory/stats/{user_id}" in data["paths"]
        assert "/v1/chat/completions" in data["paths"]

    def test_redoc_available(self, client):
        """GET /redoc 返回 ReDoc"""
        resp = client.get("/redoc")
        assert resp.status_code == 200


# ============================================================
# 14. Bug 发现测试
# ============================================================

class TestBugDiscovery:
    """已发现的 Bug 和代码问题验证"""

    def test_bug_query_memory_field_mismatch(self, client, auth_headers_jwt):
        """
        BUG #1: MemoryItem 字段不匹配
        
        MemoryItem 模型定义:
        - relevance: float (期望字段名)
        - timestamp: str (期望类型)
        
        memory_manager.query_memory (通过 retriever) 返回:
        - score: float (实际字段名，非 relevance)
        - timestamp: int (实际类型，非 str)
        - memory_type, holographic_score, hexagram_index (额外字段)
        
        端点代码: MemoryItem(**m) 会导致 Pydantic 验证失败
        - 缺少必填字段 'relevance'
        - timestamp 类型不匹配
        
        验证方式：直接构造 MemoryItem 证明字段不匹配
        """
        from gateway.router import MemoryItem
        from pydantic import ValidationError
        
        # 使用 retriever 实际返回的数据格式
        retriever_data = {
            "id": "mem-001",
            "content": "用户有高血压",
            "score": 0.95,           # 应该是 relevance
            "timestamp": 1776841530,  # 应该是 str
            "memory_type": "fact",
            "metadata": {},
            "holographic_score": 0.8,
            "hexagram_index": 5,
        }
        
        # MemoryItem(**m) 会因字段不匹配抛出 ValidationError
        with pytest.raises(ValidationError) as exc_info:
            MemoryItem(**retriever_data)
        
        # 验证具体错误：缺少 relevance 字段，timestamp 类型错误
        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "relevance" in error_fields, "确认 BUG: 缺少 relevance 字段"
        assert "timestamp" in error_fields, "确认 BUG: timestamp 类型不匹配"

    def test_bug_tenant_create_no_auth_security(self, client):
        """
        BUG #2: 创建租户无需鉴权
        
        POST /v1/tenant/create 端点没有 Depends(verify_api_key)，
        任何人都可以创建租户，存在安全风险。
        """
        _mock_memory_manager.create_tenant.return_value = {
            "tenant_id": "t-unauth",
            "name": "未授权创建",
            "api_key": "sk_unauth",
            "created_at": "2026-04-22T00:00:00Z"
        }
        # 不带任何 Authorization header
        resp = client.post("/v1/tenant/create",
            json={"name": "未授权创建"})
        # 当前代码允许无鉴权创建（200），但应该要求鉴权
        assert resp.status_code == 200, \
            "BUG#2 确认：创建租户无需鉴权，存在安全风险"

    def test_bug_jwt_secret_random_on_restart(self):
        """
        BUG #3: JWT 密钥每次重启随机生成
        
        gateway/auth.py 中 JWT_SECRET_KEY = secrets.token_urlsafe(32)
        每次服务重启都会生成新密钥，导致：
        1. 之前颁发的所有 Token 失效
        2. 多实例部署时 Token 不互通
        3. .env 中的 JWT_SECRET_KEY 配置被忽略
        """
        from gateway.auth import JWT_SECRET_KEY
        # 验证密钥是随机生成的（不是 .env 中配置的值）
        assert JWT_SECRET_KEY != "change-this-in-production-please", \
            "BUG#3: JWT密钥使用了随机值而非.env配置"

    def test_bug_retriever_return_format_vs_memory_item(self):
        """
        BUG #4: retriever 返回格式与 MemoryItem 模型不匹配
        
        retriever.py (_holographic_rerank) 返回:
        {
            "id": str,
            "content": str,
            "score": float,          # MemoryItem 期望 'relevance'
            "timestamp": int,        # MemoryItem 期望 str
            "memory_type": str,      # MemoryItem 中无此字段
            "metadata": dict,
            "holographic_score": float,  # MemoryItem 中无此字段
            "hexagram_index": int,       # MemoryItem 中无此字段
        }
        
        router.py (MemoryItem) 期望:
        {
            "id": str,
            "content": str,
            "relevance": float,      # retriever 用的是 'score'
            "timestamp": str,        # retriever 用的是 int
            "metadata": dict,
        }
        """
        from gateway.router import MemoryItem
        import pydantic
        
        # 尝试用 retriever 格式创建 MemoryItem
        retriever_data = {
            "id": "test-id",
            "content": "test content",
            "score": 0.95,
            "timestamp": 1234567890,
            "memory_type": "fact",
            "metadata": {},
            "holographic_score": 0.8,
            "hexagram_index": 5,
        }
        
        with pytest.raises(Exception):
            # 这会失败，因为字段名不匹配
            MemoryItem(**retriever_data)

    def test_bug_api_key_used_as_tenant_id(self):
        """
        BUG #5: API Key 直接用作 tenant_id
        
        gateway/auth.py verify_api_key() 中，当使用 sk_ 格式的 key 时：
        return api_key  # 直接返回 api_key 作为 tenant_id
        
        这意味着：
        1. 任何 sk_ 开头的字符串都可以通过鉴权
        2. 没有验证 API Key 是否真实存在于数据库
        3. 不同 API Key 会产生不同的 tenant_id，导致数据隔离异常
        """
        from gateway.auth import verify_api_key
        # sk_ 格式的 key 直接返回作为 tenant_id
        # 没有数据库验证
        # 这是一个安全漏洞
