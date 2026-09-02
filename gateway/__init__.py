"""Gateway层 - 路由聚合"""
from .auth import create_access_token, get_current_tenant, verify_api_key
from .middleware import setup_middleware
from .router import router

__all__ = ["router", "verify_api_key", "create_access_token", "get_current_tenant", "setup_middleware"]
