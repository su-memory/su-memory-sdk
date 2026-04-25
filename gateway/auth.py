"""
Gateway鉴权 - API Key + JWT
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
import secrets
import os
import logging

logger = logging.getLogger(__name__)

# JWT配置
JWT_SECRET_KEY = os.getenv("JWT_SECRET", os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32)))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24小时

# API Key Header
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# 密码哈希
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 模拟租户存储（生产环境用数据库）
TENANTS: dict = {}


def generate_api_key() -> str:
    """生成API Key"""
    return f"sk_{secrets.token_urlsafe(32)}"


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """创建JWT Token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """
    验证API Key，返回tenant_id
    支持两种格式：
    1. Bearer Token (JWT)
    2. 直接API Key (sk_xxx)
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 去掉Bearer前缀
    if api_key.startswith("Bearer "):
        token = api_key[7:]
        
        # 验证JWT Token
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            tenant_id: str = payload.get("tenant_id")
            if tenant_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return tenant_id
            
        except JWTError as e:
            logger.error(f"JWT verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token verification failed",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        # 直接API Key验证
        # 在生产环境查询数据库
        # 这里简化处理，实际应该查数据库
        if api_key.startswith("sk_"):
            # TODO: 生产环境需添加数据库验证，当前为简化实现
            # 基本格式验证：sk_ 前缀 + 至少32字符
            if len(api_key) < 35:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key format: key too short",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            # 暂时直接返回api_key作为tenant_id（简化处理）
            # TODO: 后续需查询数据库验证API Key有效性，并返回关联的tenant_id
            return api_key
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_tenant(tenant_id: str = Depends(verify_api_key)) -> dict:
    """获取当前租户信息"""
    # 在生产环境从数据库查询
    return {
        "tenant_id": tenant_id,
        "plan": "enterprise",
        "created_at": datetime.utcnow().isoformat()
    }
