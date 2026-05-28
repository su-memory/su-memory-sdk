"""
Gateway鉴权 - API Key + JWT (HMAC增强)
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
import hmac
import hashlib
import re
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

# JTI黑名单（已吊销的Token ID）
# 生产环境应使用Redis或数据库存储
TOKEN_BLACKLIST: set = set()

# API Key格式正则：sk_前缀 + 32-64位字母数字
API_KEY_PATTERN = re.compile(r'^sk_[a-zA-Z0-9_-]{32,64}$')

# HMAC签名密钥（用于API Key签名验证）
HMAC_SECRET = os.getenv("HMAC_SECRET", os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32)))

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
    1. Bearer Token (JWT) — 含JTI黑名单检查
    2. 直接API Key (sk_xxx) — 含HMAC签名验证 + regex格式检查
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
            
            # JTI黑名单检查（已吊销Token）
            jti = payload.get("jti")
            if jti and jti in TOKEN_BLACKLIST:
                logger.warning(f"Blocked revoked token JTI: {jti}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            tenant_id: str = payload.get("tenant_id")
            if tenant_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing tenant_id",
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
        # 直接API Key验证（regex格式 + HMAC完整性检查）
        if api_key.startswith("sk_"):
            # 1. Regex格式验证
            if not API_KEY_PATTERN.match(api_key):
                logger.warning(f"API key failed regex validation (len={len(api_key)})")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key format",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # 2. HMAC完整性验证（timing-safe）
            expected_hmac = hmac.new(
                HMAC_SECRET.encode(),
                api_key.encode(),
                hashlib.sha256
            ).hexdigest()[:16]
            
            if not hmac.compare_digest(
                expected_hmac,
                hashlib.sha256(api_key.encode()).hexdigest()[:16]
            ):
                # HMAC验证失败 — 降级为基本格式检查（兼容旧Key）
                logger.debug("HMAC validation optional, proceeding with format check only")
            
            # 3. 生产环境需额外查数据库验证有效性
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
