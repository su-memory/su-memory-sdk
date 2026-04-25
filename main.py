"""
su-memory - 企业级AI记忆中台
FastAPI 主入口
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time

from gateway.router import router as gateway_router
from gateway.middleware import setup_middleware
from storage.relational_db import init_db
from storage.vector_db import init_vector_db

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI应用
app = FastAPI(
    title="su-memory",
    description="企业级AI记忆中台 - 私有化大模型记忆解决方案",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    start_time = time.time()
    request_id = f"{time.time()}"
    
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info(f"[{request_id}] Completed in {duration:.3f}s - {response.status_code}")
        return response
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"[{request_id}] Error after {duration:.3f}s - {str(e)}")
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "code": "INTERNAL_ERROR"}
    )


# 注册中间件
setup_middleware(app)

# 注册路由
app.include_router(gateway_router, prefix="/v1")


@app.get("/")
async def root():
    """健康检查"""
    return {
        "service": "su-memory",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """详细健康检查"""
    return {
        "status": "healthy",
        "service": "su-memory",
        "version": "1.0.0"
    }


@app.on_event("startup")
async def startup_event():
    """启动初始化"""
    logger.info("su-memory starting up...")
    
    # 初始化数据库
    await init_db()
    logger.info("Database initialized")
    
    # 初始化向量库
    await init_vector_db()
    logger.info("Vector DB initialized")
    
    logger.info("su-memory ready")


@app.on_event("shutdown")
async def shutdown_event():
    """关闭清理"""
    logger.info("su-memory shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
