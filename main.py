import faulthandler
faulthandler.enable()
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.webhook import router as webhook_router
from app.api.admin import router as admin_router
from app.core.config import settings
import uvicorn
import os

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="南昌大学智慧课程平台 - 零幻觉企业微信课后伴学助教专属代理。",
    version="1.1.0" # 更新版本号
)

# 🚀 增加强制不缓存中间件，解决用户端“没有变化”的问题
from fastapi import Request
from fastapi.responses import JSONResponse
@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/admin") or request.url.path.endswith(".html"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response

import base64
from fastapi.responses import JSONResponse

# 🛡️ 增加网关来源拦截鉴权，防止家里 8.6 直连公网被扫刷
@app.middleware("http")
async def gateway_security_guard(request: Request, call_next):
    # 如果在 Render 部署，放行所有请求（Render 自带 HTTPS + 域名保护）
    if os.environ.get("RENDER", "").lower() == "true":
        return await call_next(request)
    # 如果在 Render 部署，放行所有请求（Render 自带 HTTPS + 域名保护）
    if os.environ.get("RENDER", "").lower() == "true":
        return await call_next(request)

    # 放行本地发起的绝对安全请求、Docker 虚拟网桥（172.*）、以及家网其他终端
    client_ip = request.client.host if request.client.host else ""
    if client_ip in ["127.0.0.1", "localhost"] or client_ip.startswith("172.") or (client_ip.startswith("192.168.8.") and client_ip != "192.168.8.88"):
        return await call_next(request)

    path = request.url.path

    # 放行企业微信回调、健康检查和管理面板 UI 资源
    if path.startswith(settings.API_V1_STR + "/wecom") or path == "/health" or path.startswith("/admin") or path.endswith(".html"):
        return await call_next(request)

    # 1. 检查是否带有学校主赛道 Nginx 打上的隐藏邮戳（外网任何人抓包都抓不到的内容）
    gateway_token = request.headers.get("X-AIIA-Gateway-Auth")
    if gateway_token == settings.GATEWAY_AUTH_TOKEN:
        return await call_next(request)

    # 2. 既没有内网身份，也没有邮戳，直接铁壁阻断！无商量余地。
    print(f"🚨 [骇客拦截] 拒绝非法公网 / 网关越权访问 (IP: {client_ip})")
    return JSONResponse(
        status_code=403,
        content={"detail": "Access Denied: Internal AI Engine Protected. Direct external access is forbidden."}
    )

# 挂载微信群 / 企微回调路由
app.include_router(webhook_router, prefix=settings.API_V1_STR + "/wecom")

# 挂载内部管理员面板 API
app.include_router(admin_router, prefix=settings.API_V1_STR + "/admin")

# 暴露单页 Admin 应用的静态主页
static_dir = os.path.join(os.path.dirname(__file__), "app", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/admin", StaticFiles(directory=static_dir, html=True), name="admin_static")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "llm_target": settings.LLM_API_BASE,
        "rag_persistence": settings.CHROMA_PERSIST_DIR
    }

if __name__ == "__main__":
    # 💡 核心诊断：在极端不稳定的云端环境下强制开启故障捕获
    faulthandler.enable()
    # 💡 核心安全修复：在部署环境下必须关闭 reload，防止文件变动（如缓存写入）触发进程自杀
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False, log_level="debug")
