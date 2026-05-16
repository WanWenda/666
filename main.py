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
    description="NCU RAG Tutor",
    version="1.1.0"
)

from fastapi import Request
from fastapi.responses import JSONResponse

@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/admin") or request.url.path.endswith(".html"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response

@app.middleware("http")
async def gateway_security_guard(request: Request, call_next):
    # Render 环境直接放行
    if os.environ.get("RENDER","").lower() == "true":
        return await call_next(request)

    # 放行本地请求
    client_ip = request.client.host if request.client.host else ""
    if client_ip in ["127.0.0.1", "localhost"] or client_ip.startswith("172."):
        return await call_next(request)

    # 放行 API 接口（需要网关令牌在非 Render 环境）
    path = request.url.path
    if path == "/health" or path.startswith("/admin") or path.endswith(".html"):
        return await call_next(request)

    gateway_token = request.headers.get("X-AIIA-Gateway-Auth")
    if gateway_token == settings.GATEWAY_AUTH_TOKEN:
        return await call_next(request)

    print(f"Denied external access (IP: {client_ip})")
    return JSONResponse(
        status_code=403,
        content={"detail": "Access Denied: Internal AI Engine Protected."}
    )

app.include_router(webhook_router, prefix=settings.API_V1_STR + "/wecom")
app.include_router(admin_router, prefix=settings.API_V1_STR + "/admin")

static_dir = os.path.join(os.path.dirname(__file__), "app", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/admin", StaticFiles(directory=static_dir, html=True), name="admin_static")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    faulthandler.enable()
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False, log_level="info")
