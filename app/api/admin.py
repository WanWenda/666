import os
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import shutil

# 借用已经在根目录下的 ingest 模块
import sys
# 保证能引用到 ingest_master_md
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ingest import ingest_master_md

# 引入现成的 RAGEngine 用于进行在线作答
from app.services.rag_engine import rag_engine

from fastapi import Depends
from app.api.auth import verify_jwt
router = APIRouter(dependencies=[Depends(verify_jwt)])

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

class IngestRequest(BaseModel):
    filename: str
    lecture_id: str

class ChatRequest(BaseModel):
    lecture_id: str
    question: str
    image_base64: Optional[str] = None  # 支持可选的图片 Base64 数据

@router.get("/files")
async def list_files():
    """获取 data 目录下的文件列表（支持 .md 和 .pdf）"""
    files = []
    allowed_exts = (".md", ".pdf")
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if any(f.lower().endswith(ext) for ext in allowed_exts):
                file_path = os.path.join(DATA_DIR, f)
                name_no_ext = f[:-4] if f.lower().endswith('.md') else f[:-4]
                namespace = name_no_ext.replace(".master", "")
                files.append({
                    "name": f,
                    "size": os.path.getsize(file_path),
                    "type": "pdf" if f.lower().endswith('.pdf') else "markdown",
                    "suggested_namespace": namespace,
                    "is_ingested": rag_engine.check_ingested(namespace)
                })
    return {"status": "success", "files": files}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """接收浏览器上传的 .md 或 .pdf 文件"""
    allowed_exts = (".md", ".pdf")
    if not any(file.filename.lower().endswith(ext) for ext in allowed_exts):
        raise HTTPException(status_code=400, detail="Only .md and .pdf files are allowed")
    
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"status": "success", "filename": file.filename, "message": "File uploaded successfully"}

@router.delete("/files/{filename}")
async def delete_file(filename: str):
    """删除 .md 文件"""
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        
        # 物理删除了该文档的同时，务必将其早先写入向量库（ChromaDB）的魂魄骨骸一并销毁！
        lecture_id = "lecture_" + filename.replace(".master.md", "")
        try:
            existing = rag_engine.vector_store.get(where={"lecture_id": lecture_id})
            if existing and existing.get("ids"):
                rag_engine.vector_store.delete(ids=existing["ids"])
                print(f"✅ [清道夫] 成功销毁 {filename} 残留在数据库中的灵魂碎片 ({len(existing['ids'])} 条)。")
            else:
                print(f"⚠️ [清道夫] {filename} 的物理文件已删，但向量库暂无记录。")
        except Exception as e:
            print(f"⚠️ [清道夫] 回收向量切片时出错，但不阻塞前台: {e}")
            
        return {"status": "success", "message": "File and corresponding vector chunks deleted"}
    raise HTTPException(status_code=404, detail="File not found")

@router.get("/files/{namespace}/chunks")
async def get_file_chunks(namespace: str):
    """透视特定 namespace 下的所有向量库切片"""
    try:
        chunks = rag_engine.get_namespace_chunks(namespace)
        return {"status": "success", "namespace": namespace, "total": len(chunks), "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def trigger_ingest(request: IngestRequest):
    """触发向量化入库（同步阻塞，便于前端获知确切完成时间）"""
    file_path = os.path.join(DATA_DIR, request.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found in data directory")
    
    try:
        # FastAPI 会自动在线程池中运行同步函数，不阻塞主事件循环
        ingest_master_md(file_path, request.lecture_id)
        
        return {
            "status": "success", 
            "message": f"{request.filename} 的切片与高维入库已经彻底完成！"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def process_chat(request: ChatRequest):
    """用于测试向量检索引擎的对话接口"""
    try:
        if (not request.question or not request.question.strip()) and not request.image_base64:
            raise HTTPException(status_code=400, detail="Question cannot be empty unless an image is provided")
            
        answer = await rag_engine.get_answer(
            question=request.question, 
            lecture_id=request.lecture_id,
            image_base64=request.image_base64
        )
        return {"status": "success", "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def process_chat_stream(request: ChatRequest):
    """流式 RAG 对话接口"""
    try:
        if (not request.question or not request.question.strip()) and not request.image_base64:
            raise HTTPException(status_code=400, detail="Question cannot be empty unless an image is provided")
            
        return StreamingResponse(
            rag_engine.get_answer_stream(
                question=request.question,
                lecture_id=request.lecture_id,
                image_base64=request.image_base64
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
