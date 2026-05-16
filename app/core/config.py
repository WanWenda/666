import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "NCU RAG AI Tutor"
    API_V1_STR: str = "/api/v1"
    GATEWAY_AUTH_TOKEN: str = "AIIA_LAB_NCU_RAG_SEC_2026_xYz"
    
    # WeCom Settings (企业微信对接)
    WECOM_CORP_ID: str = "your_corp_id"
    WECOM_CORP_SECRET: str = "your_corp_secret"
    WECOM_TOKEN: str = "your_webhook_token"
    WECOM_ENCODING_AES_KEY: str = "your_encoding_aes_key"
    # LLM Settings (Default to vLLM on 192.168.8.28, can be overridden by .env for SiliconFlow)
    LLM_API_BASE: str = "http://192.168.8.28:8000/v1"
    LLM_API_KEY: str = "EMPTY"
    LLM_MODEL_NAME: str = "Sehyo/Qwen3.5-35B-A3B-NVFP4" # 推理模型
    
    # Embedding Settings (Default to vLLM on Remote GB10)
    LLM_EMBEDDING_API_BASE: str = "http://192.168.8.28:8002/v1"
    LLM_EMBEDDING_MODEL_NAME: str = "Forturne/Qwen3-Embedding-8B-NVFP4"
    CHROMA_PERSIST_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma")
    
    # Reranker Settings (Default to vLLM on Remote GB10)
    RERANKER_ENDPOINT: str = "http://192.168.8.28:8003/v1/rerank"
    RERANKER_MODEL_NAME: str = "Forturne/Qwen3-Reranker-4B-NVFP4"
    
    # VLM Settings (Vision-Language Model on 192.168.8.6)
    VLM_API_BASE: str = "http://192.168.8.6:8004/v1"
    VLM_MODEL_NAME: str = "Qwen/Qwen3-VL-8B-Instruct-FP8"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
