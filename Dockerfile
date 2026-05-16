# 使用国产友好且纯净的 Python 构建底座（内网隔离部署）
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.11-slim

WORKDIR /app

# 安装必要的系统依赖 (如用于编译某些 C 库和 chromadb)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 设置 HuggingFace 国内加速防墙镜像，并将模型缓存到持久化数据盘
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HOME=/app/data/huggingface_cache

# 替换为国内 pip 源并利用 BuildKit 缓存构建，防止每次断层都重新下载几百兆依赖
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

# 微服务默认通过 8001 端口暴露
EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
