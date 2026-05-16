# NCU RAG Tutor — 智能伴读助手

这是一个基于 RAG（检索增强生成）技术的智能问答系统。你可以上传任何文档（课件、讲义、手册），然后向它提问，它会**严格基于你上传的文档内容**来回答，不会编造答案。

---

## 快速上手（3 步启动）

### 第 1 步：获取 SiliconFlow API Key

本系统需要调用云端大模型来理解文档和回答问题。API Key 是你访问大模型的凭证。

1. 打开浏览器，访问 [SiliconFlow 官网](https://siliconflow.cn)，注册并登录。
2. 进入控制台，找到「API 密钥」页面，复制你的 API Key。

> 新注册用户会获得免费额度，足够完成本次课堂实验。

### 第 2 步：配置 API Key

1. 在项目文件夹中，找到 `.env.test` 文件。
2. 将它**复制一份**并重命名为 `.env`（注意：文件名就是 `.env`，前面有一个点）。
3. 用任意文本编辑器打开 `.env`，找到这一行：
   ```
   LLM_API_KEY=your_siliconflow_api_key_here
   ```
4. 把 `your_siliconflow_api_key_here` 替换为你刚才复制的 API Key，保存文件。

> ⚠️ **安全提醒**：`.env` 文件包含你的密钥，不要上传到 GitHub 或分享给他人。

### 第 3 步：安装依赖并启动

打开终端（Mac 用「终端」，Windows 用「命令提示符」或「PowerShell」），依次执行以下命令：

```bash
# 进入项目目录（请替换为你的实际路径）
cd ncu-rag-tutor

# 创建 Python 虚拟环境（只需执行一次）
python -m venv venv

# 激活虚拟环境
# Mac / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装所有依赖（只需执行一次）
pip install -r requirements.txt

# 启动服务
python main.py
```

启动成功后，终端会显示类似 `Uvicorn running on http://0.0.0.0:8001` 的提示。

> 💡 **不用担心警告信息**：启动时你可能会看到黄色的 `LangChainDeprecationWarning` 警告和 `BM25 索引跳过` 提示，这些都是正常的，不影响使用。

### 打开网页界面

在浏览器中访问：**http://localhost:8001/admin/**

你会看到一个聊天界面，可以直接开始使用。

---

## 使用方法

### 上传文档（注入知识）

1. 在网页界面中找到「上传文档」按钮，选择你要上传的文件（支持 Markdown 格式）。
2. 上传完成后，文件会出现在文件列表中。**此时文档尚未生效。**
3. 点击文件旁边的「入库」按钮，等待系统提示完成。入库过程会将文档切片并存入知识库。
4. 入库完成后，你就可以针对文档内容提问了。

你也可以用命令行导入文档：
```bash
python ingest.py data/你的文档.md "文档名称"
```

### 提问

在聊天框中输入问题，系统会自动从你上传的文档中检索相关内容，基于检索到的信息生成回答。

### A/B 对比实验

1. **不上传文档时提问**：观察模型回答"不知道"或编造内容。
2. **上传文档后用同样的问题再问一次**：观察模型精确引用文档中的信息。

---

## 常见问题

**Q: 启动时报错 `ModuleNotFoundError`？**
A: 确认你已经激活了虚拟环境（终端提示符前面有 `(venv)` 字样），然后重新执行 `pip install -r requirements.txt`。
确认你已经激活了虚拟环[requirements.txt](requirements.txt)境（终端提示符前面有 `(venv)` 字样），然后重新执行 `pip install -r requirements.txt`。
**Q: 提问后模型回答很慢？**
A: 首次提问需要初始化向量数据库，可能需要 10-20 秒。后续提问会快很多。

**Q: 想清空知识库重新来过？**
A: 删除 `data/chroma` 文件夹，然后重启服务：
```bash
rm -rf data/chroma
python main.py
```

**Q: 端口 8001 被占用了？**
A: 关闭占用该端口的其他程序，或者修改 `main.py` 中的端口号。

---

## 技术架构（选读）

本项目采用 Agentic RAG 架构，核心流程如下：

```
用户提问 → 混合检索（BM25 关键字 + ChromaDB 向量）→ 交叉精排（Qwen3-Reranker）→ 大模型生成回答
```

- **混合检索**：同时使用关键字匹配和语义向量匹配，提高召回率。
- **交叉精排**：对检索结果进行二次排序，只保留最相关的内容送入大模型。
- **大模型生成**：基于检索到的文档片段生成回答，避免编造。

依赖的外部模型（均通过 SiliconFlow API 调用）：

| 用途 | 模型名称 |
|------|---------|
| 回答生成 | Qwen3.5-35B-A3B |
| 文本向量化 | Qwen3-Embedding-8B |
| 结果精排 | Qwen3-Reranker-8B |
| 图片识别（可选） | Qwen3-VL-8B-Instruct |
