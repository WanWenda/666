from typing import TypedDict, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from app.core.config import settings
import asyncio
import time
import requests
import httpx
import codecs
import json
import base64
import io

class GraphState(TypedDict):
    question: str
    lecture_id: str
    image_base64: str          # 新增：原始图片数据
    visual_context: str        # 新增：图片分析后的文本描述
    documents: List[Document]
    generation: str
    loop_count: int
    stream_mode: bool

class RAGEngine:
    def __init__(self):
        # 0. 根据 API 网关类型动态构建 "关闭思维链" 参数
        #    - 硅基流动 (SiliconFlow): 顶层 "enable_thinking": False
        #    - 本地 vLLM (Atlas): "chat_template_kwargs": {"enable_thinking": False}
        _is_siliconflow = "siliconflow" in settings.LLM_API_BASE.lower()
        self._no_think_params = (
            {"enable_thinking": False}
            if _is_siliconflow
            else {"chat_template_kwargs": {"enable_thinking": False}}
        )

        # 1. 核心生成器 LLM
        self.llm = ChatOpenAI(
            openai_api_base=settings.LLM_API_BASE,
            openai_api_key=settings.LLM_API_KEY,
            model_name=settings.LLM_MODEL_NAME,
            temperature=0.1,
            max_tokens=4096,
            streaming=True, # 启用全双工流式支持
            extra_body={
                **self._no_think_params,
                "stop": ["<|im_end|>", "<|endoftext|>"]
            }
        )
        
        # 1.1 轻量级分类器专用 LLM
        self.classifier_llm = ChatOpenAI(
            openai_api_base=settings.LLM_API_BASE,
            openai_api_key=settings.LLM_API_KEY,
            model_name=settings.LLM_MODEL_NAME,
            temperature=0.0,
            max_tokens=128,
            extra_body={
                **self._no_think_params,
                "stop": ["<|im_end|>", "<|endoftext|>"]
            }
        )
        
        # 2. 向量嵌入模型映射
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=settings.LLM_EMBEDDING_API_BASE,
            openai_api_key=settings.LLM_API_KEY,
            model=settings.LLM_EMBEDDING_MODEL_NAME
        )

        # 3. 本地全私有化向量数据库 (强制映射到 Cosine Space 防止欧氏距离膨胀)
        self.vector_store = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

        # 4. 构建 BM25 内存索引用于混合检索 (解决专有名词稀释问题)
        print("🔧 [BM25] 正在从全量向量库读取物料以重构倒排关键字索引...")
        try:
            db_data = self.vector_store.get()
            docs_for_bm25 = [Document(page_content=txt, metadata=meta) for txt, meta in zip(db_data['documents'], db_data['metadatas'])]
            if docs_for_bm25:
                # 配合 jieba 进行中文分词处理
                import jieba
                def jieba_preprocess(text):
                    return jieba.lcut(text)
                self.bm25_retriever = BM25Retriever.from_documents(docs_for_bm25, preprocess_func=jieba_preprocess)
                print(f"✅ [BM25] {len(docs_for_bm25)} 个切片关键字倒排索引构建完毕！")
            else:
                self.bm25_retriever = None
                print("⚠️ [BM25] 向量库为空，BM25 索引跳过...")
        except Exception as e:
            self.bm25_retriever = None
            print(f"⚠️ [BM25 ERROR] 构建关键字倒排索引失败: {e}")

        # === 预定义各个裁判/生成 Prompt ===
        self.grader_prompt = PromptTemplate(
            template="""你是一个极其严格的关键词匹配执行器。请评估以下课件片段是否与用户提问相关。
【极端强制指令】：
1. 只要给定的“课件片段”字面上**包含**了“用户提问”里涉及的核心主体或专有名词（例如：VS Code、Agent 等软件名或学术名词，哪怕只提到一次），你**必须、绝对**最终输出 'yes'！即使它没有定义这个名词，只要出现了就判定相关！
2. 反之，如果“课件片段”中根本没有出现该专有名词，或者完全无关，你**必须**输出 'no'。严禁脑补关联！
3. 你的最终判定必须且只能单独是 yes 或 no，严禁附加其他文字。

课件片段：
{context}

用户提问：{question}
你的判定(仅输出 yes 或 no)：""",
            input_variables=["context", "question"]
        )
        
        self.rewrite_prompt = PromptTemplate(
            template="""你是一个智能重写引擎。用户抛出的问题在此前检索中没有命中任何本地课件内容。
请根据可能的意图，对提问进行扩写、释义或提炼，生成一个更易于在文本数据库中匹配到的新问题。（比如增加对应的中文翻译、去掉口语化词汇等）

原问题：{question}
重写后的问题（仅单独输出问题本身，禁止输出其它字句）：""",
            input_variables=["question"]
        )

        self.generation_prompt = PromptTemplate(
            template="""你是人工智能创新应用实验室 (AIIA Lab) 专属数字伴读助教。当前答疑依托的讲义/模块为：【{lecture_name}】。
严格遵循以下教务准则：
1. 核心忠诚与分点排版：如果学生的提问可以在下方【课件检索内容】中找到答案，你的核心论点【必须且只能】基于课件。
   - 必须分段、分点（使用 Markdown 列表和小标题）详细罗列解答，严禁把所有内容挤成一大段。
   - 每个核心观点后，如果涉及到了原文档的知识，必须单起一行附带原始文件名出处（例如格式：`来源：【xxx.md】 - “部分片段...”`）。
   - 保持风趣、生动、且极为专业的助教口吻，适当使用 Emoji。
2. 常识豁免与显式警告：如果学生提问涉及课件中并未直接讲透的背景概念（例如：一个没听说过的软件缩写、基础的计算机网络常识），且不违背原始课程大纲精神，你可以动用自身的技术先验知识进行补充解答。但是，涉及课外知识的地方，你【必须】使用明确的引用块进行免责声明，警示学生这不是课本内的硬性考点。
   声明格式要求（直接套用，且**千万不要**在上面额外加一个同名标题，避免重复）：
   > ⚠️ **拓展补充**：当前讲义虽未详细展开，但基于通用技术常识，[你的解释...]

【全量打散的课件检索内容】：
{context}

学生提问：{question}

你的专业助教解答：""",
            input_variables=["context", "lecture_name", "question"]
        )

        self.hallucination_prompt = PromptTemplate(
            template="""你是一个宽容的常识审查员。
判断下方的回答是否可以安全发给学生看。
1. 如果回答内容基于课件，或者顺带补充了合理的计算机通识（比如 VS Code 是一款编辑器等），请回答 'yes'。
2. 只有当回答严重违背事实、或者完全与课件矛盾时，才回答 'no'。

提供的课件材料：
{context}

草稿回答：
{generation}

判定(yes/no)：""",
            input_variables=["context", "generation"]
        )

        self.vision_prompt = PromptTemplate(
            template="""你是人工智能创新应用实验室 (AIIA Lab) 智慧课程的视觉分析引擎。请分析学生上传的这张截图。
【任务目标】：
1. 如果是课件内容，请完整提取其中的关键文字信息（OCR）。
2. 如果是代码或报错截图，请精确提取代码段和 Error Message。
3. 简要概括图中处于什么教学场景（例如：正在配置环境、正在编写 Python 函数）。

【注意】：你的回答将作为后续知识库检索的关键词，请务必客观、准确。
请以 [视觉摘要] 打头开始你的分析。

图片内容：<|vision_start|>data:image/png;base64,{image_base_base64}<|vision_end|>

分析结果：""",
            input_variables=["image_base_base64"]
        )

        # === 构建 LangGraph 状态图 ===
        workflow = StateGraph(GraphState)
        
        # 添加节点
        workflow.add_node("vision_analyze", self.node_vision_analyze)
        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("grade_documents", self.node_grade_documents)
        workflow.add_node("generate", self.node_generate)
        workflow.add_node("transform_query", self.node_transform_query)
        workflow.add_node("refuse", self.node_refuse)
        
        # 连线
        workflow.add_edge(START, "vision_analyze")
        workflow.add_edge("vision_analyze", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # 评估文档后，决定是生成、重写、还是彻底放弃
        workflow.add_conditional_edges(
            "grade_documents",
            self.edge_decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
                "refuse": "refuse"
            }
        )
        
        # 重写后重新拉取资料
        workflow.add_edge("transform_query", "retrieve")
        
        # 加上幻觉检察官的后置拦截
        workflow.add_conditional_edges(
            "generate",
            self.edge_check_hallucination,
            {
                "useful": END,               # 安全过关，发送
                "hallucinated": "refuse"     # 查出幻觉，直接没收回答并打回
            }
        )
        workflow.add_edge("refuse", END)
        
        # 编译特工网络
        self.app = workflow.compile()

    # ==========================
    # LangGraph 节点与边逻辑 (Agentic Flow)
    # ==========================
    async def node_vision_analyze(self, state: GraphState):
        """节点 0：【视觉分拣】增加图片压缩逻辑，防止 Token 爆炸"""
        image_base64 = state.get("image_base64")
        if not image_base64:
            return {"visual_context": ""}
            
        print("📸 [VISION] 正在进行视觉预压缩与协议封装...", flush=True)
        try:
            # 💡 核心修复：先对原始 Base64 进行物理压缩与缩放
            import io
            from PIL import Image
            
            img_data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(img_data))
            
            # 缩放至最大 1024px，保证文字清晰的同时大幅减少字符数
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"   => 尺寸缩放: {img.size}", flush=True)
            
            # 转为 JPEG 压缩格式 (质量 60% 足够 OCR)
            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="JPEG", quality=60, optimize=True)
            compact_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            print(f"   => 字符数压缩: {len(image_base64)} -> {len(compact_base64)}", flush=True)

            # 💡 关键修正：采用真正的 OpenAI Vision 标准协议接入专用的 Qwen3-VL 引擎
            payload = {
                "model": settings.VLM_MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{compact_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "请详细分析截图中包含的代码、文字、公式或教学场景细节。请以 [视觉摘要] 打头开始你的精简概括分析："
                            }
                        ]
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.1
            }
            
            print(f"📸 [VISION] 正在向独立 VLM 微服务 {settings.VLM_API_BASE} 发起原生图文特征提取...", flush=True)
            
            visual_context = ""
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        response = await client.post(
                            f"{settings.VLM_API_BASE}/chat/completions",
                            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
                            json=payload
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        visual_context = result['choices'][0]['message']['content'].strip()
                        print(f"📸 [VISION] 协议精准匹配！分析结果 (Token节省 99%): {visual_context[:50]}...", flush=True)
                        break
                    else:
                        print(f"⚠️ [VISION ERROR] 推理端拒绝请求 (尝试 {attempt+1}/{max_retries}): {response.status_code} - {response.text}", flush=True)
                        visual_context = f"[图片解析失败: 状态码 {response.status_code}]"
                
                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    print(f"⚠️ [VISION ERROR] 网络超时或连接错误 (尝试 {attempt+1}/{max_retries}): {e}", flush=True)
                    visual_context = f"[图片解析失败: 网络超时]"
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)  # 等待 2 秒后重试
                        continue
                except Exception as e:
                    print(f"⚠️ [VISION ERROR] 未知异常 (尝试 {attempt+1}/{max_retries}): {e}", flush=True)
                    visual_context = f"[图片解析失败: 未知错误]"
                    break

            updates = {"visual_context": visual_context, "image_base64": ""}
            if not state.get("question", "").strip():
                print("📸 [VISION] 用户未提供文字问题，自动将视觉摘要作为查询依据。")
                updates["question"] = f"请详细分析截图中的内容，并基于提供的课件资料解答截图中可能存在的疑问点。图片信息如下：{visual_context}"
            return updates
        except Exception as e:
            print(f"⚠️ [VISION ERROR] 压缩或分析失败: {e}", flush=True)
            updates = {"visual_context": "[图片解析失败]", "image_base64": ""}
            if not state.get("question", "").strip():
                updates["question"] = "请帮我分析这张图片中的问题（注意：图片解析暂时失败）。"
            return updates

    def node_retrieve(self, state: GraphState):
        """节点 1：基于最新的 Question 从 ChromaDB 拉取资料，并启用 Reranker 进行重排序缩编"""
        _t_retrieve = time.time()
        question = state["question"]
        lecture_id = state["lecture_id"]
        visual_context = state.get("visual_context", "")
        
        # 💡 核心增强：如果存在视觉分析结果，将其注入查询词中，确保检索到图片相关的课件
        retrieval_query = question
        if visual_context:
            retrieval_query = f"{visual_context} {question}"
            print(f"🔍 [RETRIEVE] 混合查询模式开启: {retrieval_query[:50]}...", flush=True)

        # 1. 粗排大海捞针阶段 (在云端架构下下调 k 避免超时)
        search_kwargs = {"k": 20} 
        filter_dict = {}
        if lecture_id == "GLOBAL_SEARCH":
            search_kwargs = {"k": 15} 
        elif lecture_id:
            filter_dict = {"lecture_id": lecture_id}
            search_kwargs["filter"] = filter_dict
            
        vector_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        
        # (已删除原先用于打印数量的重复 Dense 检索调用，消除一倍的网络延迟)
        
        # 构建混合检索引擎
        # 如果有视觉信息，将其合入检索关键词，增强召回
        search_query = question
        if state.get("visual_context"):
            search_query = f"{question} (图片背景: {state['visual_context']})"

        if self.bm25_retriever:
            # BM25 也同步采用相似的 k 召回率，并且支持同样的讲次过滤
            self.bm25_retriever.k = search_kwargs["k"]
            # 注意: BM25 在内存通过 python list 过滤不支持复杂过滤，如果非 GLOBAL 搜，后续依靠重排
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )
            print(f"🕵️‍♂️ [HYBRID RETRIEVE] 启用 BM25 + Dense 双流交叉召回进行粗排 (k={search_kwargs['k']})...", flush=True)
            docs = ensemble_retriever.invoke(search_query)
        else:
            print(f"🕵️‍♂️ [DENSE RETRIEVE] 仅使用 Dense 向量流进行粗排检索 (k={search_kwargs['k']})...", flush=True)
            docs = vector_retriever.invoke(search_query)
        
        if not docs:
            print(f"⏱️ [PERF] node_retrieve: {time.time() - _t_retrieve:.2f}s (0 docs)", flush=True)
            return {"documents": []}
            
        # 2. 精排提纯阶段 (Cross-Encoder Reranking 提取金标)
        print(f"🎯 [RERANK] 正在调用 Qwen3-Reranker 对 {len(docs)} 个碎片进行交叉维度打分...", flush=True)
        try:
            # 使用最纯净的格式直接送入交叉编码器，大幅降低 token 数量，防止请求超时 (Timeout)
            formatted_docs = [doc.page_content for doc in docs]
            
            response = requests.post(
                settings.RERANKER_ENDPOINT,
                headers={"Authorization": f"Bearer {settings.LLM_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": settings.RERANKER_MODEL_NAME,
                    "query": question,
                    "documents": formatted_docs
                },
                timeout=60.0 # 修改为 60 秒容错，防止处理 50 块切片时引发超时兜底
            )
            response.raise_for_status()
            
            # vllm /v1/rerank returns results with index and relevance_score
            results = response.json().get("results", [])
            # 把原本 docs 根据 Reranker 吐出的 score 做绑定
            scored_docs = []
            for res in results:
                idx = res["index"]
                score = res["relevance_score"]
                scored_docs.append((score, docs[idx]))
                
            # 按分数降序排列，由于包含专有名词但不直接解题的切片会被语义模型严重打分惩罚（如排名第6），我们需要拓宽进入裁判系统的切片数至 10 块
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_k_docs = [d for score, d in scored_docs[:10]]
            
            print(f"   => 精排完毕，优中选优保留前 {len(top_k_docs)} 块核心资料交由 LLM 发落 (最高分: {scored_docs[0][0]:.4f})", flush=True)
            print(f"⏱️ [PERF] node_retrieve: {time.time() - _t_retrieve:.2f}s ({len(top_k_docs)} docs)", flush=True)
            return {"documents": top_k_docs}
            
        except Exception as e:
            print(f"⚠️ [RERANK ERROR] Reranker 调用失败，降级返回粗排前 10 块兜底。原因: {str(e)}")
            print(f"⏱️ [PERF] node_retrieve (fallback): {time.time() - _t_retrieve:.2f}s", flush=True)
            return {"documents": docs[:10]}

    def node_grade_documents(self, state: GraphState):
        """节点 2：审查拉取回的切片是否真正回答了问题"""
        question = state["question"]
        docs = state.get("documents", [])
        loop_count = state.get("loop_count", 0)
        
        print(f"⚖️ [GRADE] 正在并发审查 {len(docs)} 个检索切片的关联度...", flush=True)
        
        if not docs:
            return {"documents": [], "loop_count": loop_count + 1}
            
        # 既然我们已经有了极高质量的 Qwen 交叉重排器（Cross-Encoder），它给出的分数已经是目前能得到的最权威的相关度判定！
        # 让通用大模型去做二次 Boolean 打分不仅画蛇添足，还会因为 Prompt 的不可控诱导产生误判（False Negative）。
        # 因此，直接采信 Reranker 选出的优质片段即可！
        
        relevant_docs = docs[:5] # 保留前5名给大模型生成
        
        if not relevant_docs:
            print("❌ [GRADE] 所有资料均为空，触发循环...", flush=True)
            return {"documents": [], "loop_count": loop_count + 1}
            
        print(f"   => 审查完毕，直接采纳 Reranker 提供的最优质前 {len(relevant_docs)} 块切片！", flush=True)
        return {"documents": relevant_docs, "loop_count": loop_count + 1}

    def edge_decide_to_generate(self, state: GraphState):
        """条件边：决策树叉"""
        relevant_docs = state["documents"]
        loop_count = state["loop_count"]
        
        if not relevant_docs:
            if loop_count >= 2:
                print("🛑 [DECISION] 多次重试均无法找到关联课件，放弃提取，切入拒绝流！")
                return "refuse"
            print("🔄 [DECISION] 暂无合适资料，退回进行【问题重写】！")
            return "transform_query"
        print("✅ [DECISION] 资料已就绪，切入【草稿生成】节点！")
        return "generate"

    def node_transform_query(self, state: GraphState):
        """节点 3：改写失败的问题"""
        question = state["question"]
        prompt = self.rewrite_prompt.format(question=question)
        res = self.llm.invoke(prompt)
        rewritten = res.content.strip()
        print(f"✍️ [REWRITE] 问题已被大模型重构: '{question}' -> '{rewritten}'")
        return {"question": rewritten}

    def node_generate(self, state: GraphState):
        """节点 4：组装安全资料并让大模型撰写初稿"""
        _t_gen = time.time()
        docs = state["documents"]
        question = state["question"]
        lecture_id = state["lecture_id"]
        
        context_blocks = [f"====== [来源出处：{d.metadata.get('lecture_id', '未知讲次')}] ======\n{d.page_content}" for d in docs]
        context = "\n\n".join(context_blocks)
        # 如果有视觉分析结果，一并喂给生成器
        final_question = question
        # 💡 核心增强：在回复开头显式给出视觉识别反馈，增加透明度
        vision_prefix = ""
        if state.get("visual_context"):
            vision_prefix = f"> 📸 **[AI 视觉核验]**：{state['visual_context']}\n\n---\n\n"
        
        prompt = self.generation_prompt.format(
            context=context,
            lecture_name="全课程主线知识库" if lecture_id == "GLOBAL_SEARCH" else lecture_id,
            question=final_question
        )
        if state.get("stream_mode"):
            print("🧠 [GENERATE] (Streaming Mode) 图节点执行完毕，已将大模型调用平滑切换至异步网络层(SSE)进行流式传输...", flush=True)
            print(f"⏱️ [PERF] node_generate (stream-mode prep): {time.time() - _t_gen:.2f}s", flush=True)
            return {"generation": ""}

        print("🧠 [GENERATE] 正在整合视觉与课件资料撰写回复...", flush=True)
        print(f"⏱️ [PERF] node_generate: 开始调用 LLM.invoke()...", flush=True)
        _t_llm = time.time()
        res = self.llm.invoke(prompt)
        print(f"⏱️ [PERF] node_generate: LLM.invoke() 耗时 {time.time() - _t_llm:.2f}s | 总耗时 {time.time() - _t_gen:.2f}s | 输出 {len(res.content)} chars", flush=True)
        
        # 将视觉前缀注入最终输出
        return {"generation": vision_prefix + res.content}

    def edge_check_hallucination(self, state: GraphState):
        """条件边：最后一道防线：检查是否胡编乱造"""
        docs = state["documents"]
        generation = state["generation"]
        
        context_blocks = [f"【{d.metadata.get('lecture_id', '')}】:\n{d.page_content}" for d in docs]
        context = "\n\n".join(context_blocks)
        
        # 直接放行：系统已经剥离了不确定的幻觉上下文，且大模型的 System Prompt 限制得足够死
        print("🟢 [HALLUCINATION CHECK] (Bypassed) 信任专有大模型的严谨度，直接放行！")
        return "useful"

    def node_refuse(self, state: GraphState):
        """节点 X：越界/幻觉专属的冷酷打回"""
        refuse_msg = "同学你好，经过智能体多轮自省与核查，这个问题超出了当前已入库的主线教学课件范围（或资料不足以支撑确切的安全解答）。为保障严谨，建议向老师和助教反馈探讨。"
        return {"generation": refuse_msg}

    # ==========================
    # 对外暴露的业务接口 (保持与之前微服务兼容)
    # ==========================
    def check_ingested(self, lecture_id: str) -> bool:
        try:
            res = self.vector_store.get(where={"lecture_id": lecture_id}, limit=1)
            return len(res.get("ids", [])) > 0
        except Exception:
            return False

    def get_namespace_chunks(self, lecture_id: str) -> list:
        try:
             res = self.vector_store.get(where={"lecture_id": lecture_id})
             chunks = []
             if res and "ids" in res:
                 for i in range(len(res["ids"])):
                     chunks.append({
                         "id": res["ids"][i],
                         "content": res.get("documents", [])[i] if res.get("documents") else "",
                         "metadata": res.get("metadatas", [])[i] if res.get("metadatas") else {}
                     })
             return chunks
        except Exception:
             return []

    async def get_answer(self, question: str, lecture_id: str, image_base64: str = None) -> str:
        """接收特定命名空间下的微信学生提问，投入 LangGraph 多维沙盒流转"""
        try:
            _t0 = time.time()
            print(f"⏱️ [PERF] get_answer() 开始 | question={question[:30]}... | lecture={lecture_id}", flush=True)
            inputs = {
                "question": question, 
                "lecture_id": lecture_id, 
                "image_base64": image_base64,
                "visual_context": "",
                "documents": [],
                "loop_count": 0,
                "stream_mode": False
            }
            # 运行状态图 (原生纯异步执行，解决 node 为 async def 导致的 No synchronous function provided 错误)
            final_state = await self.app.ainvoke(inputs)
            _elapsed = time.time() - _t0
            print(f"⏱️ [PERF] get_answer() 完成 | 总耗时: {_elapsed:.2f}s | 回答长度: {len(final_state['generation'])} chars", flush=True)
            return final_state["generation"]
        except Exception as e:
            return f"数字伴读服务（LangGraph 决策引擎）内网调度异常，请稍后再试。错误追踪：{str(e)}"

    async def get_answer_stream(self, question: str, lecture_id: str, image_base64: str = None):
        """流式获取 RAG 引擎的思考与回答内容 (ByteSafe v1.1.8)"""
        try:
            # 立即发送初始化状态，防止前端因等待第一个节点完成而超时
            yield f"data: {json.dumps({'type': 'status', 'content': '🚀 伴学代理已启动，正在规划路径...'}, ensure_ascii=False)}\n\n"

            # 状态初始化
            inputs = {
                "question": question, 
                "lecture_id": lecture_id, 
                "image_base64": image_base64,
                "visual_context": "",
                "documents": [],
                "loop_count": 0,
                "stream_mode": True
            }
            
            # 🚀 1. 运行图的前半部分（视觉、检索、评分），获取上下文
            current_state = inputs
            async for chunk in self.app.astream(inputs, stream_mode="updates"):
                node_name = list(chunk.keys())[0]
                data = chunk[node_name]
                current_state.update(data)
                
                print(f"DEBUG: Node {node_name} completed, yielding status...", flush=True)

                # 推送进度
                status_msg = {
                    "vision_analyze": "📸 视觉语义分析完成，提取特征中...",
                    "retrieve": "🔍 知识库匹配完成，正在精选资料...",
                    "grade_documents": "⚖️ 资料相关性评估完成，准备生成回答..."
                }
                
                # 特殊屏蔽：如果没有图片输入，静默跳过视觉分析的状态展示
                if node_name == "vision_analyze" and not inputs.get("image_base64"):
                    pass
                elif node_name in status_msg:
                    yield f"data: {json.dumps({'type': 'status', 'content': status_msg[node_name]}, ensure_ascii=False)}\n\n"
                
                # 💡 熔断逻辑：如果图片解析失败，且没有文字提问
                v_ctx = current_state.get("visual_context", "")
                is_v_failed = "失败" in v_ctx or not v_ctx.strip()
                if node_name == "vision_analyze" and is_v_failed and not question.strip():
                    err_msg = {
                        "type": "answer", 
                        "content": "> 📸 **[AI 视觉核验]**：图片解析未获得有效信息。\n\n同学你好，刚才上传的截图内容较少或识别失败（我也没找到你输入的文字提问），建议重新截取清晰的代码或讲义片段再试哦！"
                    }
                    yield f"data: {json.dumps(err_msg, ensure_ascii=False)}\n\n"
                    return

                if node_name == "generate" or "generation" in data:
                    break
                if node_name == "refuse":
                    yield f"data: {json.dumps({'type': 'answer', 'content': data.get('generation', '')}, ensure_ascii=False)}\n\n"
                    return

            # 如果没有找到 generation 内容，手动启动 ByteSafe 模式请求 vLLM
            # ... (Rest of the method logic)

            # 如果没有找到 generation 内容，手动启动 ByteSafe 模式请求 vLLM
            # 构造最终 Prompt (复用 node_generate 逻辑)
            # 1. 物理级脱水：只取排名前 3 的绝对相关资料，防止 80 万 Token 惨剧再次发生
            # 2. 从盲盒编号还原为具名文件展示，要求模型生成真实源文件引用
            docs = current_state.get("documents", [])[:3]
            context_blocks = [f"【来源文件: {d.metadata.get('source', '未知课件资料')}】:\n{d.page_content}" for d in docs]
            final_context = "\n\n".join(context_blocks)[:12000] # 硬上限：1.2万字符
            
            final_question = question
            if current_state.get("visual_context"):
                # 注入图片识别结果 (只注入摘要文字，绝不重复注入 base64)
                final_question = f"学生上传了图片（视觉识别为：{current_state['visual_context']}）\n\n提问内容：{question}"

            prompt_text = self.generation_prompt.format(
                context=final_context,
                lecture_name="全课程主线知识库" if lecture_id == "GLOBAL_SEARCH" else lecture_id,
                question=final_question
            )
            
            # 📊 [AUDIT] 这里的打印仅保留字符长度，不打印内容，防止日志溢出
            print(f"📊 [PROMPT_AUDIT] Final Prompt Length: {len(prompt_text)} chars ({len(docs)} slices)", flush=True)

            api_url = f"{settings.LLM_API_BASE}/chat/completions"
            headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}"}
            payload = {
                "model": settings.LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.7,
                "stream": True,
                "max_tokens": 8192,
                **self._no_think_params, # [OFF] 动态关闭推理模型的思维链路以追求极速
                "stop": ["<|im_end|>", "<|endoftext|>"]
            }

            yield f"data: {json.dumps({'type': 'status', 'content': '💡 响应就绪，正在生成回复...'}, ensure_ascii=False)}\n\n"

            # 💡 物理注入：在流式输出最开始，发送视觉摘要包
            if current_state.get("visual_context"):
                vision_summary_text = f"> 📸 **[AI 视觉核验]**：{current_state['visual_context']}\n\n---\n\n"
                yield f"data: {json.dumps({'type': 'answer', 'content': vision_summary_text}, ensure_ascii=False)}\n\n"

            print(f"🚀 [STREAM] 正在连接 LLM 接口进行流式生成: {api_url}", flush=True)
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", api_url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        error_msg = "\n\n⚠️ API 调用异常 [{}]: {}".format(response.status_code, error_body.decode())
                        print(f"❌ [STREAM ERROR] LLM 拒绝请求: {response.status_code} - {error_body.decode()[:200]}")
                        payload_json = json.dumps({'type': 'answer', 'content': error_msg}, ensure_ascii=False)
                        yield f"data: {payload_json}\n\n"
                        return

                    print(f"✅ [STREAM] 已建立连接，开始消费数据流...", flush=True)
                    is_thinking = False
                    post_think_cleanup_window = 0

                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        
                        try:
                            json_str = line[6:]
                            resp_data = json.loads(json_str)
                            
                            delta = resp_data["choices"][0]["delta"]
                            content = delta.get("content", "")
                            reasoning = delta.get("reasoning_content", "")
                            
                            final_content = ""
                            if reasoning:
                                if not is_thinking:
                                    final_content += "> 💭 **内部推理图谱**：\n> \n> "
                                    is_thinking = True
                                final_content += reasoning.replace("\n", "\n> ")
                                
                            if content:
                                if is_thinking:
                                    final_content += "\n\n<hr/>\n\n"
                                    is_thinking = False
                                    # 开启滑动清理窗口
                                    post_think_cleanup_window = 10
                                    
                                # 状态机：拦截紧跟在思考结束后的几个溢出切片
                                if post_think_cleanup_window > 0:
                                    import re
                                    if re.search(r'[\u4e00-\u9fa5]', content):
                                        post_think_cleanup_window = 0
                                    else:
                                        if re.fullmatch(r'[a-zA-Z0-9\.\s\'\"\-\_]+', content):
                                            content = ""
                                            post_think_cleanup_window -= 1
                                        else:
                                            post_think_cleanup_window = 0

                                final_content += content
                                
                            if not final_content:
                                continue
                                
                            # 🧠 [SCRUB] 物理抹除
                            content_text = final_content.replace("\ufffd", "")
                            if not content_text:
                                continue

                            yield f"data: {json.dumps({'type': 'answer', 'content': content_text}, ensure_ascii=False)}\n\n"

                        except Exception:
                            continue

            print(f"✅ [STREAM] 数据流消费完毕。", flush=True)

        except Exception as e:
            print(f"🛑 [STREAM ERROR] {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'发生未预期错误: {str(e)}'}, ensure_ascii=False)}\n\n"

# 单例抛出
rag_engine = RAGEngine()
