from fastapi import APIRouter, Request, Query
from app.services.rag_engine import rag_engine

router = APIRouter()

@router.get("/callback")
async def wecom_verify(
    msg_signature: str = Query(...),
    timestamp: str = Query(...),
    nonce: str = Query(...),
    echostr: str = Query(...)
):
    """
    企业微信机器人的第一步配置校验接口。
    当我们在企微后台填入回调 URL 时，企微会发来 GET 请求验证归属权。
    需要根据 WECOM_TOKEN 验证 echostr（此处仅为骨架，需加入加解密库 `WeChatCrypt` 解析 echostr）
    """
    # 模拟验证通过，直接返回解析出的 echostr (生产环境需要结合 AES 解密)
    return int(echostr) if echostr.isdigit() else echostr


@router.post("/callback")
async def wecom_receive_message(request: Request):
    """
    学生在微信群内 @RAG助教 提问时的真实回调入口
    企微会将问题包装成 XML 发到此。
    """
    body = await request.body()
    print("接收到企业微信群聊 XML / JSON 报文:", body)
    
    # [假定代码：XML 转字典并抽取出用户的文本与提问人所在讲次上下文]
    question = "什么是大语言模型？"
    lecture_context = "lecture_5" # 实际应通过微信群的特定参数映射或学生注册的会话进行 namespace 绑定
    
    # 交给本地的 LangChain 引擎去 ChromaDB 里搜索并传给 192.168.8.28 渲染反击
    answer = await rag_engine.get_answer(question=question, lecture_id=lecture_context)
    
    # 将 answer 包回企微需要的被动回复 XML 格式返回
    # 对于慢速响应，企微要求返回 200 空串，随后微服务主动去调企微发消息 API (避免超时假死)
    return {
        "status": "success",
        "action": "sent_reply_to_wecom_group",
        "generated_answer": answer
    }
