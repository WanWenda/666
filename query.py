import sys
import asyncio
import warnings
warnings.filterwarnings("ignore")

# 引入核心中枢
from app.services.rag_engine import rag_engine

async def main():
    if len(sys.argv) < 3:
        print("Usage: python query.py <lecture_id> <question>")
        print("Example: python query.py lecture_1 '什么是符号主义？'")
        return
        
    lecture_id = sys.argv[1]
    question = sys.argv[2]
    
    print(f"\n🔍 [检索开始] 正在 {lecture_id} 命名空间下查阅官方讲义...")
    print(f"👤 学生提问：{question}\n")
    
    # 模拟企业微信收到提问后，调用大模型
    answer = await rag_engine.get_answer(question=question, lecture_id=lecture_id)
    
    print("====== RAG Tutor 零幻觉官方解答 ======")
    print(answer)
    print("======================================\n")

if __name__ == "__main__":
    asyncio.run(main())
