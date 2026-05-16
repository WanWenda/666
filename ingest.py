import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.core.config import settings

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def ingest_master_md(file_path: str, lecture_id: str):
    print(f"开始处理文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"找不到文件: {file_path}")
        return

    md_text = ""
    use_markdown_splitter = True

    if file_path.lower().endswith('.pdf'):
        print("检测到 PDF 文件，正在提取文本...")
        try:
            import fitz
            doc = fitz.open(file_path)
            pages_text = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    pages_text.append(f"--- 第 {page_num + 1} 页 ---\n{text}")
                md_text = "\n\n".join(pages_text)
            page_count = len(doc)
            doc.close()
            print(f"PDF 提取完成，共 {page_count} 页，{len(md_text)} 字符")
            use_markdown_splitter = False
        except Exception as e:
            print(f"PDF 解析失败: {e}")
            return
    else:
        import chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            detected = chardet.detect(raw_data)
            encoding = detected.get('encoding', 'utf-8')
            print(f"检测到文件编码: {encoding} (置信度: {detected.get('confidence', 0):.0%})")
        md_text = raw_data.decode(encoding, errors='replace')

    if use_markdown_splitter:
        headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(md_text)
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN, chunk_size=1000, chunk_overlap=150
        )
        final_splits = text_splitter.split_documents(md_header_splits)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        final_splits = text_splitter.create_documents([md_text])

    print(f"切分完毕，共 {len(final_splits)} 个片段")

    for split in final_splits:
        split.metadata["lecture_id"] = lecture_id
        split.metadata["source"] = os.path.basename(file_path)

    print(f"正在请求远端生成语义向量 ({settings.LLM_EMBEDDING_MODEL_NAME})...")
    embeddings = OpenAIEmbeddings(
        openai_api_base=settings.LLM_EMBEDDING_API_BASE,
        openai_api_key=settings.LLM_API_KEY,
        model=settings.LLM_EMBEDDING_MODEL_NAME
    )

    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    print(f"写入本地向量数据库: {settings.CHROMA_PERSIST_DIR}")

    try:
        tmp_store = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        existing = tmp_store.get(where={"lecture_id": lecture_id})
        if existing and existing["ids"]:
            tmp_store.delete(ids=existing["ids"])
            print(f"已清理旧数据 ({len(existing['ids'])} 条)")
    except Exception as e:
        print(f"清理旧数据时跳过: {e}")

    Chroma.from_documents(
        documents=final_splits,
        embedding=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("注入完成！")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest.py <path_to_file> <lecture_id_namespace>")
        print("Example: python ingest.py data/xxx.md lecture_1")
        print("Example: python ingest.py data/xxx.pdf lecture_1")
        sys.exit(1)
    ingest_master_md(sys.argv[1], sys.argv[2])
