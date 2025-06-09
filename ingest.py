# ingest.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 定义数据和持久化目录
DATA_PATH = "data"
VECTOR_DB_PATH = "chroma_db"
OLLAMA_EMBED_MODEL = "llama3" # 确保Ollama中已拉取此模型

def ingest_documents():
    """
    加载PDF文档，分割文本，生成嵌入，并存储到ChromaDB。
    """
    print("--- 1. 加载文档 ---")
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"已加载: {filename}")

    if not documents:
        print(f"在 '{DATA_PATH}' 目录中未找到任何PDF文件。请确保放置了文档。")
        return

    print(f"共加载 {len(documents)} 页文档。")

    print("\n--- 2. 分割文本 ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # 每个文本块的最大字符数
        chunk_overlap=100 # 文本块之间的重叠字符数，有助于保留上下文
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文档已被分割成 {len(chunks)} 个文本块。")

    print("\n--- 3. 初始化嵌入模型 ---")
    # 使用 Ollama 作为嵌入模型，指定模型名称
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    print(f"Ollama 嵌入模型 '{OLLAMA_EMBED_MODEL}' 已加载。")

    print("\n--- 4. 创建并持久化向量数据库 ---")
    # 从文本块创建 Chroma 向量数据库，并指定持久化目录
    # 如果目录不存在，Chroma 会创建它
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    print(f"向量数据库已成功创建并保存到 '{VECTOR_DB_PATH}'。")
    print("数据摄入完成！")

if __name__ == "__main__":
    ingest_documents()