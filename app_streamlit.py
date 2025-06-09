import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings # 假设你已经按照建议更新了导入
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 配置 ---
VECTOR_DB_PATH = "chroma_db"
OLLAMA_LLM_MODEL = "llama3"
OLLAMA_EMBED_MODEL = "llama3" # 必须与 ingest.py 中使用的模型一致

# --- LangSmith 配置（可选，用于禁用追踪） ---
# 如果你不使用 LangSmith，可以取消下面的注释，将 LANGCHAIN_TRACING_V2 设置为 "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

# --- 缓存 RAG 系统组件，避免每次运行时都重新加载 ---
@st.cache_resource
def setup_rag_system():
    """
    初始化RAG系统的所有组件：LLM，嵌入模型，向量数据库，检索器和RAG链。
    这个函数会被Streamlit缓存，只在第一次运行时执行。
    """
    st.info("💡 正在加载或初始化RAG系统，请稍候...")
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
        vector_store = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        st.success("✔ 向量数据库加载成功。")
    except Exception as e:
        st.error(f"❌ 加载向量数据库失败: {e}")
        st.warning("请确保已运行 `python ingest.py` 脚本并成功创建了 `'chroma_db'` 目录。")
        st.stop() # 停止应用运行，直到用户解决问题

    # 修改1: 移除 streaming=True 参数
    llm = Ollama(model=OLLAMA_LLM_MODEL, temperature=0.5)
    st.success(f"✔ Ollama LLM 模型 '{OLLAMA_LLM_MODEL}' 已加载。")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )
    st.success("✔ 检索器已设置。")

    prompt_template = PromptTemplate(
        template="""
        你是一个知识问答助手。请根据提供的上下文信息来回答问题。
        如果上下文中没有足够的信息来回答问题，请说明你无法找到相关信息，
        不要凭空捏造答案。

        上下文:
        {context}

        问题:
        {input}

        答案:
        """,
        input_variables=["context", "input"]
    )

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    st.success("✔ RAG 链已构建完成。")
    st.info("🎉 系统已准备就绪！")
    return retrieval_chain

# --- Streamlit UI 界面 ---
st.set_page_config(page_title="本地RAG知识问答系统", page_icon="📚")
st.title("📚 本地RAG知识问答系统 (Ollama + LangChain)")

# 初始化RAG链
rag_chain = setup_rag_system()

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "你好！我是你的知识问答助手，请提出你的问题。"})

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 获取用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 将用户消息添加到聊天历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示思考中的助手消息
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # 用于动态更新消息
        full_response = ""
        retrieved_docs_info = [] # 用于存储检索到的文档信息，以便后续引用来源

        with st.spinner("正在检索并生成答案，请稍候..."): # 确保 spinner 在流式输出前显示
            try:
                # 修改2: 迭代 RAG 链的流式输出
                for chunk in rag_chain.stream({"input": prompt}):
                    # LangChain 0.2.x 版本的 stream() 方法对于 create_retrieval_chain
                    # 返回的 chunk 结构可能与早期版本不同。
                    # 它通常会返回一个包含 'answer' 或 'output' 键的字典。
                    # 'context' 通常会在整个流结束后作为单独的 chunk 或在某个特定chunk中提供。
                    # 我们这里假设 'answer' 键是持续的文本流。

                    # LangChain 0.2.x 的 Chain.stream() 通常会将最终答案和检索到的context
                    # 分别作为不同的 chunk 返回，或者在某些中间步骤中。
                    # 对于 create_retrieval_chain, 'answer' 是最终文本，'context' 是检索结果。
                    
                    if "answer" in chunk: # 这是一个包含最终答案的 chunk
                        full_response += chunk["answer"]
                        message_placeholder.markdown(full_response + "▌") # 模拟光标
                    
                    # 也可以尝试捕获 context (如果它在流中作为单独的 chunk 出现)
                    # 然而，通常 context 是在生成答案之前就被确定好的，而不是流式输出的。
                    # 为了确保能捕获到 context，我们可能需要一个更直接的访问方式
                    # 或者在生成答案后手动访问 chain.get_last_run_info() 或类似方法
                    # 但在这里，我们可以假设它在某个 chunk 里一次性提供。

                # 如果 Streamlit 的 spinner 遮挡了流式输出的更新，可以在这里确保最终答案完整显示
                message_placeholder.markdown(full_response)

                # 修改3: 获取检索到的文档。LangChain 0.2.x 的 create_retrieval_chain
                # 在 stream() 方法中，如果需要精确地获取 context，
                # 最好在流结束后通过 response['context'] 访问，
                # 或者在调用 stream() 之前先运行一次检索器。
                # 但更常见的是，context 会在某个特定的 chunk 中被包含。
                # 如果上述 stream 循环中没有捕获到 context，可以考虑如下：
                # (但这会触发额外的检索，或者依赖于 LangChain 版本的 stream() 行为)
                # 目前的代码假设 context 会在流式 chunk 中出现。
                
                # 如果上述 stream 循环中没有捕获到 context，可以尝试：
                # # 模拟检索，但这不会触发 LLM 的流式输出，只是为了获取 context
                # temp_retrieved_docs = setup_rag_system().retriever.invoke(prompt)
                # for doc in temp_retrieved_docs:
                #     source = doc.metadata.get('source', '未知文件')
                #     page = doc.metadata.get('page', '未知页')
                #     file_name = os.path.basename(source)
                #     retrieved_docs_info.append(f"- {file_name} (页码: {page})")


                # 在 Streamlit 中，一个简单的 RAG Chain 可能不会在 stream() 中直接提供 'context' 键
                # 而是会作为整个 response 的一部分。如果 'context' 键在流中未能捕获到，
                # 那么需要调整获取方式。
                # 最简单的方法是，在 stream() 之前先执行一次检索。
                # 或者，如果LLM返回的response中包含context（通常如此），在流完成后再提取。
                # 针对 Streamlit 2.x 和 LangChain 0.2.x 的常见模式：
                # stream 循环结束后，如果需要 context，可以通过 response.get('context') 获取
                # 但因为这里是 stream() 模式，我们只能在 chunk 出现时捕获。
                # 考虑到 Chain.stream() 的目的是流式输出 LLM 的答案，
                # context 通常是作为输入给到 LLM 的，而不是 LLM 的流式输出。
                # 因此，我们暂时移除从 chunk['context'] 获取的逻辑，
                # 而是在流完成之后再尝试获取或通过其他方式提供。
                
                # 暂时注释掉从 chunk 中获取 context 的部分，因为它可能不会出现在每个 chunk 中
                # for chunk in rag_chain.stream({"input": prompt}):
                #    if "context" in chunk:
                #        # ... (此部分暂时不适用或需要更复杂处理)
                
                # 假设 Streamlit 上的 Streamlit UI 刷新可能导致每次都重跑
                # 所以我们还是在 Streamlit 的 session state 里直接存储 context
                # 而不是试图从 stream 里捕获。
                # 为简便起见，这里先不处理流式捕获 context 的复杂性，
                # 而是专注于答案的流式输出。
                
                # 为了能在流式输出后显示来源，我们可以在调用 stream 之前先进行检索，
                # 或者依靠 LLM 最终在答案中包含来源信息。
                # 如果希望从 LangChain 的追踪中获取，那又需要 LangSmith。
                # 一个简化的方法是，在流式输出后，重新进行一次检索来获取来源。
                # 或者，最简单是，如果 LLM 回答包含来源，则依靠 LLM 的生成。
                # 这里为了演示流式，我们暂时不进行来源引用，
                # 因为在 stream 模式下获取 context 比较复杂且依赖具体链实现。

                # 如果确实需要来源，可以在 stream 结束后执行一次普通的 chain.invoke
                # 或者在 stream 之前，先 retriever.invoke(prompt) 得到 documents，
                # 然后将这些 documents 传递给 chain，并保存它们的 metadata。
                
                # 暂时移除流式输出中获取 context 的逻辑，并简化来源引用，
                # 仅在 Streamlit UI 示例中提供一个示意。
                # 如果需要精确来源，考虑将 retrieval_chain 拆分为两步：
                # 1. 检索 (retriever.invoke) -> 获得文档
                # 2. 生成 (document_chain.stream) -> 传入文档和问题

                # 为了简便，我们现在假设 Streamlit 用户的需求是先让流式输出跑起来，
                # 来源引用可以之后再优化。
                # 或者，我们可以直接在 Prompt Template 中要求 LLM 引用来源。

                # 暂时去除来源引用，专注于流式输出。
                # 如果后续需要，可以重新添加，但需要更复杂的 context 获取策略。
                # stream模式下，response['context'] 往往是在流结束后才能完整获取。
                # for chunk in rag_chain.stream({"input": prompt}):
                #    if "context" in chunk and not retrieved_docs_info: # 尝试一次性捕获
                #        for doc in chunk["context"]:
                #             source = doc.metadata.get('source', '未知文件')
                #             page = doc.metadata.get('page', '未知页')
                #             file_name = os.path.basename(source)
                #             retrieved_docs_info.append(f"- {file_name} (页码: {page})")
                #
                # if retrieved_docs_info:
                #     source_str = "\n\n**相关来源:**\n" + "\n".join(sorted(list(set(retrieved_docs_info))))
                #     st.markdown(source_str)
                #     full_response += source_str

            except Exception as e:
                # 错误处理
                error_message = f"抱歉，在生成答案时发生错误：`{e}`\n\n请检查：\n1. Ollama 服务是否正在运行。\n2. `llama3` 模型是否已拉取。\n3. 系统资源（内存、CPU）是否充足。"
                st.error(error_message)
                full_response = error_message
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})