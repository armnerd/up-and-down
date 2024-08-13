#  与文档聊天之RAG

> 由表及里

## 何为 RAG

* 大模型的外挂数据库
* 大模型没法实时地跟进最新的动态
* 开源大模型没法把私有数据拿来训练
* 大模型对于一个专门领域的训练数据还是有限

## 过程

* 导入文档 pdf、 markdown 等
* TextSplitter 讲文档切割为小块 chunks
* 转换为向量存入数据库 embeddings

## 1. anything LLM

> https://anythingllm.com

### 1.1 优势

* RAG & long-term memory
* View & summarize documents
* Scrape websites
* Web Search
* SQL Connector
* TTS (text-to-speech) support
* STT (speech-to-text) support
* Vector Databases

### 1.2 例子

> 纯界面实现，当然代码里也是这些步骤和元素

* 首先让 ollama 运行起来备用
* 初始化 anything LLM 选择 ollama 作为模型提供者
* 选择向量数据库与 Embedder
* 创建 workspace 并上传文档
* 开聊

## 2. 从一个 up 主聊起

https://github.com/pixegami/rag-tutorial-v2

### 2.1 ollama embedding

> 文字转向量

Embedding models are models that are trained specifically to generate vector embeddings: long arrays of numbers that represent semantic meaning for a given sequence of text

https://ollama.com/blog/embedding-models

```bash
ollama pull nomic-embed-text

curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Llamas are members of the camelid family"
}'
```

### 2.2 langchain

```python
# 文字转向量
from langchain_community.embeddings.ollama import OllamaEmbeddings
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

# 加载文档
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# 拆分文档
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# 写入 Chroma
from langchain.vectorstores.chroma import Chroma
def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    ...
    db.add_documents(new_chunks, ids=new_chunk_ids)
    db.persist()
    
# 查询 RAG 并请求本地大模型
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
def query_rag(query_text: str):
    # 连接 Chroma
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 搜索 Chroma
    results = db.similarity_search_with_score(query_text, k=5)
    
    # 将 RAG 的返回作为聊天的上下文，加上原本的问题一起向大模型发问
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # 请求本地大模型
    model = Ollama(model="llama3.1")
    response_text = model.invoke(prompt)

    # 处理结果以及涉及到的文档
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text
```

## 3. LlamaIndex

https://www.llamaindex.ai

```python
pip install llama-index
pip install llama-index-llms-ollama
pip install llama-index-embeddings-huggingface

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

documents = SimpleDirectoryReader("data").load_data()

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = Ollama(model="llama3.1", request_timeout=360.0)

index = VectorStoreIndex.from_documents(
    documents,
)

# query
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
```

## 4. RAGatouille

> ColBERT is a fast and accurate retrieval model, enabling scalable BERT-based search over large text collections in tens of milliseconds.

https://github.com/AnswerDotAI/RAGatouille
