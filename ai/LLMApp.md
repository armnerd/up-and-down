# LLM 应用

* 开源大模型是一个土豆 [ Llama ]
* 直接吃也不是不行，但是不太好吃
* 需要洗干净，炸至两面金黄 [ LoRA ]
* 再涂上番茄酱 [ RAG ]
* 最好配上青菜和萝卜等 [ Agent ]

## 0. 大模型微调 [ 改装 ]

### 0.1 RAG

[ 此处按下不表 ]

### 0.2 LoRA

[ 此处按下不表 ]

## 1. 本地大模型 [ 运行 ]

### 1.1  Ollama [ 是真的方便 ]

> https://ollama.com/

```bash
ollama run llama3.1 // meta 的 llama
ollama run qwen2    // 阿里的千问
https://github.com/n4ze3m/page-assist // chrome 的可视化客户端
```

### 1.2  LM Studio [ 可以看到模型推理过程 ]

> https://lmstudio.ai/

```bash
1. 譬如要下载的模型是
https://huggingface.co/lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF/resolve/main/Phi-3.1-mini-4k-instruct-Q5_K_M.gguf

2. 找到模型目录，新建两级目录
cd /Users/you/.cache/lm-studio/models/
mkdir -p lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF

3. 应用外下载 gguf 文件，然后放入刚才的目录里
```

## 2. AI 智能体 [ 包装 ]

* 通常我们的 AI 应用是包含多个模型，需要协作完成工作的
* 这就需要有个框架把它们联合起来形成一个工作流

### 2.1 langchain [ 撸代码 ]

> LangChain is a framework for developing applications powered by large language models (LLMs).

* 一个标准，一个框架
* 一个对 AI 应用开发所需要组件的抽象和规范
* AI 应用开发领域的 k8s

##### Chat models

> Language models that use a sequence of messages as inputs and return chat messages as outputs (as opposed to using plain text). 

##### Prompt templates

> Prompt templates help to translate user input and parameters into instructions for a language model.

##### Document loaders

> These classes load Document objects. LangChain has hundreds of integrations with various data sources to load data from: Slack, Notion, Google Drive, etc.

##### Text splitters

> Once you've loaded documents, you'll often want to transform them to better suit your application. 

##### Vector stores

> One of the most common ways to store and search over unstructured data is to embed it and store the resulting embedding vectors, and then at query time to embed the unstructured query and retrieve the embedding vectors that are 'most similar' to the embedded query.

##### Retrievers

> A retriever is an interface that returns documents given an unstructured query.

##### Agents

> Agents are systems that use an LLM as a reasoning engine to determine which actions to take and what the inputs to those actions should be. The results of those actions can then be fed back into the agent and it determine whether more actions are needed, or whether it is okay to finish.

##### Callbacks

> LangChain provides a callbacks system that allows you to hook into the various stages of your LLM application. This is useful for logging, monitoring, streaming, and other tasks.

##### Quickstart

```bash
pip install langchain -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.2 AutoGen Studio [ UI ]

> AutoGen Studio is an AutoGen-powered AI app (user interface) to help you rapidly prototype AI agents, enhance them with skills, compose them into workflows and interact with them to accomplish tasks

##### Skills [ python 脚本 ] [ 这里可以集成 RAG ]

> These skills are functions tailored to accomplish specific tasks, forming the backbone of your AI agents’ capabilities.

##### Agents [ 依赖 Skills ]

> The Agents tab is where you bring your AI agents to life. Here, you can create as many agents as your project requires, each with unique characteristics and skills.

##### Workflows [ 依赖 Agents ]

> Workflows are the essence of interaction within your AI ecosystem. They define how different agents collaborate to perform tasks.

##### Quickstart

```bash
pip install autogenstudio     // 安装
autogenstudio ui --port 8081  // 运行
http://localhost:8081         // 体验
```
