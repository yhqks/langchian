"""

 Created on 2025/1/9  
 @author: yks 
 Path: D:/yhq/python31105/langchain.py
 use:rag
"""
# -*- coding: utf-8 -*-#
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.chat_models import QianfanChatEndpoint
import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory

qianfan_ak = os.getenv("QIANFAN_AK")
qianfan_sk = os.getenv("QIANFAN_SK")
# print(qianfan_sk,qianfan_ak)
# 得到模型：千帆
model = QianfanChatEndpoint(
    model="ERNIE-4.0-8K",
    temperature=0.2,
    timeout=180,
    api_key=qianfan_ak,
    secret_key=qianfan_sk,
)
# 加载数据 互联网
loader = WebBaseLoader(
    web_paths=['https://baike.baidu.com/item/%E5%BE%AE%E5%8D%9A/58302300'],  # 网页链接
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_='contentTab_NyIdV curTab_WQFiV')
    )
)
docs = loader.load()
# 长文本分割
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
res = splitter.split_documents(docs)
# 长文本形成 向量空间
vectorstore = Chroma.from_documents(documents=res, embedding=QianfanEmbeddingsEndpoint())

# 创建检索器
retrive = vectorstore.as_retriever()

# 创建提示
message = '''
你是一个助手，可以来回答关于文章内的问题,
如果你不知道就说不知道，回答内容尽可能简短。\n
{context}
'''
prompt_template = ChatPromptTemplate.from_messages([
    ('human', message),
    MessagesPlaceholder('chat_history'),
    ('human', "{input}")
])

#回答解析器
param = StrOutputParser()


# 创建链
chain1 = create_stuff_documents_chain(model, prompt_template)  # 模型与提示
# chain2 = create_retrieval_chain(retrive, chain1)  # 搜索器与上述连
# resp = chain2.invoke({'input': "请概括上面文章的主要内容"})

# 创建子链
# 子链的提示模板
contextualize_q_system_message = '''
给我一个聊天的历史记录，以及最后一次用户的问题,
同时引用上下文的历史记录，得到一个独立的问题。
当没有聊天记录的时候不回答。
仅仅只需要转述他。
'''
retrive_history_temp=ChatPromptTemplate.from_messages([
    ('human',contextualize_q_system_message),
    MessagesPlaceholder('chat_history'),
    ('human', "{input}"),

])

# 创建子链
history_chain=create_history_aware_retriever(model,retrive,retrive_history_temp)

#保存问答历史记录
store={}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#创建一个父链把前两个链整合
chain=create_retrieval_chain(history_chain,chain1)

result_chain=RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

#第一轮
resp1=result_chain.invoke(
    {'input':"请概括上面文章的主要内容"},
     config={'configurable':{'session_id':'yks'}}
)
print(resp1['answer'])

#第二轮
resp2=result_chain.invoke(
    {'input':"请问功能时间线是怎么样的"},
     config={'configurable':{'session_id':'ks'}}
)
print(resp2['answer'])


#第三轮
resp3=result_chain.invoke(
    {'input':"请问目前用户画像是什么样的"},
     config={'configurable':{'session_id':'yks'}}
)
print(resp3['answer'])



