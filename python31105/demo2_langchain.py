"""

 Created on 2025/1/9  
 @author: yks 
 Path: D:/yhq/python31105/langchain.py
 use:chatchat
"""
import os

from fastapi import FastAPI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.language_models.chat_models import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langserve import add_routes
import os

qianfan_ak = os.getenv("QIANFAN_AK")
qianfan_sk = os.getenv("QIANFAN_SK")
# print(qianfan_sk,qianfan_ak)
# 得到模型：千帆
qianfan_chat = QianfanChatEndpoint(
    model="ERNIE-3.5-8K",
    temperature=0.2,
    timeout=30,
    api_key=qianfan_ak,
    secret_key=qianfan_sk,

)

# 构造解析器
param = StrOutputParser()

# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages([
    # ('system','你是一个非常乐于助人的助手。用{language}尽可能的回答问题'),
    MessagesPlaceholder(variable_name='my_message')
])

# 得到链
chain = prompt_template | qianfan_chat

# 保存历史记录
store = {}  # 所有用户的聊天记录 key:sessionId value:chathistory


# chain.invoke({'language':'English','text':'不想上班'})


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_message'  # 每次聊天时候发送mas的key

)
config = {'configurable': {'session_id': 'zs'}}  # 当前会话定义sessionid

# 第一轮
resp = do_message.invoke(
    {
        'my_message': [HumanMessage(content='请你作为一个大模型不进行任何检索回答下面的问题')],
        'language': '中文'
    },
    config=config
)
print(resp.content)

# 第二轮
resp2 = do_message.invoke(
    {
        'my_message': [HumanMessage(content='今天北京天气怎么样')],
        'language': '中文'
    },
    config=config
)
print(resp2.content)

# 第三轮 流式输出 每次输出都是一个token
for resp3 in do_message.stream(
        {
            'my_message': [HumanMessage(content='不进行任何检索你是如何知道今天的天气的')],
            'language': '中文'
        },
        config=config
):
    print(resp3.content, end='')
