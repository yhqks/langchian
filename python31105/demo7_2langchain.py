"""

 Created on 2025/1/9  
 @author: yks 
 Path: D:/yhq/python31105/langchain.py
 use:rag
"""
from operator import itemgetter

import psycopg2
# -*- coding: ISO-8859-1 -*-#
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.chat_models import QianfanChatEndpoint
import os
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import chat_agent_executor
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

# 链接数据库
# sqlalchemy
HOSTNAME = '127.0.0.1'
PORT = '5432'
DATABASE = 'databasetest'
USERNAME = 'postgres'
PASSWORD = '123456'
pg_url = f'postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset-utf-8'
# 初始化连接
db = SQLDatabase.from_uri(pg_url)

toolKit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolKit.get_tools()

# 使用agent的整合
system_prompt = """您是一个被设计用来与SQL数据库交互的代理，
给定一个输入问题，创建一个语法正确的S0L语句并执行，然后查看查询结果并返回答案。
    

首先，你应该查看数据库中的表，看看可以查询什么。
不要跳过这一步。
然后查询最相关的表的信息。"""
system_message = SystemMessage(content=system_prompt)
agent_executor = chat_agent_executor.create_tool_calling_executor(model,tools)


resp=agent_executor.invoke({'messages':[HumanMessage(content='请问最多的表里面有多少数据')]})
re=resp['messages']
print(re)
print('+++++++++++++')
print(re[len(re)-1])