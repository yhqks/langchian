"""

 Created on 2025/1/9  
 @author: yks 
 Path: D:/yhq/python31105/langchain.py
 use:rag
"""
from operator import itemgetter


# -*- coding: ISO-8859-1 -*-#

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.chat_models import QianfanChatEndpoint
import os
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


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

# 直接使用大模型和数据库整合 只能生成sql语句 不执行 初始化生成sql语句的chain
chain_test = create_sql_query_chain(model, db)
# print(chain_test.invoke({'question': '请问最多的表里有多少条数据'}))
# 数据库相关提示语句
answer_prompt = '''
给定用户一下的问题、sql语句和sql语句执行之后的结果，回答用户问题。
Question:{question}
SQL Query:{query}
SQL Result:{result}
回答：
'''
prompt_template = ChatPromptTemplate.from_messages([
    ('human',answer_prompt)
])

# 创建一个执行sql语句的工具
execute_sql = QuerySQLDataBaseTool(db=db)
# 初始化生成sql语句的chain
# 第一步生成sql语句
# 2执行sql
chain = (RunnablePassthrough.assign(query=chain_test).assign(
    result=itemgetter('query')| execute_sql) | prompt_template | model | StrOutputParser() )
print(chain.invoke({'question': '请问最多的表里有多少条数据'}))
