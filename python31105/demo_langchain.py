"""

 Created on 2025/1/9  
 @author: yks 
 Path: D:/yhq/python31105/langchain.py
 use:run_langchain
"""
import os

from fastapi import FastAPI

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.language_models.chat_models import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

qianfan_ak = os.getenv("QIANFAN_AK")
qianfan_sk = os.getenv("QIANFAN_SK")
#得到模型：千帆
qianfan_chat = QianfanChatEndpoint(
    model="ERNIE-3.5-8K",
    temperature=0.2,
    timeout=60,
    api_key=qianfan_ak,
    secret_key=qianfan_sk,

)



#构造解析器
param=StrOutputParser()


#定义提示模板
prompt_template=ChatPromptTemplate.from_messages([
    ('human','请你作为一个大模型不进行任何检索回答下面的问题'),
    ('human','{text}')
])


#得到链
re=prompt_template|qianfan_chat|param
print(re.invoke({'text': '今天成都的天气怎么样'}))

