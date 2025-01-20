"""

 Created on 2025/1/10  
 @author: yks 
 Path: D:/yhq/python31105/demo6.py
 use:hunyuan_tx_bigmodel
"""
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    openai_api_base="https://api.hunyuan.cloud.tencent.com/v1",
    openai_api_key="sk-e4vSCPglZ9D0BbWfKXvNADgV4dCWju5Yc79PxBQSjAmF3iRd",  # app_key
    model_name="hunyuan-pro",  # 模型名称
)
search = TavilySearchResults(max_results=2)
# 模型绑定工具
model = llm.bind_tools([search])

# chain = prompt_template | model | param
res = model.invoke([HumanMessage(content='如何摸鱼')])
print(res)
