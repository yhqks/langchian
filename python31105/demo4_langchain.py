"""

 Created on 2025/1/9  
 @author: yks 
 Path: D:/yhq/python31105/langchain.py
 use:利用检索
"""
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.language_models.chat_models import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

from langgraph.prebuilt import chat_agent_executor

qianfan_ak = os.getenv("QIANFAN_AK")
qianfan_sk = os.getenv("QIANFAN_SK")
# print(qianfan_sk,qianfan_ak)
# 得到模型：千帆
qianfan_chat = QianfanChatEndpoint(
    model="ERNIE-3.5-8K",
    temperature=0.2,
    timeout=60,
    api_key=qianfan_ak,
    secret_key=qianfan_sk,

)
'''

param = StrOutputParser()
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


prompt_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name='my_message')
])
 返回搜索的结果数量max_reults
search = TavilySearchResults(max_results=2)
模型绑定工具
model_Tool = qianfan_chat.bind_tools([search])

chain = model_Tool
res=chain.invoke([HumanMessage(content='北京今天的天气怎么样?')])
print(f'res_toolcalls{res.tool_calls}')
print(f'res_content{res.content}')
print(f'model{res.content}')
print(f'model_tool{res.tool_calls}')

res2=model.invoke([HumanMessage(content='上班如何摸鱼')])
print(f'model{res2.content}')
print(f'model_tool{res2.tool_calls}')
res=chain.invoke([HumanMessage(content='今天天气怎么样')])



do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_message'  # 每次聊天时候发送mas的key

)
config = {'configurable': {'session_id': 'zs'}}  # 当前会话定义sessionid

第一轮
resp = do_message.invoke(
    {
        'my_message': [HumanMessage(content='请你作为一个大模型不进行任何检索回答下面的问题')],
        'language': '中文'
    },
    config=config
)
print(resp)

resp2 = do_message.invoke(
    {
        'my_message': [HumanMessage(content='今天克拉玛依天气怎么样')],
        'language': '中文'
    },
    config=config
)
print(resp2)
'''

search = TavilySearchResults(max_results=2)
tool = [search]
# 创建代理
agent_executor = chat_agent_executor.create_tool_calling_executor(qianfan_chat,tools=tool)
resq=agent_executor.invoke({
    'messages':HumanMessage(content='如何摸鱼？')
})
print(resq['messages'])